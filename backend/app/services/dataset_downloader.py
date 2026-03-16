"""Dataset Downloader Service.

Downloads datasets from external URLs and processes them for use in the project.
Supports common data formats (CSV, Excel, Parquet, JSON) and handles
various download scenarios including redirects and compressed files.

Now includes Selenium-based web scraping as a fallback for sites that require
JavaScript rendering or have complex download flows.
"""
import logging
import os
import uuid
import tempfile
import zipfile
import gzip
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlparse, unquote

import httpx

from app.core.config import get_settings
from app.services.schema_analyzer import SchemaAnalyzer, get_file_type, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class DatasetDownloadError(Exception):
    """Exception raised when dataset download fails."""
    pass


class DatasetDownloader:
    """Service for downloading and processing external datasets."""

    # Common dataset hosting patterns that we can handle
    KNOWN_HOSTS = {
        "kaggle.com": "kaggle",
        "www.kaggle.com": "kaggle",
        "github.com": "github",
        "raw.githubusercontent.com": "github_raw",
        "huggingface.co": "huggingface",
        "archive.ics.uci.edu": "uci",
        "data.world": "dataworld",
        "openml.org": "openml",
    }

    # File extensions we can process
    DOWNLOADABLE_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".zip", ".gz", ".tar.gz"}

    def __init__(self):
        self.settings = get_settings()
        self.upload_dir = Path(self.settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = SchemaAnalyzer()

    def download_dataset(
        self,
        source_url: str,
        project_id: str,
        dataset_name: str,
        timeout: int = 120,
        use_selenium: bool = True,
        expected_columns: Optional[list] = None,
        expected_row_count: Optional[int] = None,
    ) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
        """Download a dataset from a URL and analyze its schema.

        Args:
            source_url: URL to download from
            project_id: UUID of the project (for organizing files)
            dataset_name: Name of the dataset
            timeout: Download timeout in seconds
            use_selenium: Whether to try Selenium-based scraping as fallback
            expected_columns: Optional list of expected column names for validation
            expected_row_count: Optional expected row count for validation

        Returns:
            Tuple of (file_path, config_json, schema_summary)

        Raises:
            DatasetDownloadError: If download or processing fails
        """
        # Parse and validate URL
        parsed_url = urlparse(source_url)
        if not parsed_url.scheme in ("http", "https"):
            raise DatasetDownloadError(f"Invalid URL scheme: {parsed_url.scheme}. Only HTTP/HTTPS supported.")

        # Determine host type for special handling
        host = parsed_url.netloc.lower()
        host_type = self.KNOWN_HOSTS.get(host, "generic")

        # Transform URL if needed (e.g., GitHub blob -> raw)
        download_url = self._transform_url(source_url, host_type)

        # Download to temporary file first
        temp_dir = tempfile.mkdtemp()
        selenium_used = False
        try:
            downloaded_path, content_type = self._download_file(download_url, temp_dir, timeout)

            # Check if we got HTML instead of a data file
            if content_type and "text/html" in content_type.lower():
                # Check if the file is actually HTML
                try:
                    with open(downloaded_path, 'r', encoding='utf-8', errors='ignore') as f:
                        html_content = f.read()
                        first_bytes = html_content[:500].lower()
                        if '<html' in first_bytes or '<!doctype' in first_bytes:
                            # Try to find a direct download link in the HTML
                            download_link = self._extract_download_link(html_content, source_url)
                            if download_link:
                                # Retry with the extracted download link
                                downloaded_path, content_type = self._download_file(download_link, temp_dir, timeout)
                            elif use_selenium:
                                # Fall back to Selenium-based scraping
                                logger.info(f"Falling back to Selenium scraping for {source_url}")
                                downloaded_path = self._download_with_selenium(
                                    source_url, temp_dir, expected_columns, expected_row_count
                                )
                                selenium_used = True
                            else:
                                raise DatasetDownloadError(
                                    f"URL points to a webpage, not a downloadable data file. "
                                    f"The dataset may require manual download from: {source_url}"
                                )
                except UnicodeDecodeError:
                    pass  # Binary file, not HTML

            # Handle compressed files
            extracted_path = self._extract_if_compressed(downloaded_path, temp_dir)

            # Find the actual data file if we extracted an archive
            data_file = self._find_data_file(extracted_path)
            if not data_file:
                # Check file size to give better error
                file_size = downloaded_path.stat().st_size if downloaded_path.exists() else 0
                if file_size < 1000:
                    # Small file - likely an error page or redirect
                    # Try Selenium as last resort
                    if use_selenium and not selenium_used:
                        logger.info(f"Small file detected, trying Selenium for {source_url}")
                        try:
                            downloaded_path = self._download_with_selenium(
                                source_url, temp_dir, expected_columns, expected_row_count
                            )
                            selenium_used = True
                            extracted_path = self._extract_if_compressed(downloaded_path, temp_dir)
                            data_file = self._find_data_file(extracted_path)
                        except Exception as e:
                            logger.warning(f"Selenium fallback failed: {e}")

                if not data_file:
                    raise DatasetDownloadError(
                        f"Downloaded content is too small ({file_size} bytes) to be a valid dataset. "
                        f"The URL may require authentication or manual download: {source_url}"
                    )

            if not data_file:
                raise DatasetDownloadError(
                    f"No supported data file found. Looking for: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
                )

            # Move to permanent location
            file_id = uuid.uuid4()
            file_ext = data_file.suffix.lower()
            safe_filename = f"{file_id}_{self._sanitize_filename(dataset_name)}{file_ext}"
            final_path = self.upload_dir / safe_filename

            shutil.copy2(data_file, final_path)

            # Analyze schema
            try:
                file_type = get_file_type(final_path)
                schema_summary = self.analyzer.analyze_file(final_path)
            except Exception as e:
                # Clean up on analysis failure
                final_path.unlink(missing_ok=True)
                raise DatasetDownloadError(f"Failed to analyze downloaded file: {str(e)}")

            # Validate schema if expected columns provided
            if expected_columns:
                actual_columns = [c['name'] for c in schema_summary.get('columns', [])]
                overlap = set(expected_columns) & set(actual_columns)
                if len(overlap) < len(expected_columns) * 0.3:
                    logger.warning(
                        f"Schema mismatch for {dataset_name}: "
                        f"expected {expected_columns}, got {actual_columns}"
                    )

            # Build config
            config_json = {
                "file_path": str(final_path.absolute()),
                "original_url": source_url,
                "download_url": download_url,
                "original_filename": data_file.name,
                "file_type": file_type,
                "file_size_bytes": final_path.stat().st_size,
                "downloaded": True,
                "selenium_used": selenium_used,
            }

            return final_path, config_json, schema_summary

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _download_with_selenium(
        self,
        url: str,
        temp_dir: str,
        expected_columns: Optional[list] = None,
        expected_row_count: Optional[int] = None,
    ) -> Path:
        """Download a dataset using Selenium browser automation.

        Args:
            url: URL to scrape and download from
            temp_dir: Temporary directory for downloads
            expected_columns: Optional expected column names
            expected_row_count: Optional expected row count

        Returns:
            Path to downloaded file

        Raises:
            DatasetDownloadError: If download fails
        """
        try:
            from app.services.web_scraper import WebScraper, WebScraperError

            expected_schema = None
            if expected_columns or expected_row_count:
                expected_schema = {
                    'columns': expected_columns or [],
                    'row_count': expected_row_count,
                }

            with WebScraper(headless=True, download_dir=temp_dir) as scraper:
                file_path, metadata = scraper.find_and_download(url, expected_schema)

                if file_path:
                    return file_path
                else:
                    error = metadata.get('error', 'Unknown error')
                    raise DatasetDownloadError(f"Selenium download failed: {error}")

        except ImportError as e:
            raise DatasetDownloadError(
                f"Selenium not available. Install with: pip install selenium webdriver-manager. Error: {e}"
            )
        except Exception as e:
            raise DatasetDownloadError(f"Selenium download failed: {e}")

    def _transform_url(self, url: str, host_type: str) -> str:
        """Transform URLs for direct download where possible."""
        if host_type == "github":
            # Convert github.com/user/repo/blob/branch/file to raw URL
            if "/blob/" in url:
                return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        elif host_type == "kaggle":
            # Kaggle datasets need API access, return as-is for now
            # In future, could integrate with Kaggle API
            pass

        return url

    def _download_file(self, url: str, temp_dir: str, timeout: int) -> Tuple[Path, Optional[str]]:
        """Download a file from URL to temporary directory."""
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                response = client.get(url)
                response.raise_for_status()

                # Get filename from Content-Disposition or URL
                filename = self._get_filename_from_response(response, url)
                file_path = Path(temp_dir) / filename

                # Write content
                with open(file_path, "wb") as f:
                    f.write(response.content)

                content_type = response.headers.get("content-type", "")
                return file_path, content_type

        except httpx.TimeoutException:
            raise DatasetDownloadError(f"Download timed out after {timeout} seconds")
        except httpx.HTTPStatusError as e:
            raise DatasetDownloadError(f"HTTP error {e.response.status_code}: {e.response.reason_phrase}")
        except httpx.RequestError as e:
            raise DatasetDownloadError(f"Failed to download: {str(e)}")

    def _get_filename_from_response(self, response: httpx.Response, url: str) -> str:
        """Extract filename from response headers or URL."""
        # Try Content-Disposition header
        content_disp = response.headers.get("content-disposition", "")
        if "filename=" in content_disp:
            parts = content_disp.split("filename=")
            if len(parts) > 1:
                filename = parts[1].strip('"\'')
                if filename:
                    return filename

        # Fall back to URL path
        parsed = urlparse(url)
        path = unquote(parsed.path)
        if path and "/" in path:
            filename = path.rsplit("/", 1)[-1]
            if filename and "." in filename:
                return filename

        # Default filename
        return "downloaded_dataset.csv"

    def _extract_if_compressed(self, file_path: Path, temp_dir: str) -> Path:
        """Extract compressed files if needed."""
        suffix = file_path.suffix.lower()

        if suffix == ".zip":
            extract_dir = Path(temp_dir) / "extracted"
            extract_dir.mkdir(exist_ok=True)
            try:
                with zipfile.ZipFile(file_path, "r") as zf:
                    zf.extractall(extract_dir)
                return extract_dir
            except zipfile.BadZipFile:
                raise DatasetDownloadError("Invalid ZIP file")

        elif suffix == ".gz":
            # Handle .csv.gz, .json.gz, etc.
            new_name = file_path.stem  # Remove .gz
            extracted_path = Path(temp_dir) / new_name
            try:
                with gzip.open(file_path, "rb") as f_in:
                    with open(extracted_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return extracted_path
            except gzip.BadGzipFile:
                raise DatasetDownloadError("Invalid GZIP file")

        return file_path

    def _find_data_file(self, path: Path) -> Optional[Path]:
        """Find a supported data file in a path (file or directory)."""
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                return path
            return None

        # Search directory for supported files
        # Prefer larger files and common data filenames
        candidates = []
        for ext in SUPPORTED_EXTENSIONS:
            candidates.extend(path.rglob(f"*{ext}"))

        if not candidates:
            return None

        # Sort by size (prefer larger files) and common names
        def score_file(f: Path) -> Tuple[int, int]:
            name_lower = f.stem.lower()
            # Prefer files with common data names
            priority = 0
            if any(x in name_lower for x in ["train", "data", "dataset", "main"]):
                priority = 1
            return (priority, f.stat().st_size)

        candidates.sort(key=score_file, reverse=True)
        return candidates[0]

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as a filename."""
        # Remove or replace problematic characters
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        return safe[:100]  # Limit length

    def _extract_download_link(self, html_content: str, base_url: str) -> Optional[str]:
        """Try to extract a direct download link from an HTML page.

        This handles common patterns from dataset hosting sites.
        """
        import re
        from urllib.parse import urljoin

        # Patterns to look for download links (ordered by specificity)
        patterns = [
            # Direct file links with data extensions
            r'href=["\']([^"\']*\.(?:csv|xlsx|xls|json|parquet|zip|gz))["\']',
            # Download buttons/links (common class names)
            r'href=["\']([^"\']+)["\'][^>]*(?:class|id)=["\'][^"\']*download[^"\']*["\']',
            r'(?:class|id)=["\'][^"\']*download[^"\']*["\'][^>]*href=["\']([^"\']+)["\']',
            # Data export/download API endpoints
            r'href=["\']([^"\']*(?:/api/|/download/|/export/|/data/)[^"\']*\.(?:csv|xlsx|json|zip))["\']',
            # NYC Open Data specific pattern
            r'href=["\']([^"\']*api/views/[^"\']+/rows\.csv[^"\']*)["\']',
            # data.gov patterns
            r'href=["\']([^"\']*resources/[^"\']+\.(?:csv|xlsx|json|zip))["\']',
        ]

        # Files to skip (not actual data files)
        skip_files = [
            'manifest.json', 'package.json', 'config.json', 'settings.json',
            '.min.js', '.min.css', 'favicon', 'logo', 'icon'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            for match in matches:
                # Convert relative URLs to absolute
                full_url = urljoin(base_url, match)
                url_lower = full_url.lower()

                # Skip navigation links and non-data files
                if full_url == base_url:
                    continue
                if any(skip in url_lower for skip in ['javascript:', 'mailto:', '#']):
                    continue
                if any(skip in url_lower for skip in skip_files):
                    continue
                # Skip static asset paths
                if any(path in url_lower for path in ['/static/', '/assets/', '/images/', '/fonts/', '/js/', '/css/']):
                    continue

                return full_url

        return None


def download_and_create_data_source(
    db,
    project_id: str,
    source_url: str,
    dataset_name: str,
    licensing: str = "Unknown",
    fit_for_purpose: str = "",
    discovery_metadata: Optional[Dict[str, Any]] = None,
) -> "DataSource":
    """Download a dataset and create a DataSource record.

    This is a convenience function that combines downloading and database creation.

    Args:
        db: SQLAlchemy database session
        project_id: UUID of the project
        source_url: URL to download from
        dataset_name: Name for the dataset
        licensing: License information
        fit_for_purpose: Description of why this dataset is suitable
        discovery_metadata: Optional metadata from discovery process

    Returns:
        Created DataSource object

    Raises:
        DatasetDownloadError: If download fails
    """
    from app.models.data_source import DataSource, DataSourceType

    downloader = DatasetDownloader()

    # Download and analyze
    file_path, config_json, schema_summary = downloader.download_dataset(
        source_url=source_url,
        project_id=str(project_id),
        dataset_name=dataset_name,
    )

    # Add extra metadata to config
    config_json["licensing"] = licensing
    config_json["fit_for_purpose"] = fit_for_purpose
    config_json["source_url"] = source_url

    if discovery_metadata:
        config_json["discovered_from"] = discovery_metadata

    # Create DataSource record
    # Use FILE_UPLOAD type since we now have the actual file
    data_source = DataSource(
        project_id=project_id,
        name=dataset_name,
        type=DataSourceType.FILE_UPLOAD,  # Changed from EXTERNAL_DATASET since we downloaded it
        config_json=config_json,
        schema_summary=schema_summary,
    )

    db.add(data_source)
    db.flush()

    return data_source
