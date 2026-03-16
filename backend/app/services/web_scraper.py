"""Web Scraper Service using Selenium.

Provides browser automation for discovering and downloading datasets from web pages.
Handles JavaScript-rendered content, finding download links, and verifying data files.
"""
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebScraperError(Exception):
    """Exception raised when web scraping fails."""
    pass


class WebScraper:
    """Selenium-based web scraper for dataset discovery and download."""

    # Known dataset hosting sites and their download patterns
    SITE_PATTERNS = {
        "kaggle.com": {
            "download_selectors": [
                "a[href*='/download']",
                "button[data-testid='download-button']",
                "a.download-button",
            ],
            "requires_login": True,
            "api_available": True,
        },
        "github.com": {
            "download_selectors": [
                "a[href*='/raw/']",
                "a[data-skip-pjax='']",
                "a[href*='?raw=true']",
            ],
            "requires_login": False,
        },
        "huggingface.co": {
            "download_selectors": [
                "a[href*='/resolve/']",
                "a[download]",
                "a[href*='parquet']",
                "a[href*='.csv']",
            ],
            "requires_login": False,
        },
        "archive.ics.uci.edu": {
            "download_selectors": [
                "a[href*='.data']",
                "a[href*='.csv']",
                "a[href*='.zip']",
            ],
            "requires_login": False,
        },
        "openml.org": {
            "download_selectors": [
                "a[href*='/get_csv']",
                "a[href*='/download']",
            ],
            "requires_login": False,
        },
        "data.gov": {
            "download_selectors": [
                "a[href*='.csv']",
                "a[href*='/download']",
                "a.btn-download",
            ],
            "requires_login": False,
        },
    }

    # Data file extensions we're looking for
    DATA_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json', '.parquet', '.zip', '.gz', '.tsv', '.data'}

    def __init__(self, headless: bool = True, download_dir: Optional[str] = None):
        """Initialize the web scraper.

        Args:
            headless: Run browser in headless mode (no visible window)
            download_dir: Directory to save downloaded files
        """
        self.headless = headless
        self.download_dir = download_dir or tempfile.mkdtemp(prefix="web_scraper_")
        self._driver = None
        self._initialized = False

    def _init_driver(self):
        """Initialize Selenium WebDriver lazily."""
        if self._initialized:
            return

        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager

            options = Options()
            if self.headless:
                options.add_argument("--headless=new")

            # Common options for stability
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-popup-blocking")

            # Set download preferences
            prefs = {
                "download.default_directory": self.download_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True,
                "plugins.always_open_pdf_externally": True,
            }
            options.add_experimental_option("prefs", prefs)

            # User agent to avoid bot detection
            options.add_argument(
                "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

            service = Service(ChromeDriverManager().install())
            self._driver = webdriver.Chrome(service=service, options=options)
            self._driver.set_page_load_timeout(60)
            self._initialized = True
            logger.info("Selenium WebDriver initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            raise WebScraperError(f"Could not initialize browser: {e}")

    def close(self):
        """Close the browser and clean up resources."""
        if self._driver:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None
            self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def scrape_page(self, url: str, wait_time: int = 5) -> Dict[str, Any]:
        """Scrape a webpage and extract relevant information.

        Args:
            url: URL to scrape
            wait_time: Time to wait for JavaScript rendering (seconds)

        Returns:
            Dictionary with page info including:
            - title: Page title
            - download_links: List of potential download links
            - data_links: Links that appear to be data files
            - page_html: Full HTML content
        """
        self._init_driver()

        try:
            logger.info(f"Navigating to: {url}")
            self._driver.get(url)

            # Wait for JavaScript to render
            time.sleep(wait_time)

            # Get page source after JS rendering
            html = self._driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            # Extract page title
            title = soup.title.string if soup.title else ""

            # Find all links
            all_links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                text = a.get_text(strip=True)
                full_url = urljoin(url, href)
                all_links.append({
                    'url': full_url,
                    'text': text,
                    'classes': a.get('class', []),
                })

            # Identify download links
            download_links = self._find_download_links(soup, url)

            # Identify data file links
            data_links = self._find_data_file_links(all_links)

            return {
                'url': url,
                'title': title,
                'download_links': download_links,
                'data_links': data_links,
                'all_links': all_links,
                'page_html': html,
            }

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            raise WebScraperError(f"Failed to scrape page: {e}")

    def _find_download_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Find links that appear to be download buttons/links."""
        download_links = []

        # Get site-specific selectors
        parsed = urlparse(base_url)
        host = parsed.netloc.lower().replace('www.', '')
        site_config = None
        for site_host, config in self.SITE_PATTERNS.items():
            if site_host in host:
                site_config = config
                break

        # Try site-specific selectors first
        if site_config:
            for selector in site_config.get('download_selectors', []):
                try:
                    for elem in soup.select(selector):
                        href = elem.get('href')
                        if href:
                            download_links.append({
                                'url': urljoin(base_url, href),
                                'text': elem.get_text(strip=True),
                                'source': 'site_specific',
                            })
                except Exception:
                    continue

        # Generic download patterns
        download_patterns = [
            # Links with download-related text
            lambda a: any(word in (a.get_text() or '').lower() for word in
                         ['download', 'export', 'get data', 'csv', 'excel', 'json']),
            # Links with download-related classes
            lambda a: any('download' in c.lower() for c in (a.get('class') or [])),
            # Links with download-related href
            lambda a: any(word in (a.get('href') or '').lower() for word in
                         ['/download', '/export', '/get_csv', 'format=csv']),
        ]

        for a in soup.find_all('a', href=True):
            for pattern in download_patterns:
                if pattern(a):
                    href = a['href']
                    download_links.append({
                        'url': urljoin(base_url, href),
                        'text': a.get_text(strip=True),
                        'source': 'pattern_match',
                    })
                    break

        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in download_links:
            if link['url'] not in seen:
                seen.add(link['url'])
                unique_links.append(link)

        return unique_links

    def _find_data_file_links(self, all_links: List[Dict]) -> List[Dict[str, str]]:
        """Find links that point directly to data files."""
        data_links = []

        for link in all_links:
            url = link['url']
            parsed = urlparse(url)
            path_lower = parsed.path.lower()

            # Check if URL ends with a data file extension
            for ext in self.DATA_EXTENSIONS:
                if path_lower.endswith(ext):
                    data_links.append({
                        'url': url,
                        'text': link['text'],
                        'extension': ext,
                    })
                    break

        return data_links

    def find_and_download(
        self,
        url: str,
        expected_schema: Optional[Dict[str, Any]] = None,
        max_attempts: int = 3,
    ) -> Tuple[Optional[Path], Dict[str, Any]]:
        """Navigate to a URL, find the download link, and download the file.

        Args:
            url: Starting URL to scrape
            expected_schema: Optional schema info to validate against
            max_attempts: Maximum number of links to try

        Returns:
            Tuple of (downloaded_file_path, metadata)
        """
        self._init_driver()

        # First, scrape the page
        page_info = self.scrape_page(url)

        # Collect candidate download URLs
        candidates = []

        # Priority 1: Direct data file links
        for link in page_info.get('data_links', []):
            candidates.append(link['url'])

        # Priority 2: Download button links
        for link in page_info.get('download_links', []):
            if link['url'] not in candidates:
                candidates.append(link['url'])

        if not candidates:
            logger.warning(f"No download candidates found on {url}")
            return None, {'error': 'No download links found', 'page_info': page_info}

        # Try downloading from candidates
        for i, download_url in enumerate(candidates[:max_attempts]):
            logger.info(f"Attempting download from: {download_url}")

            try:
                downloaded_file = self._download_via_browser(download_url)
                if downloaded_file:
                    # Validate the downloaded file
                    is_valid, validation_info = self._validate_downloaded_file(
                        downloaded_file, expected_schema
                    )

                    if is_valid:
                        return downloaded_file, {
                            'source_url': url,
                            'download_url': download_url,
                            'validation': validation_info,
                        }
                    else:
                        logger.warning(f"Downloaded file failed validation: {validation_info}")
                        # Clean up invalid file
                        downloaded_file.unlink(missing_ok=True)

            except Exception as e:
                logger.warning(f"Download attempt {i+1} failed: {e}")
                continue

        return None, {'error': 'All download attempts failed', 'tried_urls': candidates[:max_attempts]}

    def _download_via_browser(self, url: str, timeout: int = 60) -> Optional[Path]:
        """Download a file by navigating to its URL.

        Args:
            url: URL to download
            timeout: Maximum time to wait for download (seconds)

        Returns:
            Path to downloaded file, or None if download failed
        """
        # Clear download directory
        download_dir = Path(self.download_dir)
        existing_files = set(download_dir.glob('*'))

        # Navigate to the download URL
        self._driver.get(url)

        # Wait for download to complete
        start_time = time.time()
        while time.time() - start_time < timeout:
            current_files = set(download_dir.glob('*'))
            new_files = current_files - existing_files

            # Filter out temporary/incomplete downloads
            completed_files = [
                f for f in new_files
                if not f.suffix.lower() in {'.crdownload', '.tmp', '.part'}
            ]

            if completed_files:
                # Return the newest completed file
                return max(completed_files, key=lambda f: f.stat().st_mtime)

            time.sleep(1)

        logger.warning(f"Download timed out after {timeout} seconds")
        return None

    def _validate_downloaded_file(
        self,
        file_path: Path,
        expected_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate that a downloaded file is a valid data file.

        Args:
            file_path: Path to the downloaded file
            expected_schema: Optional expected schema to validate against

        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'extension': file_path.suffix.lower(),
        }

        # Check file size (should be at least 100 bytes for any real data)
        if validation_info['file_size'] < 100:
            validation_info['error'] = 'File too small'
            return False, validation_info

        # Check extension
        if file_path.suffix.lower() not in self.DATA_EXTENSIONS:
            # Try to detect file type from content
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(100)

                # Check for common file signatures
                if header.startswith(b'PK'):  # ZIP file
                    validation_info['detected_type'] = 'zip'
                elif header.startswith(b'\x1f\x8b'):  # GZIP file
                    validation_info['detected_type'] = 'gzip'
                elif b',' in header and b'\n' in header:  # Likely CSV
                    validation_info['detected_type'] = 'csv'
                elif header.startswith(b'{') or header.startswith(b'['):  # JSON
                    validation_info['detected_type'] = 'json'
                elif b'<html' in header.lower() or b'<!doctype' in header.lower():
                    validation_info['error'] = 'Downloaded HTML instead of data file'
                    return False, validation_info
            except Exception as e:
                validation_info['error'] = f'Could not read file: {e}'
                return False, validation_info

        # Try to load and validate with schema analyzer
        try:
            from app.services.schema_analyzer import SchemaAnalyzer
            analyzer = SchemaAnalyzer()
            schema = analyzer.analyze_file(file_path)

            validation_info['detected_schema'] = {
                'row_count': schema.get('row_count', 0),
                'column_count': schema.get('column_count', 0),
                'columns': [c['name'] for c in schema.get('columns', [])],
            }

            # Validate against expected schema if provided
            if expected_schema:
                expected_cols = set(expected_schema.get('columns', []))
                actual_cols = set(validation_info['detected_schema']['columns'])

                # Check column overlap
                overlap = expected_cols & actual_cols
                if len(overlap) < len(expected_cols) * 0.5:
                    validation_info['warning'] = (
                        f"Schema mismatch: expected columns {expected_cols}, "
                        f"got {actual_cols}"
                    )

            return True, validation_info

        except Exception as e:
            validation_info['error'] = f'Schema analysis failed: {e}'
            return False, validation_info

    def click_download_button(self, url: str, button_selector: str) -> Optional[Path]:
        """Click a download button on a page and wait for the file.

        Args:
            url: Page URL to navigate to
            button_selector: CSS selector for the download button

        Returns:
            Path to downloaded file, or None if download failed
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        self._init_driver()

        # Clear download directory
        download_dir = Path(self.download_dir)
        existing_files = set(download_dir.glob('*'))

        try:
            self._driver.get(url)
            time.sleep(3)  # Wait for page load

            # Find and click the download button
            wait = WebDriverWait(self._driver, 10)
            button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, button_selector)))
            button.click()

            # Wait for download
            timeout = 60
            start_time = time.time()
            while time.time() - start_time < timeout:
                current_files = set(download_dir.glob('*'))
                new_files = current_files - existing_files

                completed_files = [
                    f for f in new_files
                    if not f.suffix.lower() in {'.crdownload', '.tmp', '.part'}
                ]

                if completed_files:
                    return max(completed_files, key=lambda f: f.stat().st_mtime)

                time.sleep(1)

        except Exception as e:
            logger.error(f"Failed to click download button: {e}")

        return None


def scrape_and_verify_dataset(
    url: str,
    expected_name: str,
    expected_columns: Optional[List[str]] = None,
    expected_row_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Scrape a URL, find and download the dataset, and verify it matches expectations.

    This is the main entry point for the web scraping functionality.

    Args:
        url: URL to scrape
        expected_name: Expected dataset name (for logging)
        expected_columns: Optional list of expected column names
        expected_row_count: Optional expected row count

    Returns:
        Dictionary with:
        - success: Boolean indicating if download succeeded
        - file_path: Path to downloaded file (if successful)
        - validation: Validation results
        - error: Error message (if failed)
    """
    expected_schema = None
    if expected_columns or expected_row_count:
        expected_schema = {
            'columns': expected_columns or [],
            'row_count': expected_row_count,
        }

    try:
        with WebScraper(headless=True) as scraper:
            file_path, metadata = scraper.find_and_download(url, expected_schema)

            if file_path:
                return {
                    'success': True,
                    'file_path': str(file_path),
                    'download_url': metadata.get('download_url'),
                    'validation': metadata.get('validation', {}),
                }
            else:
                return {
                    'success': False,
                    'error': metadata.get('error', 'Download failed'),
                    'details': metadata,
                }

    except Exception as e:
        logger.error(f"Web scraping failed for {url}: {e}")
        return {
            'success': False,
            'error': str(e),
        }
