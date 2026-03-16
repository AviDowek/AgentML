"""Tests for the dataset downloader service."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import httpx

from app.services.dataset_downloader import (
    DatasetDownloader,
    DatasetDownloadError,
)


class TestDatasetDownloader:
    """Tests for DatasetDownloader class."""

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        downloader = DatasetDownloader()

        assert downloader._sanitize_filename("normal_file") == "normal_file"
        assert downloader._sanitize_filename("file with spaces") == "file_with_spaces"
        assert downloader._sanitize_filename("file/with\\slashes") == "file_with_slashes"
        assert downloader._sanitize_filename("file.csv") == "file.csv"
        assert downloader._sanitize_filename("file-name_123") == "file-name_123"

    def test_transform_github_url(self):
        """Test GitHub URL transformation from blob to raw."""
        downloader = DatasetDownloader()

        # GitHub blob URL should be converted to raw
        blob_url = "https://github.com/user/repo/blob/main/data.csv"
        transformed = downloader._transform_url(blob_url, "github")
        assert transformed == "https://raw.githubusercontent.com/user/repo/main/data.csv"

        # Raw URL should stay the same
        raw_url = "https://raw.githubusercontent.com/user/repo/main/data.csv"
        transformed = downloader._transform_url(raw_url, "github_raw")
        assert transformed == raw_url

    def test_get_filename_from_response_content_disposition(self):
        """Test extracting filename from Content-Disposition header."""
        downloader = DatasetDownloader()

        response = MagicMock()
        response.headers = {"content-disposition": 'attachment; filename="test_data.csv"'}

        filename = downloader._get_filename_from_response(response, "https://example.com/data")
        assert filename == "test_data.csv"

    def test_get_filename_from_response_url_fallback(self):
        """Test extracting filename from URL when no header."""
        downloader = DatasetDownloader()

        response = MagicMock()
        response.headers = {}

        filename = downloader._get_filename_from_response(
            response,
            "https://example.com/datasets/housing_prices.csv"
        )
        assert filename == "housing_prices.csv"

    def test_get_filename_default(self):
        """Test default filename when nothing else works."""
        downloader = DatasetDownloader()

        response = MagicMock()
        response.headers = {}

        filename = downloader._get_filename_from_response(
            response,
            "https://example.com/"
        )
        assert filename == "downloaded_dataset.csv"

    @patch.object(DatasetDownloader, '_download_file')
    @patch.object(DatasetDownloader, '_extract_if_compressed')
    @patch.object(DatasetDownloader, '_find_data_file')
    def test_download_dataset_success(
        self,
        mock_find,
        mock_extract,
        mock_download
    ):
        """Test successful dataset download."""
        downloader = DatasetDownloader()

        # Create a real temp file for testing
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"col1,col2\n1,2\n3,4\n")
            temp_path = Path(f.name)

        try:
            mock_download.return_value = (temp_path, "text/csv")
            mock_extract.return_value = temp_path
            mock_find.return_value = temp_path

            # Mock the schema analyzer
            with patch('app.services.dataset_downloader.SchemaAnalyzer') as mock_analyzer:
                mock_analyzer.return_value.analyze_file.return_value = {
                    "row_count": 2,
                    "columns": [{"name": "col1", "dtype": "int64"}, {"name": "col2", "dtype": "int64"}],
                }

                file_path, config, schema = downloader.download_dataset(
                    source_url="https://example.com/data.csv",
                    project_id="test-project",
                    dataset_name="Test Dataset",
                )

                assert config["downloaded"] is True
                assert config["original_url"] == "https://example.com/data.csv"
                assert schema["row_count"] == 2
        finally:
            # Cleanup
            temp_path.unlink(missing_ok=True)

    def test_download_dataset_invalid_url_scheme(self):
        """Test that invalid URL schemes raise an error."""
        downloader = DatasetDownloader()

        with pytest.raises(DatasetDownloadError) as exc_info:
            downloader.download_dataset(
                source_url="ftp://example.com/data.csv",
                project_id="test",
                dataset_name="Test",
            )

        assert "Invalid URL scheme" in str(exc_info.value)

    def test_find_data_file_single_file(self):
        """Test finding a data file when given a single file path."""
        downloader = DatasetDownloader()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = downloader._find_data_file(temp_path)
            assert result == temp_path
        finally:
            temp_path.unlink()

    def test_find_data_file_in_directory(self):
        """Test finding a data file in a directory."""
        downloader = DatasetDownloader()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a CSV file
            csv_file = temp_path / "data.csv"
            csv_file.write_text("a,b\n1,2\n")

            # Create a non-data file
            txt_file = temp_path / "readme.txt"
            txt_file.write_text("readme")

            result = downloader._find_data_file(temp_path)
            assert result == csv_file

    def test_find_data_file_prefers_larger(self):
        """Test that larger data files are preferred."""
        downloader = DatasetDownloader()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a small CSV
            small_csv = temp_path / "small.csv"
            small_csv.write_text("a\n1\n")

            # Create a larger CSV
            large_csv = temp_path / "large.csv"
            large_csv.write_text("a,b,c\n" + "1,2,3\n" * 100)

            result = downloader._find_data_file(temp_path)
            assert result == large_csv

    def test_find_data_file_prefers_train(self):
        """Test that files with 'train' in name are preferred."""
        downloader = DatasetDownloader()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create CSVs of same size
            other_csv = temp_path / "other.csv"
            other_csv.write_text("a,b\n1,2\n")

            train_csv = temp_path / "train.csv"
            train_csv.write_text("a,b\n1,2\n")

            result = downloader._find_data_file(temp_path)
            assert result == train_csv

    def test_extract_download_link_csv(self):
        """Test extracting CSV download link from HTML."""
        downloader = DatasetDownloader()

        html = '''
        <html>
        <body>
            <a href="/data/dataset.csv">Download CSV</a>
        </body>
        </html>
        '''
        result = downloader._extract_download_link(html, "https://example.com/page")
        assert result == "https://example.com/data/dataset.csv"

    def test_extract_download_link_download_button(self):
        """Test extracting link from download button."""
        downloader = DatasetDownloader()

        html = '''
        <html>
        <body>
            <a href="/files/data.zip" class="btn-download">Download</a>
        </body>
        </html>
        '''
        result = downloader._extract_download_link(html, "https://example.com/")
        assert result == "https://example.com/files/data.zip"

    def test_extract_download_link_nyc_open_data(self):
        """Test extracting NYC Open Data API link."""
        downloader = DatasetDownloader()

        html = '''
        <html>
        <body>
            <a href="https://data.cityofnewyork.us/api/views/abc123/rows.csv?accessType=DOWNLOAD">Export</a>
        </body>
        </html>
        '''
        result = downloader._extract_download_link(html, "https://data.cityofnewyork.us/dataset/test")
        assert "api/views/abc123/rows.csv" in result

    def test_extract_download_link_none_found(self):
        """Test that None is returned when no download link found."""
        downloader = DatasetDownloader()

        html = '''
        <html>
        <body>
            <p>This page has no download links</p>
            <a href="/about">About Us</a>
        </body>
        </html>
        '''
        result = downloader._extract_download_link(html, "https://example.com/")
        assert result is None
