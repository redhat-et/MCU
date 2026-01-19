"""
Unit tests for utility functions in utils.py module.
"""

import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile
import shutil

from model_cache_manager.utils.utils import get_temp_extraction_dir


class TestGetTempExtractionDir(unittest.TestCase):
    """Tests for get_temp_extraction_dir function."""

    def test_creates_directory_if_not_exists(self):
        """Test that the function creates the directory if it doesn't exist."""
        with patch("getpass.getuser", return_value="testuser"):
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                result = get_temp_extraction_dir()

                # Verify mkdir was called with correct arguments
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                self.assertEqual(result, Path("/tmp/mcm_testuser"))

    def test_returns_existing_directory(self):
        """Test that the function returns existing directory without error."""
        # Create actual temp directory for this test
        temp_test_dir = Path(tempfile.mkdtemp())
        try:
            with patch("getpass.getuser", return_value="testuser"):
                with patch("pathlib.Path.__truediv__", return_value=temp_test_dir):
                    result = get_temp_extraction_dir()

                    # Should return the path even if it already exists
                    self.assertTrue(result.exists())
        finally:
            shutil.rmtree(temp_test_dir, ignore_errors=True)

    def test_uses_current_username(self):
        """Test that the function uses the current user's username."""
        with patch("getpass.getuser", return_value="john_doe"):
            result = get_temp_extraction_dir()

            self.assertEqual(result, Path("/tmp/mcm_john_doe"))
            self.assertIn("john_doe", str(result))

    def test_path_format(self):
        """Test that the returned path has the correct format."""
        with patch("getpass.getuser", return_value="alice"):
            result = get_temp_extraction_dir()

            self.assertEqual(result.parent, Path("/tmp"))
            self.assertEqual(result.name, "mcm_alice")
            self.assertTrue(str(result).startswith("/tmp/mcm_"))

    def test_different_users_get_different_dirs(self):
        """Test that different users get different directories."""
        with patch("getpass.getuser", return_value="user1"):
            result1 = get_temp_extraction_dir()

        with patch("getpass.getuser", return_value="user2"):
            result2 = get_temp_extraction_dir()

        self.assertNotEqual(result1, result2)
        self.assertEqual(result1, Path("/tmp/mcm_user1"))
        self.assertEqual(result2, Path("/tmp/mcm_user2"))

    def test_idempotent(self):
        """Test that calling the function multiple times returns the same path."""
        with patch("getpass.getuser", return_value="testuser"):
            result1 = get_temp_extraction_dir()
            result2 = get_temp_extraction_dir()

            self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
