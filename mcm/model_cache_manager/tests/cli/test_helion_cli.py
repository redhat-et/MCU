"""
Integration tests for CLI with Helion mode support.
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path
from typer.testing import CliRunner

from model_cache_manager.cli.main import app


class TestHelionCLI(unittest.TestCase):
    """Test suite for CLI Helion mode functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.helion_cache_dir = self.temp_dir / "helion_cache"
        self.helion_cache_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("model_cache_manager.cli.main.IndexService")
    def test_index_command_explicit_helion_mode(self, mock_index_service):
        """Test index command with explicit helion mode."""
        mock_svc = MagicMock()
        mock_svc.reindex.return_value = (2, 0)
        mock_svc.repo.root = self.helion_cache_dir
        mock_svc.cache_dir = self.helion_cache_dir
        mock_index_service.return_value = mock_svc

        result = self.runner.invoke(app, [
            "index",
            "--mode", "helion",
            "--cache-dir", str(self.helion_cache_dir),
        ])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Starting indexing process", result.stdout)
        mock_index_service.assert_called_once_with(
            cache_dir=self.helion_cache_dir,
            mode="helion",
        )

    @patch("model_cache_manager.cli.cli_helpers.detect_cache_mode")
    @patch("model_cache_manager.cli.main.IndexService")
    def test_index_command_auto_detection_helion(self, mock_index_service, mock_detect):
        """Test index command with auto-detection detecting helion."""
        mock_detect.return_value = "helion"
        mock_svc = MagicMock()
        mock_svc.reindex.return_value = (1, 0)
        mock_svc.repo.root = self.helion_cache_dir
        mock_svc.cache_dir = self.helion_cache_dir
        mock_index_service.return_value = mock_svc

        result = self.runner.invoke(app, [
            "index",
            "--cache-dir", str(self.helion_cache_dir),
        ])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Auto-detected cache mode: helion", result.stdout)
        mock_detect.assert_called_once_with(self.helion_cache_dir)
        mock_index_service.assert_called_once_with(
            cache_dir=self.helion_cache_dir,
            mode="helion",
        )

    @patch("model_cache_manager.cli.main.ensure_db")
    @patch("model_cache_manager.cli.main.SearchService")
    def test_list_command_helion_mode(self, mock_search_service, mock_ensure_db):
        """Test list command with helion mode."""
        mock_ensure_db.return_value = None
        mock_svc = MagicMock()
        mock_svc.search.return_value = []
        mock_svc.close.return_value = None
        mock_search_service.return_value = mock_svc

        result = self.runner.invoke(app, [
            "list",
            "--mode", "helion",
        ])

        self.assertEqual(result.exit_code, 0)
        mock_search_service.assert_called_once()
        call_args = mock_search_service.call_args
        self.assertEqual(call_args[1]["mode"], "helion")

    @patch("model_cache_manager.cli.main.ensure_db")
    @patch("model_cache_manager.cli.main.PruningService")
    def test_prune_command_helion_mode(self, mock_prune_service, mock_ensure_db):
        """Test prune command with helion mode."""
        mock_ensure_db.return_value = None
        mock_svc = MagicMock()
        mock_svc.prune.return_value = MagicMock(pruned=1, reclaimed=0.5)
        mock_svc.close.return_value = None
        mock_prune_service.return_value = mock_svc

        result = self.runner.invoke(app, [
            "prune",
            "--mode", "helion",
            "--yes",
        ])

        self.assertEqual(result.exit_code, 0)
        mock_prune_service.assert_called_once_with(
            cache_dir=None,
            mode="helion",
        )

    @patch("model_cache_manager.cli.main.IndexService")
    def test_index_command_helion_file_not_found(self, mock_index_service):
        """Test index command when helion cache directory doesn't exist."""
        mock_index_service.side_effect = FileNotFoundError(
            "Helion cache directory not found"
        )

        result = self.runner.invoke(app, [
            "index",
            "--mode", "helion",
            "--cache-dir", "/nonexistent/path",
        ])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Helion cache directory not found", result.stdout)


if __name__ == "__main__":
    unittest.main()
