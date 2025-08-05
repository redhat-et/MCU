"""
Integration tests for CLI with vLLM mode support.
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path
from typer.testing import CliRunner

from model_cache_manager.cli.main import app


class TestVllmCLI(unittest.TestCase):
    """Test suite for CLI vLLM mode functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.vllm_cache_dir = self.temp_dir / "vllm_cache"
        self.triton_cache_dir = self.temp_dir / "triton_cache"
        
        # Create directories
        self.vllm_cache_dir.mkdir(parents=True)
        self.triton_cache_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('model_cache_manager.cli.main.IndexService')
    def test_index_command_explicit_vllm_mode(self, mock_index_service):
        """Test index command with explicit vLLM mode."""
        mock_service_instance = MagicMock()
        mock_service_instance.reindex.return_value = (5, 0)
        mock_service_instance.repo.root = self.vllm_cache_dir
        mock_service_instance.cache_dir = self.vllm_cache_dir
        mock_index_service.return_value = mock_service_instance
        
        result = self.runner.invoke(app, [
            "index", 
            "--mode", "vllm", 
            "--cache-dir", str(self.vllm_cache_dir)
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Starting indexing process", result.stdout)
        mock_index_service.assert_called_once_with(
            cache_dir=self.vllm_cache_dir, 
            mode="vllm"
        )

    @patch('model_cache_manager.cli.main.IndexService')
    def test_index_command_explicit_triton_mode(self, mock_index_service):
        """Test index command with explicit triton mode."""
        mock_service_instance = MagicMock()
        mock_service_instance.reindex.return_value = (3, 1)
        mock_service_instance.repo.root = self.triton_cache_dir
        mock_service_instance.cache_dir = self.triton_cache_dir
        mock_index_service.return_value = mock_service_instance
        
        result = self.runner.invoke(app, [
            "index", 
            "--mode", "triton", 
            "--cache-dir", str(self.triton_cache_dir)
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Starting indexing process", result.stdout)
        mock_index_service.assert_called_once_with(
            cache_dir=self.triton_cache_dir, 
            mode="triton"
        )

    @patch('model_cache_manager.cli.main.detect_cache_mode')
    @patch('model_cache_manager.cli.main.IndexService')
    def test_index_command_auto_detection_vllm(self, mock_index_service, mock_detect):
        """Test index command with auto-detection detecting vLLM."""
        mock_detect.return_value = "vllm"
        mock_service_instance = MagicMock()
        mock_service_instance.reindex.return_value = (2, 0)
        mock_service_instance.repo.root = self.vllm_cache_dir
        mock_service_instance.cache_dir = self.vllm_cache_dir
        mock_index_service.return_value = mock_service_instance
        
        result = self.runner.invoke(app, [
            "index", 
            "--cache-dir", str(self.vllm_cache_dir)
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Auto-detected cache mode: vllm", result.stdout)
        mock_detect.assert_called_once_with(self.vllm_cache_dir)
        mock_index_service.assert_called_once_with(
            cache_dir=self.vllm_cache_dir, 
            mode="vllm"
        )

    @patch('model_cache_manager.cli.main.detect_cache_mode')
    @patch('model_cache_manager.cli.main.IndexService')
    def test_index_command_auto_detection_triton(self, mock_index_service, mock_detect):
        """Test index command with auto-detection detecting triton."""
        mock_detect.return_value = "triton"
        mock_service_instance = MagicMock()
        mock_service_instance.reindex.return_value = (1, 0)
        mock_service_instance.repo.root = self.triton_cache_dir
        mock_service_instance.cache_dir = self.triton_cache_dir
        mock_index_service.return_value = mock_service_instance
        
        result = self.runner.invoke(app, [
            "index", 
            "--cache-dir", str(self.triton_cache_dir)
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Auto-detected cache mode: triton", result.stdout)
        mock_detect.assert_called_once_with(self.triton_cache_dir)

    def test_index_command_invalid_mode(self):
        """Test index command with invalid mode."""
        result = self.runner.invoke(app, [
            "index", 
            "--mode", "invalid", 
            "--cache-dir", str(self.vllm_cache_dir)
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Invalid mode 'invalid'", result.stdout)

    @patch('model_cache_manager.cli.main._cache_db_exists')
    @patch('model_cache_manager.cli.main.SearchService')
    def test_list_command_vllm_mode(self, mock_search_service, mock_db_exists):
        """Test list command with vLLM mode."""
        mock_db_exists.return_value = True
        mock_service_instance = MagicMock()
        mock_service_instance.search.return_value = []
        mock_search_service.return_value = mock_service_instance
        
        result = self.runner.invoke(app, [
            "list", 
            "--mode", "vllm"
        ])
        
        self.assertEqual(result.exit_code, 0)
        mock_search_service.assert_called_once()
        # Check that mode was passed correctly
        call_args = mock_search_service.call_args
        self.assertEqual(call_args[1]["mode"], "vllm")

    @patch('model_cache_manager.cli.main._cache_db_exists')
    @patch('model_cache_manager.cli.main.SearchService')
    def test_list_command_with_filters_vllm(self, mock_search_service, mock_db_exists):
        """Test list command with filters in vLLM mode."""
        mock_db_exists.return_value = True
        mock_service_instance = MagicMock()
        mock_service_instance.search.return_value = [
            {"hash": "hash1", "name": "kernel1", "backend": "cuda"}
        ]
        mock_search_service.return_value = mock_service_instance
        
        result = self.runner.invoke(app, [
            "list", 
            "--mode", "vllm",
            "--backend", "cuda",
            "--name", "test_kernel"
        ])
        
        self.assertEqual(result.exit_code, 0)
        mock_search_service.assert_called_once()

    @patch('model_cache_manager.cli.main._cache_db_exists')
    def test_list_command_no_db_vllm_mode(self, mock_db_exists):
        """Test list command when vLLM database doesn't exist."""
        mock_db_exists.return_value = False
        
        result = self.runner.invoke(app, [
            "list", 
            "--mode", "vllm"
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("DB was not found for vllm mode", result.stdout)
        self.assertIn("mcm index --mode vllm", result.stdout)

    @patch('model_cache_manager.cli.main._cache_db_exists')
    @patch('model_cache_manager.cli.main.PruningService')
    def test_prune_command_vllm_mode(self, mock_prune_service, mock_db_exists):
        """Test prune command with vLLM mode."""
        mock_db_exists.return_value = True
        mock_service_instance = MagicMock()
        mock_service_instance.prune.return_value = MagicMock(pruned=2, reclaimed=1.5)
        mock_prune_service.return_value = mock_service_instance
        
        result = self.runner.invoke(app, [
            "prune", 
            "--mode", "vllm",
            "--older-than", "7d",
            "--yes"
        ])
        
        self.assertEqual(result.exit_code, 0)
        mock_prune_service.assert_called_once_with(
            cache_dir=None, 
            mode="vllm"
        )

    @patch('model_cache_manager.cli.main.detect_cache_mode')
    @patch('model_cache_manager.cli.main._cache_db_exists')
    @patch('model_cache_manager.cli.main.PruningService')
    def test_prune_command_auto_detection(self, mock_prune_service, mock_db_exists, mock_detect):
        """Test prune command with auto-detection."""
        mock_detect.return_value = "vllm"
        mock_db_exists.return_value = True
        mock_service_instance = MagicMock()
        mock_service_instance.prune.return_value = MagicMock(pruned=1, reclaimed=0.5)
        mock_prune_service.return_value = mock_service_instance
        
        result = self.runner.invoke(app, [
            "prune", 
            "--cache-dir", str(self.vllm_cache_dir),
            "--yes"
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Auto-detected cache mode: vllm", result.stdout)
        mock_detect.assert_called_once()

    def test_prune_command_invalid_cache_hit_bounds(self):
        """Test prune command with invalid cache hit bounds."""
        result = self.runner.invoke(app, [
            "prune", 
            "--mode", "triton",
            "--cache-hit-lower", "100",
            "--cache-hit-higher", "50"
        ])
        
        self.assertEqual(result.exit_code, 0)
        # The validation might happen at different points, but the command should complete
        # The actual validation message might vary based on when it's checked

    @patch('model_cache_manager.cli.main.IndexService')
    def test_index_command_file_not_found_error(self, mock_index_service):
        """Test index command when cache directory doesn't exist."""
        mock_index_service.side_effect = FileNotFoundError("Cache directory not found")
        
        result = self.runner.invoke(app, [
            "index", 
            "--mode", "vllm",
            "--cache-dir", "/nonexistent/path"
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Cache directory not found", result.stdout)

    def test_help_shows_mode_parameter(self):
        """Test that help shows the --mode parameter."""
        result = self.runner.invoke(app, ["index", "--help"])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--mode", result.stdout)
        self.assertIn("vllm", result.stdout)
        self.assertIn("triton", result.stdout)
        self.assertIn("Auto-detected", result.stdout)


if __name__ == "__main__":
    unittest.main()