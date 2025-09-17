"""
Unit tests for new vLLM mode in IndexService, SearchService, and PruningService.
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
import tempfile
import shutil

from model_cache_manager.services.index import IndexService
from model_cache_manager.services.search import SearchService
from model_cache_manager.services.prune import PruningService
from model_cache_manager.models.criteria import SearchCriteria
from model_cache_manager.models.kernel import Kernel
from model_cache_manager.data.cache_repo import VllmCacheRepository
from model_cache_manager.data.database import VllmDatabase


def create_mock_kernel(
    hash_val: str = "test_hash", name: str = "test_kernel"
) -> Kernel:
    """Helper to create a mock Kernel object."""
    kernel = Mock(spec=Kernel)
    kernel.hash = hash_val
    kernel.name = name
    kernel.backend = "cuda"
    kernel.arch = "80"
    kernel.files = []
    return kernel


class TestIndexServiceVllmNewMode(unittest.TestCase):
    """Test suite for IndexService in new vLLM mode."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "vllm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository")
    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_init_vllm_new_mode(self, mock_vllm_db, mock_vllm_repo):
        """Test IndexService initialization in new vLLM mode."""
        mock_repo_instance = MagicMock()
        mock_db_instance = MagicMock()
        mock_vllm_repo.return_value = mock_repo_instance
        mock_vllm_db.return_value = mock_db_instance

        service = IndexService(cache_dir=self.cache_dir, mode="vllm")

        # Check that service was initialized correctly
        self.assertEqual(service.mode, "vllm")
        self.assertEqual(service.cache_dir, self.cache_dir)
        mock_vllm_repo.assert_called_once_with(self.cache_dir)
        mock_vllm_db.assert_called_once()

    @patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository")
    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_index_new_vllm_structure(self, mock_vllm_db, mock_vllm_repo):
        """Test indexing with new vLLM cache structure including artifact_shape."""
        # Mock repository with new structure
        mock_repo_instance = MagicMock()
        kernel = create_mock_kernel("test_hash")
        
        # New vLLM structure yields: vllm_hash, cache_root, rank_x_y, artifact_shape, best_config, kernel
        mock_repo_instance.kernels.return_value = [
            ("vllm_hash_1", str(self.cache_dir), "rank_0_0", "artifact_shape_0", '{"config": "test"}', kernel),
            ("vllm_hash_2", str(self.cache_dir), "rank_0_1", "artifact_shape_1", None, kernel)
        ]
        
        mock_vllm_repo.return_value = mock_repo_instance
        
        # Mock database
        mock_db_instance = MagicMock()
        mock_vllm_db.return_value = mock_db_instance

        service = IndexService(cache_dir=self.cache_dir, mode="vllm")
        service.index()

        # Verify kernels were processed with artifact_shape and best_config
        self.assertEqual(mock_db_instance.insert_kernel.call_count, 2)
        
        # Check first call includes artifact_shape and best_config
        first_call = mock_db_instance.insert_kernel.call_args_list[0]
        self.assertEqual(first_call[0][0], kernel)  # kernel data
        self.assertEqual(first_call[0][1], str(self.cache_dir))  # cache_dir
        self.assertEqual(first_call[0][2], "vllm_hash_1")  # vllm_hash
        self.assertEqual(first_call[0][3], "rank_0_0")  # rank_x_y
        self.assertEqual(first_call[0][4], "artifact_shape_0")  # artifact_shape
        self.assertEqual(first_call[0][5], '{"config": "test"}')  # best_config


class TestSearchServiceVllmNewMode(unittest.TestCase):
    """Test suite for SearchService in new vLLM mode."""

    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_search_vllm_new_mode(self, mock_vllm_db):
        """Test SearchService initialization in new vLLM mode."""
        mock_db_instance = MagicMock()
        mock_vllm_db.return_value = mock_db_instance

        criteria = SearchCriteria(name="test_kernel")
        service = SearchService(criteria=criteria, mode="vllm")

        self.assertEqual(service.mode, "vllm")
        mock_vllm_db.assert_called_once()


class TestPruningServiceVllmNewMode(unittest.TestCase):
    """Test suite for PruningService in new vLLM mode."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "vllm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository")
    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_prune_vllm_new_mode(self, mock_vllm_db, mock_vllm_repo):
        """Test PruningService initialization in new vLLM mode."""
        mock_repo_instance = MagicMock()
        mock_db_instance = MagicMock()
        mock_vllm_repo.return_value = mock_repo_instance
        mock_vllm_db.return_value = mock_db_instance

        service = PruningService(cache_dir=self.cache_dir, mode="vllm")

        self.assertEqual(service.mode, "vllm")
        self.assertEqual(service.cache_dir, self.cache_dir)
        mock_vllm_repo.assert_called_once_with(self.cache_dir)
        mock_vllm_db.assert_called_once()


if __name__ == "__main__":
    unittest.main()