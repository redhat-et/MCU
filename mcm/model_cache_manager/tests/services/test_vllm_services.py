"""
Unit tests for vLLM mode in IndexService, SearchService, and PruningService.
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


class TestIndexServiceVllmMode(unittest.TestCase):
    """Test suite for IndexService in vLLM mode."""

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
    def test_init_vllm_mode(self, mock_vllm_db, mock_vllm_repo):
        """Test IndexService initialization in vLLM mode."""
        mock_repo_instance = MagicMock()
        mock_db_instance = MagicMock()
        mock_vllm_repo.return_value = mock_repo_instance
        mock_vllm_db.return_value = mock_db_instance

        service = IndexService(cache_dir=self.cache_dir, mode="vllm")

        self.assertEqual(service.mode, "vllm")
        self.assertEqual(service.cache_dir, self.cache_dir)
        mock_vllm_repo.assert_called_once_with(self.cache_dir)
        mock_vllm_db.assert_called_once()

    @patch("model_cache_manager.strategies.triton_strategy.CacheRepository")
    @patch("model_cache_manager.strategies.triton_strategy.Database")
    def test_init_triton_mode(self, mock_db, mock_repo):
        """Test IndexService initialization in triton mode (backward compatibility)."""
        mock_repo_instance = MagicMock()
        mock_db_instance = MagicMock()
        mock_repo.return_value = mock_repo_instance
        mock_db.return_value = mock_db_instance

        service = IndexService(cache_dir=self.cache_dir, mode="triton")

        self.assertEqual(service.mode, "triton")
        mock_repo.assert_called_once_with(self.cache_dir)
        mock_db.assert_called_once()

    @patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository")
    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_reindex_vllm_mode(self, mock_vllm_db, mock_vllm_repo):
        """Test reindex method in vLLM mode."""
        # Mock kernels from vLLM repo
        mock_kernel1 = create_mock_kernel("hash1", "kernel1")
        mock_kernel2 = create_mock_kernel("hash2", "kernel2")

        mock_repo_instance = MagicMock(spec=VllmCacheRepository)
        mock_repo_instance.kernels.return_value = [
            ("vllm_hash1", "/cache/root", mock_kernel1),
            ("vllm_hash2", "/cache/root", mock_kernel2),
        ]
        mock_vllm_repo.return_value = mock_repo_instance

        mock_db_instance = MagicMock(spec=VllmDatabase)
        mock_db_instance.search.return_value = []  # No existing kernels
        mock_vllm_db.return_value = mock_db_instance

        service = IndexService(cache_dir=self.cache_dir, mode="vllm")
        updated, current = service.reindex()

        self.assertEqual(updated, 2)
        self.assertEqual(current, 0)

        # Verify kernels were inserted with vLLM parameters
        self.assertEqual(mock_db_instance.insert_kernel.call_count, 2)
        mock_db_instance.insert_kernel.assert_any_call(
            mock_kernel1, "/cache/root", "vllm_hash1"
        )
        mock_db_instance.insert_kernel.assert_any_call(
            mock_kernel2, "/cache/root", "vllm_hash2"
        )

    @patch("model_cache_manager.strategies.triton_strategy.CacheRepository")
    @patch("model_cache_manager.strategies.triton_strategy.Database")
    def test_reindex_triton_mode(self, mock_db, mock_repo):
        """Test reindex method in triton mode (backward compatibility)."""
        mock_kernel1 = create_mock_kernel("hash1", "kernel1")
        mock_kernel2 = create_mock_kernel("hash2", "kernel2")

        mock_repo_instance = MagicMock()
        mock_repo_instance.kernels.return_value = [mock_kernel1, mock_kernel2]
        mock_repo_instance.cache_dir = self.cache_dir
        mock_repo_instance.root = self.cache_dir
        mock_repo.return_value = mock_repo_instance

        mock_db_instance = MagicMock()
        mock_db_instance.search.return_value = []
        mock_db.return_value = mock_db_instance

        service = IndexService(cache_dir=self.cache_dir, mode="triton")
        updated, current = service.reindex()

        self.assertEqual(updated, 2)
        self.assertEqual(current, 0)

        # Verify kernels were inserted with triton parameters
        self.assertEqual(mock_db_instance.insert_kernel.call_count, 2)
        mock_db_instance.insert_kernel.assert_any_call(
            mock_kernel1, str(self.cache_dir)
        )
        mock_db_instance.insert_kernel.assert_any_call(
            mock_kernel2, str(self.cache_dir)
        )


class TestSearchServiceVllmMode(unittest.TestCase):
    """Test suite for SearchService in vLLM mode."""

    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_init_vllm_mode(self, mock_vllm_db):
        """Test SearchService initialization in vLLM mode."""
        mock_db_instance = MagicMock()
        mock_vllm_db.return_value = mock_db_instance

        criteria = SearchCriteria(name="test_kernel")
        service = SearchService(criteria=criteria, mode="vllm")

        self.assertEqual(service.mode, "vllm")
        mock_vllm_db.assert_called_once()

    @patch("model_cache_manager.strategies.triton_strategy.Database")
    def test_init_triton_mode(self, mock_db):
        """Test SearchService initialization in triton mode."""
        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance

        criteria = SearchCriteria(name="test_kernel")
        service = SearchService(criteria=criteria, mode="triton")

        self.assertEqual(service.mode, "triton")
        mock_db.assert_called_once()

    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_search_vllm_mode(self, mock_vllm_db):
        """Test search method in vLLM mode."""
        mock_results = [
            {"hash": "hash1", "name": "kernel1"},
            {"hash": "hash2", "name": "kernel2"},
        ]

        mock_db_instance = MagicMock()
        mock_db_instance.search.return_value = mock_results
        mock_vllm_db.return_value = mock_db_instance

        criteria = SearchCriteria(name="test_kernel")
        service = SearchService(criteria=criteria, mode="vllm")
        results = service.search()

        self.assertEqual(results, mock_results)
        mock_db_instance.search.assert_called_once_with(criteria)


class TestPruningServiceVllmMode(unittest.TestCase):
    """Test suite for PruningService in vLLM mode."""

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
    def test_init_vllm_mode(self, mock_vllm_db, mock_vllm_repo):
        """Test PruningService initialization in vLLM mode."""
        mock_repo_instance = MagicMock()
        mock_db_instance = MagicMock()
        mock_vllm_repo.return_value = mock_repo_instance
        mock_vllm_db.return_value = mock_db_instance

        service = PruningService(cache_dir=self.cache_dir, mode="vllm")

        self.assertEqual(service.mode, "vllm")
        self.assertEqual(service.cache_dir, self.cache_dir)
        mock_vllm_repo.assert_called_once_with(self.cache_dir)
        mock_vllm_db.assert_called_once()

    @patch("model_cache_manager.strategies.triton_strategy.CacheRepository")
    @patch("model_cache_manager.strategies.triton_strategy.Database")
    def test_init_triton_mode(self, mock_db, mock_repo):
        """Test PruningService initialization in triton mode."""
        mock_repo_instance = MagicMock()
        mock_db_instance = MagicMock()
        mock_repo.return_value = mock_repo_instance
        mock_db.return_value = mock_db_instance

        service = PruningService(cache_dir=self.cache_dir, mode="triton")

        self.assertEqual(service.mode, "triton")
        mock_repo.assert_called_once_with(self.cache_dir)
        mock_db.assert_called_once()

    @patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository")
    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_prune_basic_functionality_vllm(self, mock_vllm_db, mock_vllm_repo):
        """Test basic prune functionality in vLLM mode."""
        # Mock search results - include both vllm_hash and triton_cache_key for vLLM mode
        mock_search_results = [
            {"vllm_hash": "vllm_hash1", "triton_cache_key": "hash1", "name": "kernel1"},
            {"vllm_hash": "vllm_hash2", "triton_cache_key": "hash2", "name": "kernel2"},
        ]

        mock_repo_instance = MagicMock()
        mock_vllm_repo.return_value = mock_repo_instance

        mock_db_instance = MagicMock()
        mock_db_instance.search.return_value = mock_search_results
        mock_db_instance.estimate_space.return_value = 2048
        mock_db_instance.get_session.return_value.__enter__.return_value = MagicMock()
        mock_vllm_db.return_value = mock_db_instance

        service = PruningService(cache_dir=self.cache_dir, mode="vllm")

        with patch.object(service, "_delete_kernel_unified", return_value=1024), patch.object(
            service, "_confirm", return_value=True
        ):

            criteria = SearchCriteria(older_than_timestamp=1000000.0)
            result = service.prune(criteria, auto_confirm=True)

            self.assertIsNotNone(result)
            self.assertEqual(result.pruned, 2)
            mock_db_instance.search.assert_called_once()


if __name__ == "__main__":
    unittest.main()
