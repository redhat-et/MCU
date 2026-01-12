"""
Unit tests for vLLM mode in IndexService, SearchService, and PruningService.
"""
# pylint: disable=duplicate-code

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil

from model_cache_manager.services.index import IndexService
from model_cache_manager.services.search import SearchService
from model_cache_manager.services.prune import PruningService
from model_cache_manager.models.criteria import SearchCriteria


class TestIndexServiceVllmMode(unittest.TestCase):
    """Test suite for IndexService in vLLM mode."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "vllm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
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
        """Test IndexService initialization in triton mode."""
        mock_repo_instance = MagicMock()
        mock_db_instance = MagicMock()
        mock_repo.return_value = mock_repo_instance
        mock_db.return_value = mock_db_instance

        service = IndexService(cache_dir=self.cache_dir, mode="triton")

        self.assertEqual(service.mode, "triton")
        self.assertEqual(service.cache_dir, self.cache_dir)
        mock_repo.assert_called_once_with(self.cache_dir)
        mock_db.assert_called_once()

    @patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository")
    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_reindex_vllm_mode(self, mock_vllm_db, mock_vllm_repo):
        """Test reindex method in vLLM mode."""
        mock_kernel1 = MagicMock()
        mock_kernel1.hash = "hash1"
        mock_kernel1.name = "kernel1"

        mock_kernel2 = MagicMock()
        mock_kernel2.hash = "hash2"
        mock_kernel2.name = "kernel2"

        mock_repo_instance = MagicMock()
        mock_repo_instance.kernels.return_value = [
            (
                "vllm_hash1",
                "/cache/root",
                "rank_0_0",
                "artifact_compile_range_0",
                '{"config": "test"}',
                mock_kernel1,
            ),
            (
                "vllm_hash2",
                "/cache/root",
                "rank_1_0",
                "artifact_compile_range_1",
                None,
                mock_kernel2,
            ),
        ]
        mock_vllm_repo.return_value = mock_repo_instance

        mock_db_instance = MagicMock()
        mock_db_instance.search.return_value = []
        mock_db_instance.bulk_insert_kernels.return_value = 2
        mock_vllm_db.return_value = mock_db_instance

        service = IndexService(cache_dir=self.cache_dir, mode="vllm")
        updated, current = service.reindex()

        self.assertEqual(updated, 2)
        self.assertEqual(current, 0)
        mock_db_instance.bulk_insert_kernels.assert_called_once()

        call_args = mock_db_instance.bulk_insert_kernels.call_args[0][0]
        self.assertEqual(len(call_args), 2)
        self.assertEqual(len(call_args[0]), 6)
        self.assertEqual(len(call_args[1]), 6)

    @patch("model_cache_manager.strategies.triton_strategy.CacheRepository")
    @patch("model_cache_manager.strategies.triton_strategy.Database")
    def test_reindex_triton_mode(self, mock_db, mock_repo):
        """Test reindex method in triton mode."""
        mock_kernel1 = MagicMock()
        mock_kernel1.hash = "hash1"
        mock_kernel1.name = "kernel1"

        mock_kernel2 = MagicMock()
        mock_kernel2.hash = "hash2"
        mock_kernel2.name = "kernel2"

        mock_repo_instance = MagicMock()
        mock_repo_instance.kernels.return_value = [mock_kernel1, mock_kernel2]
        mock_repo_instance.cache_dir = self.cache_dir
        mock_repo_instance.root = self.cache_dir
        mock_repo.return_value = mock_repo_instance

        mock_db_instance = MagicMock()
        mock_db_instance.search.return_value = []
        mock_db_instance.bulk_insert_kernels.return_value = 2
        mock_db.return_value = mock_db_instance

        service = IndexService(cache_dir=self.cache_dir, mode="triton")
        updated, current = service.reindex()

        self.assertEqual(updated, 2)
        self.assertEqual(current, 0)
        mock_db_instance.bulk_insert_kernels.assert_called_once()


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
        expected_results = [
            {"hash": "hash1", "name": "kernel1"},
            {"hash": "hash2", "name": "kernel2"},
        ]

        mock_db_instance = MagicMock()
        mock_db_instance.search.return_value = expected_results
        mock_vllm_db.return_value = mock_db_instance

        criteria = SearchCriteria(name="test_kernel")
        service = SearchService(criteria=criteria, mode="vllm")
        results = service.search()

        self.assertEqual(results, expected_results)
        mock_db_instance.search.assert_called_once_with(criteria)


class TestPruningServiceVllmMode(unittest.TestCase):
    """Test suite for PruningService in vLLM mode."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "vllm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
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
        self.assertEqual(service.cache_dir, self.cache_dir)
        mock_repo.assert_called_once_with(self.cache_dir)
        mock_db.assert_called_once()

    @patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository")
    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_prune_basic_functionality_vllm(self, mock_vllm_db, mock_vllm_repo):
        """Test basic prune functionality in vLLM mode."""
        search_results = [
            {
                "vllm_hash": "vllm_hash1",
                "triton_cache_key": "hash1",
                "rank_x_y": "rank_0_0",
                "name": "kernel1",
            },
            {
                "vllm_hash": "vllm_hash2",
                "triton_cache_key": "hash2",
                "rank_x_y": "rank_1_0",
                "name": "kernel2",
            },
        ]

        mock_repo_instance = MagicMock()
        mock_vllm_repo.return_value = mock_repo_instance

        mock_db_instance = MagicMock()
        mock_db_instance.search.return_value = search_results
        mock_db_instance.estimate_space.return_value = 2048
        mock_db_instance.get_session.return_value.__enter__.return_value = MagicMock()
        mock_vllm_db.return_value = mock_db_instance

        service = PruningService(cache_dir=self.cache_dir, mode="vllm")
        criteria = SearchCriteria(older_than_timestamp=1000000.0)

        with patch.object(service, "_delete_kernel_unified", return_value=1024), \
             patch.object(service, "_confirm", return_value=True):
            result = service.prune(criteria, auto_confirm=True)

            self.assertIsNotNone(result)
            self.assertEqual(result.pruned, 2)
            mock_db_instance.search.assert_called_once()


if __name__ == "__main__":
    unittest.main()
