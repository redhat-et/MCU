"""
Unit tests for new vLLM mode in IndexService, SearchService, and PruningService.
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


class TestIndexServiceVllmNewMode(unittest.TestCase):
    """Test suite for IndexService in new vLLM mode."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "vllm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
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

        self.assertEqual(service.mode, "vllm")
        self.assertEqual(service.cache_dir, self.cache_dir)
        mock_vllm_repo.assert_called_once_with(self.cache_dir)
        mock_vllm_db.assert_called_once()

    @patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository")
    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_index_new_vllm_structure(self, mock_vllm_db, mock_vllm_repo):
        """Test indexing with new vLLM structure including artifact_compile_range."""
        mock_kernel = MagicMock()
        mock_kernel.hash = "test_hash"
        mock_kernel.name = "test_kernel"

        mock_repo_instance = MagicMock()
        mock_repo_instance.kernels.return_value = [
            (
                "vllm_hash_1",
                str(self.cache_dir),
                "rank_0_0",
                "artifact_compile_range_0",
                '{"config": "test"}',
                mock_kernel,
            ),
            (
                "vllm_hash_2",
                str(self.cache_dir),
                "rank_0_1",
                "artifact_compile_range_1",
                None,
                mock_kernel,
            ),
        ]
        mock_vllm_repo.return_value = mock_repo_instance

        mock_db_instance = MagicMock()
        mock_db_instance.bulk_insert_kernels.return_value = 2
        mock_vllm_db.return_value = mock_db_instance

        service = IndexService(cache_dir=self.cache_dir, mode="vllm")
        service.reindex()

        mock_db_instance.bulk_insert_kernels.assert_called_once()

        call_args = mock_db_instance.bulk_insert_kernels.call_args[0][0]
        self.assertEqual(len(call_args), 2)

        first_kernel_data = call_args[0]
        self.assertEqual(len(first_kernel_data), 6)
        self.assertEqual(first_kernel_data[0], mock_kernel)
        self.assertEqual(first_kernel_data[1], str(self.cache_dir))
        self.assertEqual(first_kernel_data[2], "vllm_hash_1")
        self.assertEqual(first_kernel_data[3], "rank_0_0")
        self.assertEqual(first_kernel_data[4], "artifact_compile_range_0")
        self.assertEqual(first_kernel_data[5], '{"config": "test"}')


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
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "vllm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
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
