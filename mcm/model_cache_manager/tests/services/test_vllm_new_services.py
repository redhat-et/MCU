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
from model_cache_manager.tests.test_utils import (
    create_mock_kernel,
    setup_service_mocks,
    assert_service_init,
    create_test_search_results,
    setup_test_class_fixtures,
    teardown_test_fixtures,
    setup_reindex_test,
    assert_reindex_results,
    setup_search_test,
    setup_prune_test,
    assert_prune_results,
    setup_init_test_mocks,
    assert_vllm_service_init,
)


class TestIndexServiceVllmNewMode(unittest.TestCase):
    """Test suite for IndexService in new vLLM mode."""

    def setUp(self):
        """Set up test fixtures."""
        setup_test_class_fixtures(self)

    def tearDown(self):
        """Clean up after each test method."""
        teardown_test_fixtures(self)

    @patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository")
    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_init_vllm_new_mode(self, mock_vllm_db, mock_vllm_repo):
        """Test IndexService initialization in new vLLM mode."""
        setup_init_test_mocks(mock_vllm_repo, mock_vllm_db)
        service = IndexService(cache_dir=self.cache_dir, mode="vllm")
        assert_vllm_service_init(self, service, self.cache_dir, mock_vllm_repo, mock_vllm_db)

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
        mock_db_instance.bulk_insert_kernels.return_value = 2
        mock_vllm_db.return_value = mock_db_instance

        service = IndexService(cache_dir=self.cache_dir, mode="vllm")
        service.reindex()

        # Verify kernels were processed with bulk_insert_kernels
        mock_db_instance.bulk_insert_kernels.assert_called_once()

        # Check the batch contains the expected kernel data
        call_args = mock_db_instance.bulk_insert_kernels.call_args[0][0]
        self.assertEqual(len(call_args), 2)  # Two kernels in the batch

        # Check first kernel includes artifact_shape and best_config
        first_kernel_data = call_args[0]
        self.assertEqual(len(first_kernel_data), 6)  # kernel, cache_dir, vllm_hash, rank_x_y, artifact_shape, best_config
        self.assertEqual(first_kernel_data[0], kernel)  # kernel data
        self.assertEqual(first_kernel_data[1], str(self.cache_dir))  # cache_dir
        self.assertEqual(first_kernel_data[2], "vllm_hash_1")  # vllm_hash
        self.assertEqual(first_kernel_data[3], "rank_0_0")  # rank_x_y
        self.assertEqual(first_kernel_data[4], "artifact_shape_0")  # artifact_shape
        self.assertEqual(first_kernel_data[5], '{"config": "test"}')  # best_config


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
        setup_test_class_fixtures(self)

    def tearDown(self):
        """Clean up after each test method."""
        teardown_test_fixtures(self)

    @patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository")
    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_prune_vllm_new_mode(self, mock_vllm_db, mock_vllm_repo):
        """Test PruningService initialization in new vLLM mode."""
        setup_init_test_mocks(mock_vllm_repo, mock_vllm_db)

        service = PruningService(cache_dir=self.cache_dir, mode="vllm")
        assert_vllm_service_init(self, service, self.cache_dir, mock_vllm_repo, mock_vllm_db)


if __name__ == "__main__":
    unittest.main()
