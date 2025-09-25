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
)


class TestIndexServiceVllmMode(unittest.TestCase):
    """Test suite for IndexService in vLLM mode."""

    def setUp(self):
        """Set up test fixtures."""
        setup_test_class_fixtures(self)

    def tearDown(self):
        """Clean up after each test method."""
        teardown_test_fixtures(self)

    @patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository")
    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_init_vllm_mode(self, mock_vllm_db, mock_vllm_repo):
        """Test IndexService initialization in vLLM mode."""
        _, _, mock_repo_instance, mock_db_instance = setup_service_mocks("index", "vllm")
        mock_vllm_repo.return_value = mock_repo_instance
        mock_vllm_db.return_value = mock_db_instance

        service = IndexService(cache_dir=self.cache_dir, mode="vllm")

        assert_service_init(self, service, "vllm", self.cache_dir, mock_vllm_repo, mock_vllm_db)

    @patch("model_cache_manager.strategies.triton_strategy.CacheRepository")
    @patch("model_cache_manager.strategies.triton_strategy.Database")
    def test_init_triton_mode(self, mock_db, mock_repo):
        """Test IndexService initialization in triton mode (backward compatibility)."""
        _, _, mock_repo_instance, mock_db_instance = setup_service_mocks("index", "triton")
        mock_repo.return_value = mock_repo_instance
        mock_db.return_value = mock_db_instance

        service = IndexService(cache_dir=self.cache_dir, mode="triton")

        assert_service_init(self, service, "triton", self.cache_dir, mock_repo, mock_db)

    @patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository")
    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_reindex_vllm_mode(self, mock_vllm_db, mock_vllm_repo):
        """Test reindex method in vLLM mode."""
        mock_repo_instance, mock_db_instance, mock_kernels = setup_reindex_test(
            "vllm", mock_vllm_repo, mock_vllm_db
        )

        service = IndexService(cache_dir=self.cache_dir, mode="vllm")
        assert_reindex_results(self, service, mock_db_instance, "vllm")

    @patch("model_cache_manager.strategies.triton_strategy.CacheRepository")
    @patch("model_cache_manager.strategies.triton_strategy.Database")
    def test_reindex_triton_mode(self, mock_db, mock_repo):
        """Test reindex method in triton mode (backward compatibility)."""
        mock_repo_instance, mock_db_instance, mock_kernels = setup_reindex_test(
            "triton", mock_repo, mock_db, self.cache_dir
        )

        service = IndexService(cache_dir=self.cache_dir, mode="triton")
        assert_reindex_results(self, service, mock_db_instance, "triton", cache_dir=self.cache_dir)


class TestSearchServiceVllmMode(unittest.TestCase):
    """Test suite for SearchService in vLLM mode."""

    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_init_vllm_mode(self, mock_vllm_db):
        """Test SearchService initialization in vLLM mode."""
        _, _, _, mock_db_instance = setup_service_mocks("search", "vllm")
        mock_vllm_db.return_value = mock_db_instance

        criteria = SearchCriteria(name="test_kernel")
        service = SearchService(criteria=criteria, mode="vllm")

        assert_service_init(self, service, "vllm", None, None, mock_vllm_db)

    @patch("model_cache_manager.strategies.triton_strategy.Database")
    def test_init_triton_mode(self, mock_db):
        """Test SearchService initialization in triton mode."""
        _, _, _, mock_db_instance = setup_service_mocks("search", "triton")
        mock_db.return_value = mock_db_instance

        criteria = SearchCriteria(name="test_kernel")
        service = SearchService(criteria=criteria, mode="triton")

        assert_service_init(self, service, "triton", None, None, mock_db)

    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_search_vllm_mode(self, mock_vllm_db):
        """Test search method in vLLM mode."""
        mock_db_instance, mock_results = setup_search_test(mock_vllm_db)

        criteria = SearchCriteria(name="test_kernel")
        service = SearchService(criteria=criteria, mode="vllm")
        results = service.search()

        self.assertEqual(results, mock_results)
        mock_db_instance.search.assert_called_once_with(criteria)


class TestPruningServiceVllmMode(unittest.TestCase):
    """Test suite for PruningService in vLLM mode."""

    def setUp(self):
        """Set up test fixtures."""
        setup_test_class_fixtures(self)

    def tearDown(self):
        """Clean up after each test method."""
        teardown_test_fixtures(self)

    @patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository")
    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_init_vllm_mode(self, mock_vllm_db, mock_vllm_repo):
        """Test PruningService initialization in vLLM mode."""
        _, _, mock_repo_instance, mock_db_instance = setup_service_mocks("prune", "vllm")
        mock_vllm_repo.return_value = mock_repo_instance
        mock_vllm_db.return_value = mock_db_instance

        service = PruningService(cache_dir=self.cache_dir, mode="vllm")

        assert_service_init(self, service, "vllm", self.cache_dir, mock_vllm_repo, mock_vllm_db)

    @patch("model_cache_manager.strategies.triton_strategy.CacheRepository")
    @patch("model_cache_manager.strategies.triton_strategy.Database")
    def test_init_triton_mode(self, mock_db, mock_repo):
        """Test PruningService initialization in triton mode."""
        _, _, mock_repo_instance, mock_db_instance = setup_service_mocks("prune", "triton")
        mock_repo.return_value = mock_repo_instance
        mock_db.return_value = mock_db_instance

        service = PruningService(cache_dir=self.cache_dir, mode="triton")

        assert_service_init(self, service, "triton", self.cache_dir, mock_repo, mock_db)

    @patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository")
    @patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase")
    def test_prune_basic_functionality_vllm(self, mock_vllm_db, mock_vllm_repo):
        """Test basic prune functionality in vLLM mode."""
        mock_repo_instance, mock_db_instance = setup_prune_test(
            mock_vllm_repo, mock_vllm_db
        )

        service = PruningService(cache_dir=self.cache_dir, mode="vllm")
        criteria = SearchCriteria(older_than_timestamp=1000000.0)
        assert_prune_results(self, service, criteria, mock_db_instance)


if __name__ == "__main__":
    unittest.main()
