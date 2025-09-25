"""
Unit tests for legacy vLLM mode in IndexService, SearchService, and PruningService.
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
from model_cache_manager.data.cache_repo import VllmLegacyCacheRepository
from model_cache_manager.data.vllm_legacy_database import VllmLegacyDatabase
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


class TestIndexServiceVllmLegacyMode(unittest.TestCase):
    """Test suite for IndexService in legacy vLLM mode."""

    def setUp(self):
        """Set up test fixtures."""
        setup_test_class_fixtures(self)

    def tearDown(self):
        """Clean up after each test method."""
        teardown_test_fixtures(self)

    @patch("model_cache_manager.strategies.vllm_legacy_strategy.VllmLegacyCacheRepository")
    @patch("model_cache_manager.strategies.vllm_legacy_strategy.VllmLegacyDatabase")
    def test_init_vllm_legacy_mode(self, mock_vllm_db, mock_vllm_repo):
        """Test IndexService initialization in vLLM mode."""
        setup_init_test_mocks(mock_vllm_repo, mock_vllm_db)
        service = IndexService(cache_dir=self.cache_dir, mode="vllm-legacy")
        assert_vllm_service_init(self, service, self.cache_dir, mock_vllm_repo, mock_vllm_db)

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

    @patch("model_cache_manager.strategies.vllm_legacy_strategy.VllmLegacyCacheRepository")
    @patch("model_cache_manager.strategies.vllm_legacy_strategy.VllmLegacyDatabase")
    def test_reindex_vllm_mode(self, mock_vllm_db, mock_vllm_repo):
        """Test reindex method in vLLM mode."""
        mock_repo_instance, mock_db_instance, mock_kernels = setup_reindex_test(
            "vllm_legacy", mock_vllm_repo, mock_vllm_db
        )

        service = IndexService(cache_dir=self.cache_dir, mode="vllm-legacy")
        assert_reindex_results(self, service, mock_db_instance, "vllm_legacy")

    @patch("model_cache_manager.strategies.triton_strategy.CacheRepository")
    @patch("model_cache_manager.strategies.triton_strategy.Database")
    def test_reindex_triton_mode(self, mock_db, mock_repo):
        """Test reindex method in triton mode (backward compatibility)."""
        mock_repo_instance, mock_db_instance, mock_kernels = setup_reindex_test(
            "triton", mock_repo, mock_db, self.cache_dir
        )

        service = IndexService(cache_dir=self.cache_dir, mode="triton")
        assert_reindex_results(self, service, mock_db_instance, "triton", cache_dir=self.cache_dir)


class TestSearchServiceVllmLegacyMode(unittest.TestCase):
    """Test suite for SearchService in legacy vLLM mode."""

    @patch("model_cache_manager.strategies.vllm_legacy_strategy.VllmLegacyDatabase")
    def test_init_vllm_legacy_mode(self, mock_vllm_db):
        """Test SearchService initialization in vLLM mode."""
        mock_db_instance = MagicMock()
        mock_vllm_db.return_value = mock_db_instance

        criteria = SearchCriteria(name="test_kernel")
        service = SearchService(criteria=criteria, mode="vllm-legacy")

        self.assertEqual(service.mode, "vllm-legacy")
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

    @patch("model_cache_manager.strategies.vllm_legacy_strategy.VllmLegacyDatabase")
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
        service = SearchService(criteria=criteria, mode="vllm-legacy")
        results = service.search()

        self.assertEqual(results, mock_results)
        mock_db_instance.search.assert_called_once_with(criteria)


class TestPruningServiceVllmLegacyMode(unittest.TestCase):
    """Test suite for PruningService in legacy vLLM mode."""

    def setUp(self):
        """Set up test fixtures."""
        setup_test_class_fixtures(self)

    def tearDown(self):
        """Clean up after each test method."""
        teardown_test_fixtures(self)

    @patch("model_cache_manager.strategies.vllm_legacy_strategy.VllmLegacyCacheRepository")
    @patch("model_cache_manager.strategies.vllm_legacy_strategy.VllmLegacyDatabase")
    def test_init_vllm_legacy_mode(self, mock_vllm_db, mock_vllm_repo):
        """Test PruningService initialization in vLLM mode."""
        setup_init_test_mocks(mock_vllm_repo, mock_vllm_db)
        service = PruningService(cache_dir=self.cache_dir, mode="vllm-legacy")
        assert_vllm_service_init(self, service, self.cache_dir, mock_vllm_repo, mock_vllm_db)

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

    @patch("model_cache_manager.strategies.vllm_legacy_strategy.VllmLegacyCacheRepository")
    @patch("model_cache_manager.strategies.vllm_legacy_strategy.VllmLegacyDatabase")
    def test_prune_basic_functionality_vllm(self, mock_vllm_db, mock_vllm_repo):
        """Test basic prune functionality in vLLM mode."""
        mock_repo_instance, mock_db_instance = setup_prune_test(
            mock_vllm_repo, mock_vllm_db
        )

        service = PruningService(cache_dir=self.cache_dir, mode="vllm-legacy")
        criteria = SearchCriteria(older_than_timestamp=1000000.0)
        assert_prune_results(self, service, criteria, mock_db_instance)


if __name__ == "__main__":
    unittest.main()
