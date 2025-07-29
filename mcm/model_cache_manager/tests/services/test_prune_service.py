"""
Unit tests for the PruningService.
"""

import unittest
from unittest.mock import (
    patch,
    MagicMock,
)
from pathlib import Path
import logging

from model_cache_manager.services.prune import PruningService, PruneStats
from model_cache_manager.models.criteria import SearchCriteria
from model_cache_manager.utils.mcm_constants import IR_EXTS

from model_cache_manager.data.cache_repo import CacheRepository
from model_cache_manager.data.database import Database

logging.disable(logging.CRITICAL)


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def create_mock_kernel_data(
    hash_val: str,
    name: str,
    backend: str = "cuda",
    arch: str = "80",
    mod_time_offset_secs: int = 0,
    total_size_bytes: int = 1024,
    runtime_hits: int = 0,
) -> dict:
    """Helper to create consistent mock kernel data dictionary."""
    base_timestamp = 1747681046.0
    return {
        "hash": hash_val,
        "name": name,
        "backend": backend,
        "arch": arch,
        "modified_time": base_timestamp + mod_time_offset_secs,
        "total_size": total_size_bytes,
        "runtime_hits": runtime_hits,
    }


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-instance-attributes
class TestPruningService(unittest.TestCase):
    """Test suite for the PruningService."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_cache_repo_instance = MagicMock(spec=CacheRepository)
        self.mock_cache_repo_instance.root = Path("/fake/triton/cache")

        self.mock_db_instance = MagicMock(spec=Database)

        self.patch_cache_repo = patch(
            "model_cache_manager.services.prune.CacheRepository",
            return_value=self.mock_cache_repo_instance,
        )
        self.patch_db = patch(
            "model_cache_manager.services.prune.Database",
            return_value=self.mock_db_instance,
        )

        self.mock_cache_repo_constructor = self.patch_cache_repo.start()
        self.mock_db_constructor = self.patch_db.start()

        self.pruning_service = PruningService(cache_dir=Path("/fake_cache_dir_param"))

        self.kernel1_data = create_mock_kernel_data(
            "hash1", "kernel_one", mod_time_offset_secs=0, runtime_hits=5
        )
        self.kernel2_data = create_mock_kernel_data(
            "hash2",
            "kernel_two",
            backend="rocm",
            mod_time_offset_secs=-3600,
            runtime_hits=50,
        )
        self.kernel3_data = create_mock_kernel_data(
            "hash3", "kernel_one", mod_time_offset_secs=-7200, runtime_hits=150
        )

    def tearDown(self):
        """Clean up after each test method."""
        self.patch_cache_repo.stop()
        self.patch_db.stop()

    @patch(
        "model_cache_manager.services.prune.PruningService._delete_kernel",
        return_value=1024,
    )
    @patch("model_cache_manager.services.prune.Confirm.ask")
    def test_prune_identifies_all_kernels_with_no_criteria_and_auto_confirm(
        self, mock_rich_confirm_ask: MagicMock, mock_delete_kernel: MagicMock
    ):
        """Test pruning all kernels (auto_confirm=True, prompt should not appear)."""
        criteria = SearchCriteria()
        kernels_to_prune = [self.kernel1_data, self.kernel2_data]
        self.mock_db_instance.search.return_value = kernels_to_prune
        self.mock_db_instance.estimate_space.return_value = 2048

        mock_session = MagicMock()
        self.mock_db_instance.get_session.return_value.__enter__.return_value = (
            mock_session
        )

        stats = self.pruning_service.prune(
            criteria, delete_ir_only=False, auto_confirm=True
        )

        self.mock_db_instance.search.assert_called_once_with(criteria)
        self.mock_db_instance.estimate_space.assert_called_once_with(
            ["hash1", "hash2"], IR_EXTS
        )
        mock_rich_confirm_ask.assert_not_called()

        self.assertEqual(mock_delete_kernel.call_count, 2)
        mock_delete_kernel.assert_any_call("hash1", mock_session, False)
        mock_delete_kernel.assert_any_call("hash2", mock_session, False)
        mock_session.commit.assert_called_once()

        self.assertIsInstance(stats, PruneStats)
        self.assertEqual(stats.pruned, 2)
        self.assertAlmostEqual(stats.reclaimed, 2048 / (1024 * 1024))

    @patch(
        "model_cache_manager.services.prune.PruningService._delete_kernel",
        return_value=512,
    )
    @patch("model_cache_manager.services.prune.Confirm.ask")
    def test_prune_identifies_kernels_by_name_and_auto_confirm(
        self, mock_rich_confirm_ask: MagicMock, mock_delete_kernel: MagicMock
    ):
        """Test pruning by name (auto_confirm=True, prompt should not appear)."""
        criteria = SearchCriteria(name="kernel_one")
        kernels_matching_name = [self.kernel1_data, self.kernel3_data]
        self.mock_db_instance.search.return_value = kernels_matching_name
        self.mock_db_instance.estimate_space.return_value = 1024

        mock_session = MagicMock()
        self.mock_db_instance.get_session.return_value.__enter__.return_value = (
            mock_session
        )

        stats = self.pruning_service.prune(
            criteria, delete_ir_only=True, auto_confirm=True
        )

        self.mock_db_instance.search.assert_called_once_with(criteria)
        self.mock_db_instance.estimate_space.assert_called_once_with(
            ["hash1", "hash3"], IR_EXTS
        )
        mock_rich_confirm_ask.assert_not_called()

        self.assertEqual(mock_delete_kernel.call_count, 2)
        mock_delete_kernel.assert_any_call("hash1", mock_session, True)
        mock_delete_kernel.assert_any_call("hash3", mock_session, True)
        mock_session.commit.assert_called_once()

        self.assertEqual(stats.pruned, 2)
        self.assertAlmostEqual(stats.reclaimed, 1024 / (1024 * 1024))

    @patch(
        "model_cache_manager.services.prune.PruningService._delete_kernel",
        return_value=1024,
    )
    @patch("model_cache_manager.services.prune.Confirm.ask")
    def test_prune_filters_by_cache_hit_range_and_auto_confirms(
        self, mock_rich_confirm_ask: MagicMock, mock_delete_kernel: MagicMock
    ):
        """Test pruning correctly filters by cache hit lower and higher bounds."""
        criteria = SearchCriteria(cache_hit_lower=10, cache_hit_higher=100)
        kernels_to_prune = [self.kernel2_data]
        self.mock_db_instance.search.return_value = kernels_to_prune
        self.mock_db_instance.estimate_space.return_value = 1024
        mock_session = MagicMock()
        self.mock_db_instance.get_session.return_value.__enter__.return_value = (
            mock_session
        )

        stats = self.pruning_service.prune(
            criteria, delete_ir_only=False, auto_confirm=True
        )

        self.mock_db_instance.search.assert_called_once_with(criteria)
        self.mock_db_instance.estimate_space.assert_called_once_with(["hash2"], IR_EXTS)
        mock_rich_confirm_ask.assert_not_called()
        mock_delete_kernel.assert_called_once_with("hash2", mock_session, False)
        mock_session.commit.assert_called_once()
        self.assertIsInstance(stats, PruneStats)
        self.assertEqual(stats.pruned, 1)
        self.assertAlmostEqual(stats.reclaimed, 1024 / (1024 * 1024))

    @patch("model_cache_manager.services.prune.Confirm.ask")
    def test_prune_no_kernels_match_criteria(self, mock_rich_confirm_ask: MagicMock):
        """Test prune operation when no kernels match the given criteria."""
        criteria = SearchCriteria(name="non_existent_kernel")
        self.mock_db_instance.search.return_value = []

        stats = self.pruning_service.prune(
            criteria, delete_ir_only=True, auto_confirm=True
        )

        self.mock_db_instance.search.assert_called_once_with(criteria)
        mock_rich_confirm_ask.assert_not_called()
        self.mock_db_instance.estimate_space.assert_not_called()
        self.mock_db_instance.get_session.assert_not_called()

        self.assertIsInstance(stats, PruneStats)
        self.assertEqual(stats.pruned, 0)
        self.assertEqual(stats.reclaimed, 0.0)
        self.assertFalse(stats.aborted)

    @patch("model_cache_manager.services.prune.Confirm.ask", return_value=False)
    @patch("model_cache_manager.services.prune.PruningService._delete_kernel")
    def test_prune_user_cancels_operation_when_prompted(
        self, mock_delete_kernel: MagicMock, mock_rich_confirm_ask: MagicMock
    ):
        """Test prune operation when user cancels (auto_confirm=False)."""
        criteria = SearchCriteria(name="kernel_one")
        self.mock_db_instance.search.return_value = [self.kernel1_data]
        self.mock_db_instance.estimate_space.return_value = 1024

        stats = self.pruning_service.prune(
            criteria, delete_ir_only=True, auto_confirm=False
        )

        self.mock_db_instance.search.assert_called_once_with(criteria)
        self.mock_db_instance.estimate_space.assert_called_once_with(["hash1"], IR_EXTS)
        mock_rich_confirm_ask.assert_called_once()
        mock_delete_kernel.assert_not_called()
        self.mock_db_instance.get_session.assert_not_called()

        self.assertIsNone(
            stats, "Stats should be None when prune is cancelled by user."
        )


if __name__ == "__main__":
    unittest.main()
