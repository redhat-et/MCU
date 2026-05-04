"""
Unit tests for PruningService with Helion mode.
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import logging

from model_cache_manager.services.prune import PruningService, PruneStats
from model_cache_manager.models.criteria import SearchCriteria
from model_cache_manager.utils.mcm_constants import IR_EXTS, MODE_HELION
from model_cache_manager.utils.utils import create_kernel_identifier

from model_cache_manager.data.cache_repo import HelionCacheRepository
from model_cache_manager.data.database import HelionDatabase

logging.disable(logging.CRITICAL)


class TestHelionPruningService(unittest.TestCase):
    """Test suite for PruningService in Helion mode."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_repo = MagicMock(spec=HelionCacheRepository)
        self.mock_repo.root = Path("/fake/helion/cache")

        self.mock_db = MagicMock(spec=HelionDatabase)

        self.patch_repo = patch(
            "model_cache_manager.strategies.helion_strategy.HelionCacheRepository",
            return_value=self.mock_repo,
        )
        self.patch_db = patch(
            "model_cache_manager.strategies.helion_strategy.HelionDatabase",
            return_value=self.mock_db,
        )

        self.patch_repo.start()
        self.patch_db.start()

        self.svc = PruningService(
            cache_dir=Path("/fake/helion/cache"), mode=MODE_HELION
        )

        self.helion_kernel = {
            "triton_cache_key": "KERNEL_ABC",
            "name": "_helion_add",
            "backend": "cuda",
            "arch": "89",
            "helion_hash": "abc123",
            "best_config": '{"backend_cache_key": "KERNEL_ABC"}',
            "is_best": True,
            "total_size": 2048,
        }

    def tearDown(self):
        """Clean up after each test method."""
        self.patch_repo.stop()
        self.patch_db.stop()

    @patch(
        "model_cache_manager.services.prune.PruningService._delete_kernel_unified",
        return_value=2048,
    )
    @patch("model_cache_manager.services.prune.Confirm.ask")
    def test_prune_ir_only_helion(self, _mock_confirm, mock_delete):
        """Test IR-only prune for helion kernels."""
        criteria = SearchCriteria()
        self.mock_db.search.return_value = [self.helion_kernel]
        self.mock_db.estimate_space.return_value = 1024

        mock_session = MagicMock()
        self.mock_db.get_session.return_value.__enter__.return_value = mock_session

        stats = self.svc.prune(criteria, delete_ir_only=True, auto_confirm=True)

        self.mock_db.search.assert_called_once_with(criteria)
        self.mock_db.estimate_space.assert_called_once_with(
            ["KERNEL_ABC"], IR_EXTS
        )

        self.assertEqual(mock_delete.call_count, 1)
        call_args = mock_delete.call_args[0]
        self.assertEqual(call_args[0].hash_key, "KERNEL_ABC")
        self.assertTrue(call_args[2])  # ir_only=True

        self.assertIsInstance(stats, PruneStats)
        self.assertEqual(stats.pruned, 1)

    @patch(
        "model_cache_manager.services.prune.PruningService._delete_kernel_unified",
        return_value=4096,
    )
    @patch("model_cache_manager.services.prune.Confirm.ask")
    def test_prune_full_helion(self, _mock_confirm, mock_delete):
        """Test full prune for helion kernels."""
        criteria = SearchCriteria()
        self.mock_db.search.return_value = [self.helion_kernel]
        self.mock_db.estimate_space.return_value = 4096

        mock_session = MagicMock()
        self.mock_db.get_session.return_value.__enter__.return_value = mock_session

        stats = self.svc.prune(criteria, delete_ir_only=False, auto_confirm=True)

        self.mock_db.estimate_space.assert_called_once_with(
            ["KERNEL_ABC"], None
        )
        self.assertEqual(mock_delete.call_count, 1)
        call_args = mock_delete.call_args[0]
        self.assertFalse(call_args[2])  # ir_only=False

        self.assertIsInstance(stats, PruneStats)
        self.assertEqual(stats.pruned, 1)

    @patch("model_cache_manager.services.prune.delete_ir_files_from_dirs")
    @patch("model_cache_manager.services.prune.get_kernel_directories")
    def test_delete_ir_file_records_uses_helion_fk(  # pylint: disable=no-member
        self, mock_get_dirs, mock_delete_ir
    ):
        """Test that _delete_ir_file_records queries by triton_cache_key for helion."""
        mock_get_dirs.return_value = [Path("/fake/triton/0/KERNEL_ABC")]
        mock_delete_ir.return_value = (512, ["_helion_add.ttir", "_helion_add.llir"])

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = []

        mock_kernel_record = MagicMock()
        mock_kernel_record.files = []
        mock_session.get.return_value = mock_kernel_record

        identifier = create_kernel_identifier(mode=MODE_HELION, hash="KERNEL_ABC")

        # pylint: disable=protected-access
        self.svc._delete_kernel_unified(identifier, mock_session, ir_only=True)

        mock_session.query.assert_called()
        mock_query.filter.assert_called()
        filter_args = mock_query.filter.call_args[0]

        found_triton_key_filter = False
        found_cache_dir_filter = False
        for arg in filter_args:
            arg_str = str(arg)
            if "triton_cache_key" in arg_str:
                found_triton_key_filter = True
            if "cache_dir" in arg_str:
                found_cache_dir_filter = True

        self.assertTrue(
            found_triton_key_filter,
            "Filter should use triton_cache_key for helion mode"
        )
        self.assertTrue(
            found_cache_dir_filter,
            "Filter should include cache_dir for helion mode"
        )

    @patch("model_cache_manager.services.prune.delete_kernel_directories")
    @patch("model_cache_manager.services.prune.get_kernel_directories")
    def test_full_delete_removes_kernel_record(
        self, mock_get_dirs, mock_delete_dirs
    ):
        """Test that full prune deletes the kernel record from DB."""
        mock_get_dirs.return_value = [Path("/fake/triton/0/KERNEL_ABC")]
        mock_delete_dirs.return_value = 4096

        mock_session = MagicMock()
        mock_kernel_record = MagicMock()
        mock_session.get.return_value = mock_kernel_record

        identifier = create_kernel_identifier(mode=MODE_HELION, hash="KERNEL_ABC")

        # pylint: disable=protected-access
        self.svc._delete_kernel_unified(identifier, mock_session, ir_only=False)

        mock_session.delete.assert_called_once_with(mock_kernel_record)


if __name__ == "__main__":
    unittest.main()
