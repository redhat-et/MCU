"""
Unit tests for the PruningService.
"""

import json
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
from model_cache_manager.utils.utils import create_kernel_identifier

from model_cache_manager.data.cache_repo import CacheRepository
from model_cache_manager.data.database import Database
from model_cache_manager.tests.test_utils import create_mock_kernel_data

logging.disable(logging.CRITICAL)


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
            "model_cache_manager.strategies.triton_strategy.CacheRepository",
            return_value=self.mock_cache_repo_instance,
        )
        self.patch_db = patch(
            "model_cache_manager.strategies.triton_strategy.Database",
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
        "model_cache_manager.services.prune.PruningService._delete_kernel_unified",
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
            ["hash1", "hash2"], None
        )
        mock_rich_confirm_ask.assert_not_called()

        self.assertEqual(mock_delete_kernel.call_count, 2)
        # Check that the unified method was called with the correct KernelIdentifier objects
        self.assertTrue(any(
            call[0][0].hash_key == "hash1" and call[0][1] == mock_session and call[0][2] is False
            for call in mock_delete_kernel.call_args_list
        ))
        self.assertTrue(any(
            call[0][0].hash_key == "hash2" and call[0][1] == mock_session and call[0][2] is False
            for call in mock_delete_kernel.call_args_list
        ))
        mock_session.commit.assert_called_once()

        self.assertIsInstance(stats, PruneStats)
        self.assertEqual(stats.pruned, 2)
        self.assertAlmostEqual(stats.reclaimed, 2048 / (1024 * 1024))

    @patch(
        "model_cache_manager.services.prune.PruningService._delete_kernel_unified",
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
        # Check that the unified method was called with the correct KernelIdentifier objects
        self.assertTrue(any(
            call[0][0].hash_key == "hash1" and call[0][1] == mock_session and call[0][2] is True
            for call in mock_delete_kernel.call_args_list
        ))
        self.assertTrue(any(
            call[0][0].hash_key == "hash3" and call[0][1] == mock_session and call[0][2] is True
            for call in mock_delete_kernel.call_args_list
        ))
        mock_session.commit.assert_called_once()

        self.assertEqual(stats.pruned, 2)
        self.assertAlmostEqual(stats.reclaimed, 1024 / (1024 * 1024))

    @patch(
        "model_cache_manager.services.prune.PruningService._delete_kernel_unified",
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
        self.mock_db_instance.estimate_space.assert_called_once_with(["hash2"], None)
        mock_rich_confirm_ask.assert_not_called()
        # Check that the unified method was called with the correct KernelIdentifier object
        self.assertEqual(mock_delete_kernel.call_count, 1)
        call_args = mock_delete_kernel.call_args[0]
        self.assertEqual(call_args[0].hash_key, "hash2")
        self.assertEqual(call_args[1], mock_session)
        self.assertEqual(call_args[2], False)
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
    @patch("model_cache_manager.services.prune.PruningService._delete_kernel_unified")
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

    @patch(
        "model_cache_manager.services.prune.PruningService._delete_kernel_unified",
        return_value=1024,
    )
    @patch("model_cache_manager.services.prune.Confirm.ask")
    def test_prune_non_best_kernels_only(
        self, mock_rich_confirm_ask: MagicMock, mock_delete_kernel: MagicMock
    ):
        """Test pruning only non-best kernels when only_best=False."""
        # Create mock kernel data for vLLM mode with best_config
        # Note: kernel1_best is not included in kernels_to_prune since it's a best kernel
        # and we're testing pruning of non-best kernels only

        # Kernel 2: This is a non-best kernel (triton_cache_key doesn't match)
        kernel2_non_best = {
            "hash": "hash2",
            "triton_cache_key": "tck2",
            "vllm_hash": "vllm1",
            "name": "kernel_one",
            "backend": "cuda",
            "arch": "ampere",
            "best_config": json.dumps({"triton_cache_hash": "tck1"}),  # Doesn't match
            "is_best": False,
            "total_size": 1024,
        }

        # Kernel 3: Another non-best kernel
        kernel3_non_best = {
            "hash": "hash3",
            "triton_cache_key": "tck3",
            "vllm_hash": "vllm2",
            "name": "kernel_two",
            "backend": "cuda",
            "arch": "ampere",
            "best_config": json.dumps({"triton_cache_hash": "tck4"}),  # Doesn't match
            "is_best": False,
            "total_size": 1024,
        }

        # When only_best=False, we want to prune only non-best kernels
        criteria = SearchCriteria(only_best=False)

        # The database search should return only non-best kernels when only_best=False
        # (This filtering happens in the database layer)
        kernels_to_prune = [kernel2_non_best, kernel3_non_best]
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
        # Only non-best kernels should be pruned
        self.mock_db_instance.estimate_space.assert_called_once_with(
            ["hash2", "hash3"], None
        )
        mock_rich_confirm_ask.assert_not_called()

        # Verify that only the non-best kernels were deleted
        self.assertEqual(mock_delete_kernel.call_count, 2)
        deleted_hashes = [call[0][0].hash_key for call in mock_delete_kernel.call_args_list]
        self.assertIn("hash2", deleted_hashes)
        self.assertIn("hash3", deleted_hashes)

        mock_session.commit.assert_called_once()

        self.assertIsInstance(stats, PruneStats)
        self.assertEqual(stats.pruned, 2)
        self.assertAlmostEqual(stats.reclaimed, 2048 / (1024 * 1024))

    @patch(
        "model_cache_manager.services.prune.PruningService._delete_kernel_unified",
        return_value=1024,
    )
    @patch("model_cache_manager.services.prune.Confirm.ask")
    def test_prune_only_best_kernels(
        self, mock_rich_confirm_ask: MagicMock, mock_delete_kernel: MagicMock
    ):
        """Test pruning only best kernels when only_best=True."""
        # Kernel 1: This is a best kernel
        kernel1_best = {
            "hash": "hash1",
            "triton_cache_key": "tck1",
            "vllm_hash": "vllm1",
            "name": "kernel_one",
            "backend": "cuda",
            "arch": "ampere",
            "best_config": json.dumps({"triton_cache_hash": "tck1"}),
            "is_best": True,
            "total_size": 1024,
        }

        # Kernel 2: This is also a best kernel
        kernel2_best = {
            "hash": "hash2",
            "triton_cache_key": "tck2",
            "vllm_hash": "vllm2",
            "name": "kernel_two",
            "backend": "cuda",
            "arch": "ampere",
            "best_config": json.dumps({"triton_cache_hash": "tck2"}),
            "is_best": True,
            "total_size": 1024,
        }

        # When only_best=True, we want to prune only best kernels
        criteria = SearchCriteria(only_best=True)

        # The database search should return only best kernels when only_best=True
        kernels_to_prune = [kernel1_best, kernel2_best]
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
        # Only best kernels should be pruned
        self.mock_db_instance.estimate_space.assert_called_once_with(
            ["hash1", "hash2"], None
        )
        mock_rich_confirm_ask.assert_not_called()

        # Verify that only the best kernels were deleted
        self.assertEqual(mock_delete_kernel.call_count, 2)
        deleted_hashes = [call[0][0].hash_key for call in mock_delete_kernel.call_args_list]
        self.assertIn("hash1", deleted_hashes)
        self.assertIn("hash2", deleted_hashes)

        mock_session.commit.assert_called_once()

        self.assertIsInstance(stats, PruneStats)
        self.assertEqual(stats.pruned, 2)
        self.assertAlmostEqual(stats.reclaimed, 2048 / (1024 * 1024))

    @patch(
        "model_cache_manager.services.prune.PruningService._delete_kernel_unified",
        return_value=1024,
    )
    @patch("model_cache_manager.services.prune.Confirm.ask")
    def test_prune_vllm_with_artifact_compile_range(
        self, _mock_rich_confirm_ask: MagicMock, mock_delete_kernel: MagicMock
    ):
        """Test pruning vLLM kernels with artifact_compile_range in primary key."""
        # Create mock kernel data for new vLLM mode with artifact_compile_range
        kernel_with_shape = {
            "hash": "hash1",
            "triton_cache_key": "tck1",
            "vllm_hash": "vllm1",
            "rank_x_y": "0_0",
            "artifact_compile_range": "1024x768",  # New vLLM structure includes this
            "name": "kernel_one",
            "backend": "cuda",
            "arch": "ampere",
            "total_size": 1024,
        }

        criteria = SearchCriteria()
        kernels_to_prune = [kernel_with_shape]
        self.mock_db_instance.search.return_value = kernels_to_prune
        self.mock_db_instance.estimate_space.return_value = 1024

        mock_session = MagicMock()
        self.mock_db_instance.get_session.return_value.__enter__.return_value = (
            mock_session
        )

        # Patch the strategy to return vllm mode
        with patch.object(
            self.pruning_service, 'mode', 'vllm'
        ), patch.object(
            self.pruning_service.strategy, 'extract_identifiers_from_row'
        ) as mock_extract:
            # Create a KernelIdentifier with artifact_compile_range
            identifier = create_kernel_identifier(
                mode='vllm',
                vllm_hash='vllm1',
                triton_cache_key='tck1',
                rank_x_y='0_0',
            )
            identifier.artifact_compile_range = '1024x768'
            mock_extract.return_value = identifier

            stats = self.pruning_service.prune(
                criteria, delete_ir_only=False, auto_confirm=True
            )

            # Verify the identifier passed to delete includes artifact_compile_range
            self.assertEqual(mock_delete_kernel.call_count, 1)
            call_args = mock_delete_kernel.call_args[0]
            passed_identifier = call_args[0]
            self.assertEqual(passed_identifier.artifact_compile_range, '1024x768')
            self.assertEqual(passed_identifier.vllm_hash, 'vllm1')
            self.assertEqual(passed_identifier.hash_key, 'tck1')
            self.assertEqual(passed_identifier.rank_x_y, '0_0')

            self.assertIsInstance(stats, PruneStats)
            self.assertEqual(stats.pruned, 1)


if __name__ == "__main__":
    unittest.main()
