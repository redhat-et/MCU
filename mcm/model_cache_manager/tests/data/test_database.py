"""Test suite for the Database class."""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from model_cache_manager.models.kernel import Kernel, KernelFile
from model_cache_manager.tests.test_utils import (
    setup_kernel_orm_mock,
    setup_sqlite_insert_mock,
    setup_query_mock,
    setup_tuple_mock,
    setup_engine_and_session_mock,
)


def create_mock_kernel(hash_val="test_hash", name="test_kernel"):
    """Create a mock kernel for testing."""
    mock_kernel = MagicMock(spec=Kernel)
    mock_kernel.hash = hash_val
    mock_kernel.name = name
    mock_kernel.backend = "cuda"
    mock_kernel.arch = "80"
    mock_kernel.triton_version = "3.3.0"
    mock_kernel.metadata = {"key": "value"}
    mock_kernel.modified_time = 1234567890.0
    mock_kernel.cache_dir = "/test/cache"

    # Mock files
    mock_file1 = MagicMock(spec=KernelFile)
    mock_file1.file_type = "ptx"
    mock_file1.path = Path("kernel.ptx")
    mock_file1.size = 1024

    mock_file2 = MagicMock(spec=KernelFile)
    mock_file2.file_type = "ttir"
    mock_file2.path = Path("kernel.ttir")
    mock_file2.size = 512

    mock_kernel.files = [mock_file1, mock_file2]

    for attr in [
        "warp_size",
        "num_warps",
        "num_stages",
        "num_ctas",
        "maxnreg",
        "cluster_dims",
        "ptx_version",
        "enable_fp_fusion",
        "launch_cooperative_grid",
        "supported_fp8_dtypes",
        "deprecated_fp8_dtypes",
        "default_dot_input_precision",
        "allowed_dot_input_precisions",
        "max_num_imprecise_acc_default",
        "extern_libs",
        "debug",
        "backend_name",
        "sanitize_overflow",
        "shared",
        "tmem_size",
        "global_scratch_size",
        "global_scratch_align",
        "waves_per_eu",
        "kpack",
        "matrix_instr_nonkdim",
    ]:
        setattr(mock_kernel, attr, None)

    return mock_kernel


class TestDatabase(unittest.TestCase):
    """Test suite for the Database."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_kernel = create_mock_kernel()

    @patch("model_cache_manager.data.database.tuple_")
    @patch("model_cache_manager.data.database.sqlite_insert")
    @patch("model_cache_manager.data.database.KernelFileOrm")
    @patch("model_cache_manager.data.database.KernelOrm")
    @patch("model_cache_manager.data.database.Base")
    @patch("model_cache_manager.data.database.create_engine_and_session")
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def test_bulk_insert_kernels_success(
        self,
        mock_create_engine_session,
        mock_base,  # pylint: disable=unused-argument
        mock_kernel_orm,
        mock_file_orm,
        mock_sqlite_insert,
        mock_tuple,
    ):
        # pylint: enable=too-many-arguments,too-many-positional-arguments,too-many-locals
        """Test successful bulk kernel insertion for regular Database."""
        _, _, mock_session = setup_engine_and_session_mock(
            mock_create_engine_session
        )

        # Setup file ORM mock attributes
        mock_file_orm.kernel_hash = MagicMock()
        mock_file_orm.kernel_cache_dir = MagicMock()

        # pylint: disable=import-outside-toplevel
        from model_cache_manager.data.database import Database

        # pylint: enable=import-outside-toplevel

        db = Database()

        db.get_session = MagicMock(return_value=mock_session)

        # Use shared setup utilities
        setup_kernel_orm_mock(mock_kernel_orm)
        setup_sqlite_insert_mock(mock_sqlite_insert)
        setup_query_mock(mock_session)

        mock_query = MagicMock()

        def mock_query_func(entity):  # pylint: disable=unused-argument
            return mock_query

        mock_session.query = mock_query_func

        # Use shared tuple mock setup
        setup_tuple_mock(mock_tuple)

        # Create test data
        mock_kernel2 = create_mock_kernel("hash2", "kernel2")

        kernels_data = [
            (self.mock_kernel, "/test/cache"),
            (mock_kernel2, "/test/cache2"),
        ]

        result = db.bulk_insert_kernels(kernels_data, batch_size=2)

        # Verify the correct number of kernels were inserted
        self.assertEqual(result, 2)

        # Verify bulk operations were used
        mock_session.bulk_insert_mappings.assert_called_once_with(
            mock_file_orm,
            [
                {
                    "kernel_hash": "test_hash",
                    "kernel_cache_dir": "/test/cache",
                    "type": "ptx",
                    "rel_path": "kernel.ptx",
                    "size": 1024,
                },
                {
                    "kernel_hash": "test_hash",
                    "kernel_cache_dir": "/test/cache",
                    "type": "ttir",
                    "rel_path": "kernel.ttir",
                    "size": 512,
                },
                {
                    "kernel_hash": "hash2",
                    "kernel_cache_dir": "/test/cache2",
                    "type": "ptx",
                    "rel_path": "kernel.ptx",
                    "size": 1024,
                },
                {
                    "kernel_hash": "hash2",
                    "kernel_cache_dir": "/test/cache2",
                    "type": "ttir",
                    "rel_path": "kernel.ttir",
                    "size": 512,
                },
            ],
        )
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
