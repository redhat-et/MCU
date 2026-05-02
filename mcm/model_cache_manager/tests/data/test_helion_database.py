"""
Unit tests for the HelionDatabase and HelionKernelOrm.
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
import time

from model_cache_manager.data.database import HelionDatabase
from model_cache_manager.data.db_models import HelionKernelOrm
from model_cache_manager.models.kernel import Kernel, KernelFile
from model_cache_manager.models.criteria import SearchCriteria
from model_cache_manager.tests.test_utils import (
    setup_kernel_orm_mock,
    setup_sqlite_insert_mock,
    setup_query_mock,
    setup_tuple_mock,
    setup_engine_and_session_mock,
)


def create_mock_kernel(
    hash_val="test_hash",
    name="_helion_add",
    backend="cuda",
    arch="89",
    files=None,
):
    """Helper to create a mock Kernel object."""
    if files is None:
        f1 = Mock(spec=KernelFile)
        f1.path = Path("/test/_helion_add.ptx")
        f1.file_type = "ptx"
        f1.size = 1024

        f2 = Mock(spec=KernelFile)
        f2.path = Path("/test/_helion_add.json")
        f2.file_type = "json"
        f2.size = 512

        files = [f1, f2]

    kernel = Mock(spec=Kernel)
    kernel.hash = hash_val
    kernel.name = name
    kernel.backend = backend
    kernel.arch = arch
    kernel.files = files
    kernel.metadata = {"test": "metadata"}
    kernel.modified_time = time.time()
    kernel.warp_size = 32
    kernel.num_warps = 2
    kernel.num_stages = 1
    kernel.num_ctas = 1
    kernel.maxnreg = None
    kernel.cluster_dims = None
    kernel.ptx_version = None
    kernel.enable_fp_fusion = True
    kernel.launch_cooperative_grid = False
    kernel.supported_fp8_dtypes = []
    kernel.deprecated_fp8_dtypes = []
    kernel.default_dot_input_precision = "tf32"
    kernel.allowed_dot_input_precisions = ["tf32", "ieee"]
    kernel.max_num_imprecise_acc_default = 0
    kernel.extern_libs = []
    kernel.debug = False
    kernel.backend_name = "cuda"
    kernel.sanitize_overflow = False
    kernel.triton_version = "3.3.0"
    kernel.shared = 0
    kernel.tmem_size = None
    kernel.global_scratch_size = None
    kernel.global_scratch_align = None
    kernel.waves_per_eu = None
    kernel.kpack = None
    kernel.matrix_instr_nonkdim = None

    return kernel


class TestHelionKernelOrm(unittest.TestCase):
    """Test suite for the HelionKernelOrm."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = MagicMock()
        self.mock_kernel = create_mock_kernel()

    def test_upsert_from_dto(self):
        """Test upsert_from_dto method."""
        cache_dir = "/test/helion/cache"

        with patch(
            "model_cache_manager.data.db_models.sqlite_insert"
        ) as mock_insert, patch(
            "model_cache_manager.data.db_models.HelionKernelFileOrm"
        ):
            mock_stmt = MagicMock()
            mock_insert.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            HelionKernelOrm.upsert_from_dto(
                self.mock_session,
                self.mock_kernel,
                cache_dir,
                helion_hash="abc123",
                best_config='{"backend_cache_key": "test_hash"}',
                is_best=True,
            )

            mock_insert.assert_called_once_with(HelionKernelOrm)
            self.mock_session.execute.assert_called_once()
            self.mock_session.query.assert_called()  # pylint: disable=no-member

    def test_get_helion_kernel_values(self):
        """Test get_helion_kernel_values returns correct dict."""
        values = HelionKernelOrm.get_helion_kernel_values(
            self.mock_kernel, "/cache", "h1", '{"key": "val"}', True
        )

        self.assertEqual(values["triton_cache_key"], "test_hash")
        self.assertEqual(values["cache_dir"], "/cache")
        self.assertEqual(values["helion_hash"], "h1")
        self.assertEqual(values["best_config"], '{"key": "val"}')
        self.assertTrue(values["is_best"])
        self.assertEqual(values["name"], "_helion_add")

    def test_to_dict(self):
        """Test to_dict maps kernel_metadata_json to metadata."""
        orm = HelionKernelOrm()
        orm.triton_cache_key = "key1"
        orm.cache_dir = "/cache"
        orm.kernel_metadata_json = {"test": "data"}

        mock_col1 = MagicMock()
        mock_col1.key = "triton_cache_key"
        mock_col2 = MagicMock()
        mock_col2.key = "cache_dir"
        mock_col3 = MagicMock()
        mock_col3.key = "kernel_metadata_json"

        mock_table = MagicMock()
        mock_table.columns = [mock_col1, mock_col2, mock_col3]
        orm.__table__ = mock_table

        result = orm.to_dict()

        self.assertIn("metadata", result)
        self.assertNotIn("kernel_metadata_json", result)
        self.assertEqual(result["metadata"], {"test": "data"})


class TestHelionDatabase(unittest.TestCase):
    """Test suite for the HelionDatabase."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_kernel = create_mock_kernel()

    @patch("model_cache_manager.data.database.create_engine_and_session")
    def test_init(self, mock_create_engine_session):
        """Test HelionDatabase initialization."""
        mock_engine, mock_session_local, _ = setup_engine_and_session_mock(
            mock_create_engine_session
        )

        with patch("model_cache_manager.data.database_utils.ensure_schema") as mock_ensure:
            db = HelionDatabase()

            mock_create_engine_session.assert_called_once_with("helion")
            self.assertEqual(db.engine, mock_engine)
            self.assertEqual(db.SessionLocal, mock_session_local)
            mock_ensure.assert_called_once_with(mock_engine)

    @patch("model_cache_manager.data.database.create_engine_and_session")
    def test_insert_kernel_success(self, mock_create_engine_session):
        """Test successful kernel insertion."""
        _, _, mock_session = setup_engine_and_session_mock(mock_create_engine_session)

        with patch("model_cache_manager.data.database_utils.ensure_schema"), patch(
            "model_cache_manager.data.database.HelionKernelOrm"
        ) as mock_orm:
            db = HelionDatabase()

            db.insert_kernel(
                self.mock_kernel, "/cache", "h1", '{"key": "val"}', True
            )

            mock_orm.upsert_from_dto.assert_called_once()
            args = mock_orm.upsert_from_dto.call_args[0]
            self.assertEqual(args[0], mock_session)
            self.assertEqual(args[1], self.mock_kernel)
            self.assertEqual(args[2], "/cache")
            self.assertEqual(args[3], "h1")
            self.assertEqual(args[4], '{"key": "val"}')
            self.assertTrue(args[5])
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()

    @patch("model_cache_manager.data.database.create_engine_and_session")
    def test_bulk_insert_kernels_success(self, mock_create_engine_session):
        """Test successful bulk kernel insertion."""
        _, _, mock_session = setup_engine_and_session_mock(mock_create_engine_session)

        class MockHelionFileOrm:  # pylint: disable=too-few-public-methods
            """Mock ORM class."""
            triton_cache_key = MagicMock()
            cache_dir = MagicMock()

        with patch("model_cache_manager.data.database_utils.ensure_schema"), patch(
            "model_cache_manager.data.database.HelionKernelOrm"
        ) as mock_orm, patch(
            "model_cache_manager.data.database.HelionKernelFileOrm", MockHelionFileOrm
        ), patch(
            "model_cache_manager.data.database.sqlite_insert"
        ) as mock_sqlite_insert, patch(
            "model_cache_manager.data.database.tuple_"
        ) as mock_tuple:
            db = HelionDatabase()

            setup_kernel_orm_mock(mock_orm)
            setup_sqlite_insert_mock(mock_sqlite_insert)
            setup_query_mock(mock_session)
            setup_tuple_mock(mock_tuple)

            kernels_data = [
                (self.mock_kernel, "/cache", "h1", '{"key":"v"}', True),
            ]

            result = db.bulk_insert_kernels(kernels_data, batch_size=10)

            self.assertEqual(result, 1)
            mock_session.bulk_insert_mappings.assert_called_once()
            file_dicts = mock_session.bulk_insert_mappings.call_args[0][1]
            self.assertEqual(len(file_dicts), 2)
            self.assertEqual(file_dicts[0]["triton_cache_key"], "test_hash")
            self.assertEqual(file_dicts[0]["cache_dir"], "/cache")
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()

    @patch("model_cache_manager.data.database.create_engine_and_session")
    def test_bulk_insert_empty(self, mock_create_engine_session):
        """Test bulk insert with empty list."""
        setup_engine_and_session_mock(mock_create_engine_session)

        with patch("model_cache_manager.data.database_utils.ensure_schema"):
            db = HelionDatabase()
            result = db.bulk_insert_kernels([])
            self.assertEqual(result, 0)

    @patch("model_cache_manager.data.database.create_engine_and_session")
    def test_search_with_criteria(self, mock_create_engine_session):
        """Test search method with criteria."""
        _, _, mock_session = setup_engine_and_session_mock(mock_create_engine_session)

        mock_orm_result = MagicMock()
        mock_orm_result.to_dict.return_value = {
            "triton_cache_key": "key1",
            "name": "_helion_add",
        }
        mock_orm_result.is_best = True
        mock_orm_result.best_config = None

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = [mock_orm_result]
        mock_session.query.return_value = mock_query

        with patch("model_cache_manager.data.database_utils.ensure_schema"), patch(
            "model_cache_manager.data.database.HelionKernelOrm"
        ):
            db = HelionDatabase()
            criteria = SearchCriteria(name="_helion_add")
            results = db.search(criteria)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["triton_cache_key"], "key1")
            self.assertTrue(results[0]["is_best"])
            mock_session.close.assert_called_once()

    @patch("model_cache_manager.data.database.create_engine_and_session")
    def test_search_filters_by_only_best(self, mock_create_engine_session):
        """Test search respects only_best filtering."""
        _, _, mock_session = setup_engine_and_session_mock(mock_create_engine_session)

        best_kernel = MagicMock()
        best_kernel.to_dict.return_value = {"triton_cache_key": "k1"}
        best_kernel.is_best = True
        best_kernel.best_config = None

        non_best_kernel = MagicMock()
        non_best_kernel.to_dict.return_value = {"triton_cache_key": "k2"}
        non_best_kernel.is_best = False
        non_best_kernel.best_config = None

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = [best_kernel, non_best_kernel]
        mock_session.query.return_value = mock_query

        with patch("model_cache_manager.data.database_utils.ensure_schema"), patch(
            "model_cache_manager.data.database.HelionKernelOrm"
        ):
            db = HelionDatabase()

            results = db.search(SearchCriteria(only_best=True))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["triton_cache_key"], "k1")

            results = db.search(SearchCriteria(only_best=False))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["triton_cache_key"], "k2")

    @patch("model_cache_manager.data.database.create_engine_and_session")
    def test_close(self, mock_create_engine_session):
        """Test database close method."""
        mock_engine, _, _ = setup_engine_and_session_mock(mock_create_engine_session)

        with patch("model_cache_manager.data.database_utils.ensure_schema"):
            db = HelionDatabase()
            db.close()
            mock_engine.dispose.assert_called_once()


if __name__ == "__main__":
    unittest.main()
