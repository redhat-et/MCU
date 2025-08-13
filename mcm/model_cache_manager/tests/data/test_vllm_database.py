"""
Unit tests for the VllmDatabase and VllmKernelOrm.
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
import time

from model_cache_manager.data.database import VllmDatabase
from model_cache_manager.data.db_models import VllmKernelOrm, BaseKernelMixin
from model_cache_manager.models.kernel import Kernel, KernelFile
from model_cache_manager.models.criteria import SearchCriteria


def create_mock_kernel(
    hash_val: str = "test_hash",
    name: str = "test_kernel",
    backend: str = "cuda",
    arch: str = "80",
    files: list = None,
) -> Kernel:
    """Helper to create a mock Kernel object."""
    if files is None:
        mock_file1 = Mock(spec=KernelFile)
        mock_file1.path = Path("/test/kernel.ptx")
        mock_file1.file_type = "ptx"
        mock_file1.size = 1024

        mock_file2 = Mock(spec=KernelFile)
        mock_file2.path = Path("/test/kernel.json")
        mock_file2.file_type = "json"
        mock_file2.size = 512

        files = [mock_file1, mock_file2]

    kernel = Mock(spec=Kernel)
    kernel.hash = hash_val
    kernel.name = name
    kernel.backend = backend
    kernel.arch = arch
    kernel.files = files
    kernel.metadata = {"test": "metadata"}
    kernel.modified_time = time.time()
    kernel.warp_size = 32
    kernel.num_warps = 4
    kernel.num_stages = 2
    kernel.num_ctas = 1
    kernel.maxnreg = None
    kernel.cluster_dims = None
    kernel.ptx_version = "7.0"
    kernel.enable_fp_fusion = True
    kernel.launch_cooperative_grid = False
    kernel.supported_fp8_dtypes = []
    kernel.deprecated_fp8_dtypes = []
    kernel.default_dot_input_precision = "fp16"
    kernel.allowed_dot_input_precisions = ["fp16", "fp32"]
    kernel.max_num_imprecise_acc_default = None
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


class TestBaseKernelMixin(unittest.TestCase):
    """Test suite for the BaseKernelMixin."""

    def test_get_common_kernel_values(self):
        """Test _get_common_kernel_values method."""
        mock_kernel = create_mock_kernel()

        values = BaseKernelMixin._get_common_kernel_values(mock_kernel)  # pylint: disable=protected-access

        self.assertEqual(values["backend"], "cuda")
        self.assertEqual(values["arch"], "80")
        self.assertEqual(values["name"], "test_kernel")
        self.assertEqual(values["triton_version"], "3.3.0")
        self.assertEqual(values["total_size"], 1536)  # 1024 + 512


class TestVllmKernelOrm(unittest.TestCase):
    """Test suite for the VllmKernelOrm."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = MagicMock()
        self.mock_kernel = create_mock_kernel()

    def test_upsert_from_dto(self):
        """Test upsert_from_dto method."""
        vllm_cache_root = "/test/vllm/cache"
        vllm_hash = "test_vllm_hash"

        with patch('model_cache_manager.data.db_models.sqlite_insert') as mock_insert, \
             patch('model_cache_manager.data.db_models.VllmKernelFileOrm'):

            mock_stmt = MagicMock()
            mock_insert.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            VllmKernelOrm.upsert_from_dto(
                self.mock_session,
                self.mock_kernel,
                vllm_cache_root,
                vllm_hash
            )

            # Verify sqlite_insert was called
            mock_insert.assert_called_once_with(VllmKernelOrm)

            # Verify kernel values were set correctly
            if hasattr(mock_stmt.values, 'call_args') and mock_stmt.values.call_args:
                kernel_values = mock_stmt.values.call_args[0][0]
                self.assertEqual(kernel_values["vllm_cache_root"], vllm_cache_root)
                self.assertEqual(kernel_values["vllm_hash"], vllm_hash)
                self.assertEqual(kernel_values["triton_cache_key"], "test_hash")

            # Verify session operations
            self.mock_session.execute.assert_called_once()
            self.mock_session.query.assert_called_once()

    def test_to_dict(self):
        """Test to_dict method inherited from BaseKernelMixin."""
        # Create a mock VllmKernelOrm instance
        vllm_kernel = VllmKernelOrm()
        vllm_kernel.vllm_cache_root = "/test/cache"
        vllm_kernel.vllm_hash = "test_hash"
        vllm_kernel.triton_cache_key = "triton_key"
        vllm_kernel.kernel_metadata_json = {"test": "metadata"}

        # Mock the __table__.columns attribute
        mock_column1 = MagicMock()
        mock_column1.key = "vllm_cache_root"
        mock_column2 = MagicMock()
        mock_column2.key = "vllm_hash"
        mock_column3 = MagicMock()
        mock_column3.key = "triton_cache_key"
        mock_column4 = MagicMock()
        mock_column4.key = "kernel_metadata_json"

        mock_table = MagicMock()
        mock_table.columns = [mock_column1, mock_column2, mock_column3, mock_column4]
        vllm_kernel.__table__ = mock_table

        result = vllm_kernel.to_dict()

        # Should have metadata instead of kernel_metadata_json
        self.assertIn("metadata", result)
        self.assertNotIn("kernel_metadata_json", result)
        self.assertEqual(result["metadata"], {"test": "metadata"})


class TestVllmDatabase(unittest.TestCase):
    """Test suite for the VllmDatabase."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_kernel = create_mock_kernel()

    @patch('model_cache_manager.data.database.create_engine_and_session')
    def test_init(self, mock_create_engine_session):
        """Test VllmDatabase initialization."""
        mock_engine = MagicMock()
        mock_session_local = MagicMock()
        mock_create_engine_session.return_value = (mock_engine, mock_session_local)

        with patch('model_cache_manager.data.database.Base') as mock_base:
            db = VllmDatabase()

            mock_create_engine_session.assert_called_once_with("vllm")
            self.assertEqual(db.engine, mock_engine)
            self.assertEqual(db.SessionLocal, mock_session_local)
            mock_base.metadata.create_all.assert_called_once_with(bind=mock_engine)

    @patch('model_cache_manager.data.database.create_engine_and_session')
    def test_insert_kernel_success(self, mock_create_engine_session):
        """Test successful kernel insertion."""
        mock_engine = MagicMock()
        mock_session_local = MagicMock()
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_create_engine_session.return_value = (mock_engine, mock_session_local)

        with patch('model_cache_manager.data.database.Base'), \
             patch('model_cache_manager.data.database.VllmKernelOrm') as mock_kernel_orm:

            db = VllmDatabase()
            vllm_cache_root = "/test/cache"
            vllm_hash = "test_hash"

            db.insert_kernel(self.mock_kernel, vllm_cache_root, vllm_hash)

            mock_kernel_orm.upsert_from_dto.assert_called_once_with(
                mock_session, self.mock_kernel, vllm_cache_root, vllm_hash
            )
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()

    @patch('model_cache_manager.data.database.create_engine_and_session')
    def test_search_with_criteria(self, mock_create_engine_session):
        """Test search method with various criteria."""
        mock_engine = MagicMock()
        mock_session_local = MagicMock()
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_create_engine_session.return_value = (mock_engine, mock_session_local)

        # Mock query results
        mock_kernel_orm = MagicMock()
        mock_kernel_orm.to_dict.return_value = {"hash": "test_hash", "name": "test_kernel"}
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = [mock_kernel_orm]
        mock_session.query.return_value = mock_query

        with patch('model_cache_manager.data.database.Base'), \
             patch('model_cache_manager.data.database.VllmKernelOrm') as mock_vllm_kernel_orm:

            db = VllmDatabase()

            criteria = SearchCriteria(
                cache_dir="/test/cache",
                name="test_kernel",
                backend="cuda",
                arch="80"
            )

            results = db.search(criteria)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["hash"], "test_hash")
            mock_session.query.assert_called_with(mock_vllm_kernel_orm)
            mock_session.close.assert_called_once()

    @patch('model_cache_manager.data.database.create_engine_and_session')
    def test_search_with_time_filters(self, mock_create_engine_session):
        """Test search method with time-based filters."""
        mock_engine = MagicMock()
        mock_session_local = MagicMock()
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_create_engine_session.return_value = (mock_engine, mock_session_local)

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []
        mock_session.query.return_value = mock_query

        with patch('model_cache_manager.data.database.Base'), \
             patch('model_cache_manager.data.database.VllmKernelOrm') as mock_vllm_kernel_orm, \
             patch('model_cache_manager.data.database.and_'):

            # Mock the column attributes to avoid comparison issues
            mock_vllm_kernel_orm.modified_time = MagicMock()
            mock_vllm_kernel_orm.runtime_hits = MagicMock()

            db = VllmDatabase()

            criteria = SearchCriteria(
                older_than_timestamp=1000000.0,
                younger_than_timestamp=2000000.0,
                cache_hit_lower=5,
                cache_hit_higher=100
            )

            results = db.search(criteria)

            # Verify basic operations occurred
            self.assertIsInstance(results, list)
            mock_session.close.assert_called_once()

    @patch('model_cache_manager.data.database.create_engine_and_session')
    def test_close(self, mock_create_engine_session):
        """Test database close method."""
        mock_engine = MagicMock()
        mock_session_local = MagicMock()
        mock_create_engine_session.return_value = (mock_engine, mock_session_local)

        with patch('model_cache_manager.data.database.Base'):
            db = VllmDatabase()
            db.close()

            mock_engine.dispose.assert_called_once()


if __name__ == "__main__":
    unittest.main()
