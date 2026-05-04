# pylint: disable=duplicate-code
"""
Unit tests for the VllmDatabase and VllmKernelOrm.
"""

import unittest
from unittest.mock import MagicMock, patch

from model_cache_manager.data.database import VllmDatabase
from model_cache_manager.data.db_models import VllmKernelOrm, BaseKernelMixin
from model_cache_manager.models.kernel import VllmKernelMetadata
from model_cache_manager.models.criteria import SearchCriteria
from model_cache_manager.tests.test_utils import (
    create_mock_kernel,
    setup_kernel_orm_mock,
    setup_sqlite_insert_mock,
    setup_query_mock,
    setup_tuple_mock,
    setup_engine_and_session_mock,
)


class TestBaseKernelMixin(unittest.TestCase):
    """Test suite for the BaseKernelMixin."""

    def test_get_common_kernel_values(self):
        """Test get_common_kernel_values method."""
        mock_kernel = create_mock_kernel()

        values = BaseKernelMixin.get_common_kernel_values(mock_kernel)

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
        cache_dir = "/test/vllm/cache"
        vllm_hash = "test_vllm_hash"

        with patch(
            "model_cache_manager.data.db_models.sqlite_insert"
        ) as mock_insert, patch("model_cache_manager.data.db_models.VllmKernelFileOrm"):

            mock_stmt = MagicMock()
            mock_insert.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            rank_x_y = "rank_0_0"
            artifact_compile_range = "artifact_compile_range_0"
            vllm_meta = VllmKernelMetadata(
                vllm_hash=vllm_hash,
                rank_x_y=rank_x_y,
                artifact_compile_range=artifact_compile_range
            )
            VllmKernelOrm.upsert_from_dto(
                self.mock_session, self.mock_kernel, cache_dir, vllm_meta
            )

            # Verify sqlite_insert was called
            mock_insert.assert_called_once_with(VllmKernelOrm)

            # Verify kernel values were set correctly
            if hasattr(mock_stmt.values, "call_args") and mock_stmt.values.call_args:
                kernel_values = mock_stmt.values.call_args[0][0]
                self.assertEqual(kernel_values["cache_dir"], cache_dir)
                self.assertEqual(kernel_values["vllm_hash"], vllm_hash)
                self.assertEqual(kernel_values["triton_cache_key"], "test_hash")

            # Verify session operations
            self.mock_session.execute.assert_called_once()
            # Verify query was called
            self.mock_session.query.assert_called()  # pylint: disable=no-member

    def test_to_dict(self):
        """Test to_dict method inherited from BaseKernelMixin."""
        # Create a mock VllmKernelOrm instance
        vllm_kernel = VllmKernelOrm()
        vllm_kernel.cache_dir = "/test/cache"
        vllm_kernel.vllm_hash = "test_hash"
        vllm_kernel.triton_cache_key = "triton_key"
        vllm_kernel.kernel_metadata_json = {"test": "metadata"}

        # Mock the __table__.columns attribute
        mock_column1 = MagicMock()
        mock_column1.key = "cache_dir"
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

    @patch("model_cache_manager.data.database.create_engine_and_session")
    def test_init(self, mock_create_engine_session):
        """Test VllmDatabase initialization."""
        mock_engine, mock_session_local, _ = setup_engine_and_session_mock(
            mock_create_engine_session
        )

        with patch("model_cache_manager.data.database_utils.ensure_schema") as mock_ensure_schema:
            db = VllmDatabase()

            mock_create_engine_session.assert_called_once_with("vllm")
            self.assertEqual(db.engine, mock_engine)
            self.assertEqual(db.SessionLocal, mock_session_local)
            mock_ensure_schema.assert_called_once_with(mock_engine)

    @patch("model_cache_manager.data.database.create_engine_and_session")
    def test_insert_kernel_success(self, mock_create_engine_session):
        """Test successful kernel insertion."""
        _, _, mock_session = setup_engine_and_session_mock(mock_create_engine_session)

        with patch("model_cache_manager.data.database_utils.ensure_schema"), patch(
            "model_cache_manager.data.database.VllmKernelOrm"
        ) as mock_kernel_orm:

            db = VllmDatabase()
            cache_dir = "/test/cache"
            vllm_hash = "test_hash"
            rank_x_y = "rank_0_0"
            artifact_compile_range = "test_shape"

            vllm_meta = VllmKernelMetadata(
                vllm_hash=vllm_hash,
                rank_x_y=rank_x_y,
                artifact_compile_range=artifact_compile_range
            )
            db.insert_kernel(self.mock_kernel, cache_dir, vllm_meta)

            # Verify that upsert_from_dto was called with the correct arguments
            mock_kernel_orm.upsert_from_dto.assert_called_once()
            args = mock_kernel_orm.upsert_from_dto.call_args[0]
            self.assertEqual(args[0], mock_session)
            self.assertEqual(args[1], self.mock_kernel)
            self.assertEqual(args[2], cache_dir)
            self.assertEqual(args[3].vllm_hash, vllm_hash)
            self.assertEqual(args[3].rank_x_y, rank_x_y)
            self.assertEqual(args[3].artifact_compile_range, artifact_compile_range)
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()

    @patch("model_cache_manager.data.database.create_engine_and_session")
    def test_bulk_insert_kernels_success(self, mock_create_engine_session):
        """Test successful bulk kernel insertion."""
        _, _, mock_session = setup_engine_and_session_mock(mock_create_engine_session)

        # Create a minimal mock ORM class that SQLAlchemy can work with
        class MockVllmKernelFileOrm:  # pylint: disable=too-few-public-methods
            """Mock ORM class for VllmKernelFileOrm."""

            cache_dir = MagicMock()
            vllm_hash = MagicMock()
            triton_cache_key = MagicMock()
            rank_x_y = MagicMock()

        with patch("model_cache_manager.data.database_utils.ensure_schema"), patch(
            "model_cache_manager.data.database.VllmKernelOrm"
        ) as mock_kernel_orm, patch(
            "model_cache_manager.data.database.VllmKernelFileOrm", MockVllmKernelFileOrm
        ), patch(
            "model_cache_manager.data.database.sqlite_insert"
        ) as mock_sqlite_insert, patch(
            "model_cache_manager.data.database.tuple_"
        ) as mock_tuple:

            db = VllmDatabase()

            # Use shared setup utilities
            setup_kernel_orm_mock(mock_kernel_orm)
            setup_sqlite_insert_mock(mock_sqlite_insert)
            setup_query_mock(mock_session)
            setup_tuple_mock(mock_tuple)

            # Create test data
            mock_kernel2 = MagicMock()
            mock_kernel2.hash = "hash2"
            mock_kernel2.files = []

            kernels_data = [
                (self.mock_kernel, "/test/cache", "vllm_hash1", "rank_0_0"),
                (mock_kernel2, "/test/cache", "vllm_hash2", "rank_1_0"),
            ]

            result = db.bulk_insert_kernels(kernels_data, batch_size=2)

            # Verify the correct number of kernels were inserted
            self.assertEqual(result, 2)

            # Verify bulk operations were used
            mock_session.bulk_insert_mappings.assert_called_once_with(
                MockVllmKernelFileOrm,
                [
                    {
                        "cache_dir": "/test/cache",
                        "vllm_hash": "vllm_hash1",
                        "triton_cache_key": "test_hash",
                        "rank_x_y": "rank_0_0",
                        "type": "ptx",
                        "rel_path": "kernel.ptx",
                        "size": 1024,
                    },
                    {
                        "cache_dir": "/test/cache",
                        "vllm_hash": "vllm_hash1",
                        "triton_cache_key": "test_hash",
                        "rank_x_y": "rank_0_0",
                        "type": "ttir",
                        "rel_path": "kernel.ttir",
                        "size": 512,
                    },
                ],
            )
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()

    @patch("model_cache_manager.data.database.create_engine_and_session")
    def test_search_with_criteria(self, mock_create_engine_session):
        """Test search method with various criteria."""
        _, _, mock_session = setup_engine_and_session_mock(mock_create_engine_session)

        # Mock query results
        mock_kernel_orm = MagicMock()
        mock_kernel_orm.to_dict.return_value = {
            "hash": "test_hash",
            "name": "test_kernel",
        }
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = [mock_kernel_orm]
        mock_session.query.return_value = mock_query

        with patch("model_cache_manager.data.database_utils.ensure_schema"), patch(
            "model_cache_manager.data.database.VllmKernelOrm"
        ):

            db = VllmDatabase()

            criteria = SearchCriteria(
                cache_dir="/test/cache", name="test_kernel", backend="cuda", arch="80"
            )

            results = db.search(criteria)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["hash"], "test_hash")
            # Verify query was called
            mock_session.query.assert_called()  # pylint: disable=no-member
            mock_session.close.assert_called_once()

    @patch("model_cache_manager.data.database.create_engine_and_session")
    def test_search_with_time_filters(self, mock_create_engine_session):
        """Test search method with time-based filters."""
        _, _, mock_session = setup_engine_and_session_mock(mock_create_engine_session)

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []
        mock_session.query.return_value = mock_query

        with patch("model_cache_manager.data.database_utils.ensure_schema"), patch(
            "model_cache_manager.data.database.VllmKernelOrm"
        ) as mock_vllm_kernel_orm, patch("model_cache_manager.data.database.and_"):

            # Mock the column attributes to avoid comparison issues
            mock_vllm_kernel_orm.modified_time = MagicMock()
            mock_vllm_kernel_orm.runtime_hits = MagicMock()

            db = VllmDatabase()

            criteria = SearchCriteria(
                older_than_timestamp=1000000.0,
                younger_than_timestamp=2000000.0,
                cache_hit_lower=5,
                cache_hit_higher=100,
            )

            results = db.search(criteria)

            # Verify basic operations occurred
            self.assertIsInstance(results, list)
            mock_session.close.assert_called_once()

    @patch("model_cache_manager.data.database.create_engine_and_session")
    def test_close(self, mock_create_engine_session):
        """Test database close method."""
        mock_engine, _, _ = setup_engine_and_session_mock(mock_create_engine_session)

        with patch("model_cache_manager.data.database_utils.ensure_schema"):
            db = VllmDatabase()
            db.close()

            mock_engine.dispose.assert_called_once()


if __name__ == "__main__":
    unittest.main()
