"""
Shared test utilities for database tests.
"""

import tempfile
import shutil
from unittest.mock import MagicMock, Mock
from pathlib import Path
from model_cache_manager.models.kernel import Kernel, KernelFile


def setup_kernel_orm_mock(mock_kernel_orm):
    """
    Set up common mock configuration for KernelOrm/VllmKernelOrm.

    Args:
        mock_kernel_orm: The mock ORM class to configure
    """
    mock_kernel_orm.get_common_kernel_values = MagicMock(return_value={})

    mock_columns = [MagicMock(name="col1"), MagicMock(name="col2")]
    mock_columns[0].name = "col1"
    mock_columns[1].name = "col2"
    mock_kernel_orm.__table__ = MagicMock()
    mock_kernel_orm.__table__.columns = mock_columns


def setup_sqlite_insert_mock(mock_sqlite_insert):
    """
    Set up mock for sqlite_insert chain.

    Args:
        mock_sqlite_insert: The mock sqlite_insert to configure

    Returns:
        The configured mock statement
    """
    mock_stmt = MagicMock()
    mock_sqlite_insert.return_value = MagicMock()
    mock_sqlite_insert.return_value.values.return_value = mock_stmt
    mock_stmt.on_conflict_do_update.return_value = mock_stmt
    return mock_stmt


def setup_query_mock(mock_session):
    """
    Set up mock for session query operations.

    Args:
        mock_session: The mock session to configure

    Returns:
        tuple: (mock_query, mock_filter)
    """
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_filter
    mock_filter.delete.return_value = None
    return mock_query, mock_filter


def setup_tuple_mock(mock_tuple):
    """
    Set up mock for tuple_ function used in composite key filtering.

    Args:
        mock_tuple: The mock tuple_ to configure
    """
    mock_tuple.return_value = MagicMock()
    mock_tuple.return_value.in_ = MagicMock(return_value=True)


def setup_engine_and_session_mock(mock_create_engine_session):
    """
    Set up mock for create_engine_and_session.

    Args:
        mock_create_engine_session: The mock create_engine_and_session to configure

    Returns:
        tuple: (mock_engine, mock_session_local, mock_session)
    """
    mock_engine = MagicMock()
    mock_session_local = MagicMock()
    mock_session = MagicMock()
    mock_session_local.return_value = mock_session
    mock_create_engine_session.return_value = (mock_engine, mock_session_local)
    return mock_engine, mock_session_local, mock_session


def create_mock_kernel_file(
    file_type: str = "ptx", file_path: str = "/test/kernel.ptx", size: int = 1024
) -> Mock:
    """Create a mock KernelFile object.

    Args:
        file_type: Type of the kernel file (ptx, ttir, json, etc.)
        file_path: Path to the file
        size: Size of the file in bytes

    Returns:
        Mock KernelFile object
    """
    mock_file = Mock(spec=KernelFile)
    mock_file.file_type = file_type
    mock_file.path = Path(file_path)
    mock_file.size = size
    return mock_file


# pylint: disable=too-many-arguments,too-many-positional-arguments
def create_mock_kernel(
    hash_val: str = "test_hash",
    name: str = "test_kernel",
    backend: str = "cuda",
    arch: str = "80",
    triton_version: str = "3.3.0",
    metadata: dict = None,
    modified_time: float = 1234567890.0,
    cache_dir: str = "/test/cache",
    files: list = None,
    **kwargs,
) -> Mock:
    """Create a comprehensive mock Kernel object.

    Args:
        hash_val: Kernel hash
        name: Kernel name
        backend: Backend (cuda, etc.)
        arch: Architecture (80, etc.)
        triton_version: Triton version
        metadata: Kernel metadata dictionary
        modified_time: Modification timestamp
        cache_dir: Cache directory path
        files: List of KernelFile objects (will create defaults if None)
        **kwargs: Additional attributes to set on the kernel

    Returns:
        Mock Kernel object with all attributes set
    """
    if metadata is None:
        metadata = {"key": "value"}

    if files is None:
        files = [
            create_mock_kernel_file("ptx", f"{cache_dir}/kernel.ptx", 1024),
            create_mock_kernel_file("ttir", f"{cache_dir}/kernel.ttir", 512),
        ]

    mock_kernel = Mock(spec=Kernel)
    mock_kernel.hash = hash_val
    mock_kernel.name = name
    mock_kernel.backend = backend
    mock_kernel.arch = arch
    mock_kernel.triton_version = triton_version
    mock_kernel.metadata = metadata
    mock_kernel.modified_time = modified_time
    mock_kernel.cache_dir = cache_dir
    mock_kernel.files = files

    for attr, value in kwargs.items():
        setattr(mock_kernel, attr, value)

    return mock_kernel


# pylint: disable=too-many-arguments,too-many-positional-arguments
def create_mock_kernel_data(
    hash_val: str,
    name: str,
    backend: str = "cuda",
    arch: str = "80",
    mod_time_offset_secs: int = 0,
    total_size_bytes: int = 1024,
    runtime_hits: int = 0,
    **kwargs,
) -> dict:
    """Helper to create consistent mock kernel data dictionary.

    Args:
        hash_val: Kernel hash
        name: Kernel name
        backend: Backend
        arch: Architecture
        mod_time_offset_secs: Offset from base timestamp
        total_size_bytes: Total size in bytes
        runtime_hits: Number of runtime hits
        **kwargs: Additional fields to include

    Returns:
        Dictionary with kernel data
    """
    base_timestamp = 1747681046.0
    data = {
        "hash": hash_val,
        "name": name,
        "backend": backend,
        "arch": arch,
        "modified_time": base_timestamp + mod_time_offset_secs,
        "total_size": total_size_bytes,
        "runtime_hits": runtime_hits,
    }
    data.update(kwargs)
    return data


def setup_test_class_fixtures(test_case):
    """
    Set up common test class fixtures (setUp).

    Args:
        test_case: The unittest TestCase instance
    """
    test_case.temp_dir = Path(tempfile.mkdtemp())
    test_case.cache_dir = test_case.temp_dir / "vllm_cache"
    test_case.cache_dir.mkdir(parents=True, exist_ok=True)


def teardown_test_fixtures(test_case):
    """
    Clean up test fixtures (tearDown).

    Args:
        test_case: The unittest TestCase instance
    """
    shutil.rmtree(test_case.temp_dir, ignore_errors=True)
