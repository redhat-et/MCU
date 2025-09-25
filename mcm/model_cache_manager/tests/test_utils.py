"""
Shared test utilities for database tests.
"""

from unittest.mock import MagicMock, Mock, patch
from pathlib import Path
from model_cache_manager.models.kernel import Kernel, KernelFile


def setup_kernel_orm_mock(mock_kernel_orm):
    """
    Set up common mock configuration for KernelOrm/VllmKernelOrm.

    Args:
        mock_kernel_orm: The mock ORM class to configure
    """
    # Mock the get_common_kernel_values method
    mock_kernel_orm.get_common_kernel_values = MagicMock(return_value={})

    # Mock table columns
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
    file_type: str = "ptx",
    file_path: str = "/test/kernel.ptx",
    size: int = 1024
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
    **kwargs
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
    
    # Set any additional attributes
    for attr, value in kwargs.items():
        setattr(mock_kernel, attr, value)
        
    return mock_kernel


def create_mock_kernel_data(
    hash_val: str,
    name: str,
    backend: str = "cuda",
    arch: str = "80",
    mod_time_offset_secs: int = 0,
    total_size_bytes: int = 1024,
    runtime_hits: int = 0,
    **kwargs
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


def setup_service_mocks(service_type: str, mode: str = "vllm"):
    """
    Create standard mock setup for service tests.

    Args:
        service_type: Type of service ("index", "search", "prune")
        mode: Mode of operation ("vllm", "vllm_legacy", "triton")

    Returns:
        tuple: (mock_repo, mock_db, mock_repo_instance, mock_db_instance)
    """
    mock_repo = MagicMock()
    mock_db = MagicMock()
    mock_repo_instance = MagicMock()
    mock_db_instance = MagicMock()

    mock_repo.return_value = mock_repo_instance
    mock_db.return_value = mock_db_instance

    # Set up common database methods
    mock_db_instance.search.return_value = []
    mock_db_instance.get_session.return_value.__enter__.return_value = MagicMock()
    mock_db_instance.bulk_insert_kernels.return_value = 0
    mock_db_instance.estimate_space.return_value = 0

    # Set up common repository methods
    mock_repo_instance.kernels.return_value = []
    mock_repo_instance.cache_dir = Path("/test/cache")
    mock_repo_instance.root = Path("/test/cache")

    return mock_repo, mock_db, mock_repo_instance, mock_db_instance


def assert_service_init(test_case, service, expected_mode, expected_cache_dir,
                        mock_repo, mock_db, repo_call_args=None, db_call_args=None):
    """
    Common assertions for service initialization tests.

    Args:
        test_case: The unittest TestCase instance
        service: The service instance
        expected_mode: Expected mode value
        expected_cache_dir: Expected cache directory
        mock_repo: Mock repository class
        mock_db: Mock database class
        repo_call_args: Expected repository call arguments
        db_call_args: Expected database call arguments
    """
    test_case.assertEqual(service.mode, expected_mode)

    if hasattr(service, 'cache_dir'):
        test_case.assertEqual(service.cache_dir, expected_cache_dir)

    if repo_call_args is not None:
        mock_repo.assert_called_once_with(*repo_call_args)
    elif mock_repo:
        if expected_cache_dir:
            mock_repo.assert_called_once_with(expected_cache_dir)
        else:
            mock_repo.assert_called_once()

    if db_call_args is not None:
        mock_db.assert_called_once_with(*db_call_args)
    else:
        mock_db.assert_called_once()


def create_test_search_results(mode: str = "vllm", count: int = 2):
    """
    Create mock search results for different modes.

    Args:
        mode: Mode of operation ("vllm", "triton")
        count: Number of results to create

    Returns:
        List of search result dictionaries
    """
    results = []
    for i in range(1, count + 1):
        if mode == "vllm":
            results.append({
                "vllm_hash": f"vllm_hash{i}",
                "triton_cache_key": f"hash{i}",
                "name": f"kernel{i}",
                "rank_x_y": "rank_0_0",
            })
        else:
            results.append({
                "hash": f"hash{i}",
                "name": f"kernel{i}",
            })
    return results


def patch_service_dependencies(mode: str):
    """
    Create decorators for patching service dependencies based on mode.

    Args:
        mode: Mode of operation ("vllm", "vllm_legacy", "triton")

    Returns:
        List of patch decorators to apply
    """
    if mode == "vllm":
        return [
            patch("model_cache_manager.strategies.vllm_strategy.VllmCacheRepository"),
            patch("model_cache_manager.strategies.vllm_strategy.VllmDatabase"),
        ]
    elif mode == "vllm_legacy":
        return [
            patch("model_cache_manager.strategies.vllm_legacy_strategy.VllmLegacyCacheRepository"),
            patch("model_cache_manager.strategies.vllm_legacy_strategy.VllmLegacyDatabase"),
        ]
    else:  # triton
        return [
            patch("model_cache_manager.strategies.triton_strategy.CacheRepository"),
            patch("model_cache_manager.strategies.triton_strategy.Database"),
        ]


def setup_test_class_fixtures(test_case):
    """
    Set up common test class fixtures (setUp and tearDown).

    Args:
        test_case: The unittest TestCase instance
    """
    import tempfile
    import shutil
    from pathlib import Path

    test_case.temp_dir = Path(tempfile.mkdtemp())
    test_case.cache_dir = test_case.temp_dir / "vllm_cache"
    test_case.cache_dir.mkdir(parents=True, exist_ok=True)


def teardown_test_fixtures(test_case):
    """
    Clean up test fixtures.

    Args:
        test_case: The unittest TestCase instance
    """
    import shutil
    shutil.rmtree(test_case.temp_dir, ignore_errors=True)


def setup_reindex_test(mode: str, mock_repo, mock_db, cache_dir=None):
    """
    Set up mocks for reindex tests.

    Args:
        mode: Mode of operation ("vllm", "vllm_legacy", "triton")
        mock_repo: Mock repository class
        mock_db: Mock database class
        cache_dir: Cache directory path

    Returns:
        tuple: (mock_repo_instance, mock_db_instance, mock_kernels)
    """
    mock_kernel1 = create_mock_kernel("hash1", "kernel1")
    mock_kernel2 = create_mock_kernel("hash2", "kernel2")

    mock_repo_instance = MagicMock()
    mock_db_instance = MagicMock()

    if mode == "vllm":
        # New vLLM format with artifact_shape and best_config
        mock_repo_instance.kernels.return_value = [
            ("vllm_hash1", "/cache/root", "rank_0_0", "artifact_shape_0", '{"config": "test"}', mock_kernel1),
            ("vllm_hash2", "/cache/root", "rank_1_0", "artifact_shape_1", None, mock_kernel2),
        ]
    elif mode == "vllm_legacy":
        # Legacy vLLM format without artifact_shape and best_config
        mock_repo_instance.kernels.return_value = [
            ("vllm_hash1", "/cache/root", "rank_0_0", mock_kernel1),
            ("vllm_hash2", "/cache/root", "rank_1_0", mock_kernel2),
        ]
    else:  # triton
        mock_repo_instance.kernels.return_value = [mock_kernel1, mock_kernel2]
        if cache_dir:
            mock_repo_instance.cache_dir = cache_dir
            mock_repo_instance.root = cache_dir

    mock_repo.return_value = mock_repo_instance

    mock_db_instance.search.return_value = []
    mock_db_instance.bulk_insert_kernels.return_value = 2
    mock_db.return_value = mock_db_instance

    return mock_repo_instance, mock_db_instance, [mock_kernel1, mock_kernel2]


def assert_reindex_results(test_case, service, mock_db_instance, mode: str,
                          expected_updated: int = 2, cache_dir=None):
    """
    Common assertions for reindex test results.

    Args:
        test_case: The unittest TestCase instance
        service: The service instance
        mock_db_instance: Mock database instance
        mode: Mode of operation
        expected_updated: Expected number of updated records
        cache_dir: Cache directory (for triton mode)
    """
    updated, current = service.reindex()

    test_case.assertEqual(updated, expected_updated)
    test_case.assertEqual(current, 0)

    # Verify bulk_insert_kernels was called with correct parameters
    if mode == "vllm":
        # New vLLM with artifact_shape and best_config
        mock_db_instance.bulk_insert_kernels.assert_called_once()
        call_args = mock_db_instance.bulk_insert_kernels.call_args[0][0]
        test_case.assertEqual(len(call_args), 2)
        # Check structure of new vLLM parameters
        test_case.assertEqual(len(call_args[0]), 6)  # (kernel, path, hash, rank, artifact_shape, best_config)
        test_case.assertEqual(len(call_args[1]), 6)
    elif mode == "vllm_legacy":
        # Legacy vLLM without artifact_shape and best_config
        mock_db_instance.bulk_insert_kernels.assert_called_once()
        call_args = mock_db_instance.bulk_insert_kernels.call_args[0][0]
        test_case.assertEqual(len(call_args), 2)
        # Check structure of legacy vLLM parameters
        test_case.assertEqual(len(call_args[0]), 4)  # (kernel, path, hash, rank)
        test_case.assertEqual(len(call_args[1]), 4)
    else:  # triton
        mock_db_instance.bulk_insert_kernels.assert_called_once()
        if cache_dir:
            # For triton mode with cache_dir
            call_args = mock_db_instance.bulk_insert_kernels.call_args[0][0]
            test_case.assertEqual(len(call_args), 2)
            test_case.assertEqual(len(call_args[0]), 2)  # (kernel, cache_dir)


def setup_search_test(mock_db):
    """
    Set up mocks for search tests.

    Args:
        mock_db: Mock database class

    Returns:
        tuple: (mock_db_instance, mock_results)
    """
    mock_results = [
        {"hash": "hash1", "name": "kernel1"},
        {"hash": "hash2", "name": "kernel2"},
    ]

    mock_db_instance = MagicMock()
    mock_db_instance.search.return_value = mock_results
    mock_db.return_value = mock_db_instance

    return mock_db_instance, mock_results


def setup_prune_test(mock_repo, mock_db, search_results=None):
    """
    Set up mocks for prune tests.

    Args:
        mock_repo: Mock repository class
        mock_db: Mock database class
        search_results: Optional search results to use

    Returns:
        tuple: (mock_repo_instance, mock_db_instance)
    """
    if search_results is None:
        search_results = create_test_search_results("vllm", 2)

    mock_repo_instance = MagicMock()
    mock_repo.return_value = mock_repo_instance

    mock_db_instance = MagicMock()
    mock_db_instance.search.return_value = search_results
    mock_db_instance.estimate_space.return_value = 2048
    mock_db_instance.get_session.return_value.__enter__.return_value = MagicMock()
    mock_db.return_value = mock_db_instance

    return mock_repo_instance, mock_db_instance


def assert_prune_results(test_case, service, criteria, mock_db_instance,
                        expected_pruned: int = 2, auto_confirm: bool = True):
    """
    Common assertions for prune test results.

    Args:
        test_case: The unittest TestCase instance
        service: The service instance
        criteria: Search criteria
        mock_db_instance: Mock database instance
        expected_pruned: Expected number of pruned items
        auto_confirm: Whether to auto-confirm pruning
    """
    with patch.object(service, "_delete_kernel_unified", return_value=1024), \
         patch.object(service, "_confirm", return_value=True):

        result = service.prune(criteria, auto_confirm=auto_confirm)

        test_case.assertIsNotNone(result)
        test_case.assertEqual(result.pruned, expected_pruned)
        mock_db_instance.search.assert_called_once()


def setup_init_test_mocks(mock_repo, mock_db):
    """
    Set up standard mocks for initialization tests.

    Args:
        mock_repo: Mock repository class
        mock_db: Mock database class

    Returns:
        tuple: (mock_repo_instance, mock_db_instance)
    """
    mock_repo_instance = MagicMock()
    mock_db_instance = MagicMock()
    mock_repo.return_value = mock_repo_instance
    mock_db.return_value = mock_db_instance
    return mock_repo_instance, mock_db_instance


def assert_vllm_service_init(test_case, service, cache_dir, mock_repo, mock_db, expected_mode=None):
    """
    Common assertions for vLLM service initialization.

    Args:
        test_case: The unittest TestCase instance
        service: The service instance
        cache_dir: Expected cache directory
        mock_repo: Mock repository class
        mock_db: Mock database class
        expected_mode: Expected mode (defaults to service.mode)
    """
    if expected_mode:
        test_case.assertEqual(service.mode, expected_mode)
    test_case.assertEqual(service.cache_dir, cache_dir)
    mock_repo.assert_called_once_with(cache_dir)
    mock_db.assert_called_once()
