"""
Shared test utilities for database tests.
"""

from unittest.mock import MagicMock


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
