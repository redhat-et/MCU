"""
Unit tests for RuntimeStatsCollector.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from model_cache_manager.runtime.tracker import RuntimeStatsCollector


def test_single_hit_flushes_and_clears():
    """Test single cache hit"""
    collector = RuntimeStatsCollector()

    # deterministic timestamp
    with patch("model_cache_manager.runtime.tracker.time.time", return_value=42.0):
        collector.record_access("abcd1234", hit=True)

    collector.flush()

    # internal buffer must be empty after flush
    # pylint: disable=protected-access
    assert not collector._pending_records


def test_updates_kernel(fake_kernel, mock_session):
    """Test kernel update"""
    collector = RuntimeStatsCollector()
    collector.record_access("abcd1234", hit=True)
    collector.record_access("abcd1234", hit=False)
    collector.flush()

    # the fake KernelOrm instance should now show 1 hit
    assert fake_kernel.runtime_hits == 1
    assert fake_kernel.last_access_time > 0
    assert mock_session.commit.call_count == 2


def create_query_side_effect(mock_kernel1, mock_kernel2):
    """Helper function for query side effect"""

    # pylint: disable=unused-argument
    def query_side_effect(model):
        query_mock = MagicMock()
        filter_mock = MagicMock()
        query_mock.filter.return_value = filter_mock

        def filter_side_effect(*args):
            # Extract the cache_key from the filter arguments
            for arg in args:
                if hasattr(arg, "left") and hasattr(arg.left, "key"):
                    if arg.left.key == "hash" and hasattr(arg, "right"):
                        if hasattr(arg.right, "value"):
                            cache_key = arg.right.value
                        else:
                            cache_key = arg.right
                        if cache_key == "key1":
                            filter_mock.all.return_value = [mock_kernel1]
                        elif cache_key == "key2":
                            filter_mock.all.return_value = [mock_kernel2]
                        break
            return filter_mock

        query_mock.filter = filter_side_effect
        return query_mock

    return query_side_effect


# pylint: disable=protected-access, too-many-statements
def test_data_persistence_with_multiple_records():
    """
    Tests that the collector correctly aggregates multiple records
    and that the flush method attempts to write the correct,
    aggregated hit to the database
    """

    mock_db = MagicMock()
    mock_session = MagicMock()
    mock_db.get_session.return_value = mock_session

    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock(return_value=None)

    mock_kernel1 = MagicMock()
    mock_kernel1.runtime_hits = 0
    mock_kernel1.last_access_time = None

    mock_kernel2 = MagicMock()
    mock_kernel2.runtime_hits = 5
    mock_kernel2.last_access_time = None

    mock_session.query.side_effect = create_query_side_effect(
        mock_kernel1, mock_kernel2
    )

    # Patch the Database class to return our mock
    with patch("model_cache_manager.runtime.tracker.Database", return_value=mock_db):
        collector = RuntimeStatsCollector()

        # Temporarily mock _persist_to_database to prevent auto-persistence
        original_persist = collector._persist_to_database
        collector._persist_to_database = Mock()

        test_cache_dir = Path("/test/cache")

        collector.record_access(cache_key="key1", hit=True, cache_dir=test_cache_dir)
        collector.record_access(cache_key="key1", hit=True, cache_dir=test_cache_dir)
        collector.record_access(cache_key="key1", hit=False, cache_dir=test_cache_dir)

        collector.record_access(cache_key="key2", hit=False, cache_dir=test_cache_dir)
        collector.record_access(cache_key="key2", hit=False, cache_dir=test_cache_dir)
        collector.record_access(cache_key="key2", hit=True, cache_dir=test_cache_dir)

        # Verify pending records before flush
        assert len(collector._pending_records) == 6

        collector._persist_to_database = original_persist
        collector.flush()

        assert mock_db.get_session.called

        # For key1: 0 + 2 hits
        assert mock_kernel1.runtime_hits == 2
        assert mock_kernel1.last_access_time > 0

        # For key2: 5 + 1 hit, 2
        assert mock_kernel2.runtime_hits == 6
        assert mock_kernel2.last_access_time > 0

        mock_session.commit.assert_called_once()

        assert len(collector._pending_records) == 0
