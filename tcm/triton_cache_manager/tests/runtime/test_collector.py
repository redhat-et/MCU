"""
Unit tests for RuntimeStatsCollector.
"""

from unittest.mock import patch
from triton_cache_manager.runtime.tracker import RuntimeStatsCollector


def test_single_hit_flushes_and_clears():
    """Test single cache hit"""
    collector = RuntimeStatsCollector()

    # deterministic timestamp
    with patch("triton_cache_manager.runtime.tracker.time.time", return_value=42.0):
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

    # the fake KernelOrm instance should now show 1 hit / 1 miss
    assert fake_kernel.runtime_hits == 1
    assert fake_kernel.runtime_misses == 1
    assert fake_kernel.last_access_time > 0
    assert mock_session.commit.call_count == 2
