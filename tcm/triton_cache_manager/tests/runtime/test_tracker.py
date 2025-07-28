"""
End-to-end test for TCMTrackingCacheManager with a stubbed
FileCacheManager so the real cache is not touched
and Triton is not used directly
"""

import pytest
from triton_cache_manager.runtime import tracker as tr


@pytest.fixture()
def stub_file_cache_manager(monkeypatch):
    """
    Dummy FileCacheManager
    """
    calls = {"n": 0}

    class Dummy:
        """A dummy FileCacheManager"""
        def __init__(self, *_, **__):
            self.cache_dir = "/tmp/tcm"

        def get_group(self, *_):
            """Returns a dummy item on the first call and None thereafter."""
            calls["n"] += 1
            return {"test": "tracker"} if calls["n"] == 1 else None

        # unused in this test but must exist
        def get_file(self, *_):
            """From FileCacheManager"""
            return None

        def put(self, *_):
            """From FileCacheManager"""
            return ""
        # pylint: disable=unnecessary-pass
        def put_group(self, *_):
            """From FileCacheManager"""
            pass

    monkeypatch.setattr(tr, "FileCacheManager", Dummy)
    return Dummy

# pylint: disable=unused-argument, redefined-outer-name
def test_hit_is_recorded(monkeypatch, stub_file_cache_manager, fake_kernel):
    """Test cache hit"""
    # fresh collector instance, inserted into module under test
    fresh_collector = tr.RuntimeStatsCollector()
    monkeypatch.setattr(tr, "_runtime_collector", fresh_collector, raising=True)

    mgr = tr.TCMTrackingCacheManager("abcd1234")

    assert mgr.get_group("entry") == {"test": "tracker"}
    assert mgr.get_group("entry") is None

    fresh_collector.flush()

    # kernel stats should show 1 hit
    assert fake_kernel.runtime_hits == 1
