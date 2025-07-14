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
    Dummy FileCacheManager that
    """
    calls = {"n": 0}

    class Dummy:
        def __init__(self, *_, **__):
            self.cache_dir = "/tmp/tcm"

        def get_group(self, *_):
            calls["n"] += 1
            return {"test": "tracker"} if calls["n"] == 1 else None

        # unused in this test but must exist
        def get_file(self, *_):
            return None

        def put(self, *_):
            return ""

        def put_group(self, *_):
            pass

    monkeypatch.setattr(tr, "FileCacheManager", Dummy)
    return Dummy


def test_hit_and_miss_are_recorded(monkeypatch, stub_file_cache_manager, fake_kernel):
    """Test cache hit/miss"""
    # fresh collector instance, inserted into module under test
    fresh_collector = tr.RuntimeStatsCollector()
    monkeypatch.setattr(tr, "_runtime_collector", fresh_collector, raising=True)

    mgr = tr.TCMTrackingCacheManager("abcd1234")

    assert mgr.get_group("entry") == {"test": "tracker"}
    assert mgr.get_group("entry") is None

    fresh_collector.flush()

    # kernel stats should show 1 hit & 1 miss
    assert fake_kernel.runtime_hits == 1
    assert fake_kernel.runtime_misses == 1
