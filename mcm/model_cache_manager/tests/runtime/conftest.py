"""
Shared fixtures that stub out the DB layer used by
model_cache_manager.runtime.tracker
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from model_cache_manager.runtime import tracker


# pylint: disable=redefined-outer-name
@pytest.fixture()
def fake_kernel():
    """Small stand-in for KernelOrm rows"""
    return SimpleNamespace(
        hash="abcd1234",
        cache_dir="/tmp/mcm",
        runtime_hits=0,
        runtime_misses=0,
        last_access_time=0.0,
    )


# pylint: disable=no-member
@pytest.fixture()
def mock_session(fake_kernel):
    """Session with just enough behaviour for the tracker"""
    q = MagicMock()
    q.filter.return_value.all.return_value = [fake_kernel]

    sess = MagicMock()
    sess.query.return_value = q
    sess.commit.return_value = None
    sess.rollback.return_value = None
    sess.close.return_value = None
    return sess


# pylint: disable=redefined-outer-name
@pytest.fixture(autouse=True)
def patch_database(monkeypatch, mock_session):
    """
    Replace tracker.Database() with a lambda that gives a fake
    containing get_session() --> mock_session
    """

    fake_db = MagicMock()
    fake_db.get_session.return_value = mock_session
    fake_db.close.return_value = None

    monkeypatch.setattr(tracker, "Database", lambda: fake_db)
    yield
