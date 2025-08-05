# pylint: disable=c-extension-no-member # Handles potential Pylint issues with SQLAlchemy C extensions
"""
Database configuration and SQLAlchemy engine/session setup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from ..utils.paths import get_db_path

DB_PATH: Path = get_db_path()
SQLALCHEMY_DATABASE_URL: str = f"sqlite:///{DB_PATH.resolve()}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_engine_and_session(
    mode: str = "triton",
) -> Tuple[Engine, sessionmaker[Session]]:
    """
    Create database engine and session factory for the specified mode.

    Args:
        mode: Cache mode - 'triton' or 'vllm'

    Returns:
        Tuple of (engine, session_factory)
    """
    db_path = get_db_path(mode)
    database_url = f"sqlite:///{db_path.resolve()}"

    mode_engine = create_engine(database_url, connect_args={"check_same_thread": False})

    mode_session_local = sessionmaker(
        autocommit=False, autoflush=False, bind=mode_engine
    )

    return mode_engine, mode_session_local
