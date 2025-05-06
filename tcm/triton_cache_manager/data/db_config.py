# pylint: disable=c-extension-no-member # Handles potential Pylint issues with SQLAlchemy C extensions
"""
Database configuration and SQLAlchemy engine/session setup.
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ..utils.paths import get_db_path

DB_PATH: Path = get_db_path()
SQLALCHEMY_DATABASE_URL: str = f"sqlite:///{DB_PATH.resolve()}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
