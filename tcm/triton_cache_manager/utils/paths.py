"""
Path utility functions for the Triton Cache Manager.

This module provides functions to get standard paths used by the application.
"""

from pathlib import Path
from triton_cache_manager.utils.config import settings


def get_cache_dir() -> Path:
    """
    Get the path to the Triton cache directory.

    Returns:
        Path to the Triton cache directory.
    """
    return settings.triton_cache_dir


def get_db_path() -> Path:
    """
    Get the path to the database file.

    Returns:
        Path to the SQLite database file.
    """
    return settings.data_dir / settings.db_filename
