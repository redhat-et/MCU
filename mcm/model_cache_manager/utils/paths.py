"""
Path utility functions for the Model Cache Manager.

This module provides functions to get standard paths used by the application.
"""

from pathlib import Path
from model_cache_manager.utils.config import settings
from model_cache_manager.utils.mcm_constants import (
    MODE_TRITON,
    MODE_VLLM,
    MODE_VLLM_LEGACY,
    MODE_HELION,
)


def get_cache_dir(mode: str = MODE_TRITON) -> Path:
    """
    Get the path to the cache directory for the given mode.

    Returns:
        Path to the cache directory.
    """

    if mode in (MODE_VLLM, MODE_VLLM_LEGACY):
        return settings.model_cache_dir_vllm
    if mode == MODE_HELION:
        return settings.model_cache_dir_helion
    return settings.model_cache_dir


def get_db_path(mode: str = MODE_TRITON) -> Path:
    """
    Get the path to the database file.

    Args:
        mode: Cache mode - 'triton' for standard cache, 'vllm' for vLLM cache

    Returns:
        Path to the SQLite database file.
    """
    db_filename = f"{settings.db_filename}{mode}.db"
    return settings.data_dir / db_filename
