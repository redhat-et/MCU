"""
Path utility functions for the Model Cache Manager.

This module provides functions to get standard paths used by the application.
"""

from pathlib import Path
from model_cache_manager.utils.config import settings


def get_cache_dir(mode: str = 'triton') -> Path:
    """
    Get the path to the Triton cache directory.

    Returns:
        Path to the Triton cache directory.
    """

    if mode == "vllm":
        return settings.model_cache_dir_vllm
    return settings.model_cache_dir


def get_db_path(mode: str = "triton") -> Path:
    """
    Get the path to the database file.
    
    Args:
        mode: Cache mode - 'triton' for standard cache, 'vllm' for vLLM cache

    Returns:
        Path to the SQLite database file.
    """
    if mode == "vllm":
        return settings.data_dir / "cache_vllm.db"
    return settings.data_dir / settings.db_filename
