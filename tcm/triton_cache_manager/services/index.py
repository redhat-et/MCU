"""
Service for indexing Triton kernels from cache into database.

This module provides the main service for scanning the cache and populating the database.
"""

from __future__ import annotations
from typing import Tuple
from pathlib import Path
from ..data.cache_repo import CacheRepository
from ..data.database import Database
from ..models.criteria import SearchCriteria
from ..utils.paths import get_cache_dir

class IndexService:
    """
    Build or update the kernel index.

    This service scans the Triton cache directory and inserts or updates
    kernel metadata in the database.
    """

    def __init__(self, cache_dir: Path | None = None):
        """
        Initialize the index service.

        Args:
            cache_dir: Path to the Triton cache directory. If None, uses the default.
        """
        self.cache_dir = cache_dir or get_cache_dir()
        self.repo = CacheRepository(cache_dir)
        self.db = Database()

    def reindex(self) -> Tuple[int, int]:
        """
        Scan the cache directory and update the database.

        Returns:
            Number of kernels indexed.
        """
        criteria = SearchCriteria()

        current_kernels = len(self.db.search(criteria))

        updated_kernels = 0
        for kernel in self.repo.kernels():
            self.db.insert_kernel(kernel, self.cache_dir)
            updated_kernels += 1

        return updated_kernels, current_kernels

    def close(self):
        """Close the database connection."""
        self.db.close()
