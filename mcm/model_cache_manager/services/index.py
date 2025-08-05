"""
Service for indexing Triton kernels from cache into database.

This module provides the main service for scanning the cache and populating the database.
"""

from __future__ import annotations
from typing import Tuple
from pathlib import Path
from ..data.cache_repo import CacheRepository, VllmCacheRepository
from ..data.database import Database, VllmDatabase
from ..models.criteria import SearchCriteria
from ..utils.paths import get_cache_dir


class IndexService:
    """
    Build or update the kernel index.

    This service scans the Triton cache directory and inserts or updates
    kernel metadata in the database.
    """

    def __init__(self, cache_dir: Path | None = None, mode: str = "triton"):
        """
        Initialize the index service.

        Args:
            cache_dir: Path to the cache directory. If None, uses the default.
            mode: Cache mode - 'triton' for standard Triton cache, 'vllm' for vLLM cache.
        """
        self.mode = mode
        if mode == "vllm":
            self.cache_dir = cache_dir or get_cache_dir(mode=mode)
            self.repo = VllmCacheRepository(self.cache_dir)
            self.db = VllmDatabase()
        else:
            self.cache_dir = cache_dir or get_cache_dir(mode=mode)
            self.repo = CacheRepository(self.cache_dir)
            self.db = Database()

    def reindex(self) -> Tuple[int, int]:
        """
        Scan the cache directory and update the database.

        Returns:
            Number of kernels indexed.
        """
        if self.mode == "vllm":
            criteria = SearchCriteria(cache_dir=self.cache_dir)
            current_kernels = len(self.db.search(criteria))

            updated_kernels = 0
            for vllm_hash, vllm_cache_root, kernel in self.repo.kernels():
                self.db.insert_kernel(kernel, vllm_cache_root, vllm_hash)
                updated_kernels += 1

            return updated_kernels, current_kernels
        else:
            criteria = SearchCriteria(cache_dir=self.cache_dir)
            current_kernels = len(self.db.search(criteria))

            updated_kernels = 0
            for kernel in self.repo.kernels():
                self.db.insert_kernel(kernel, str(self.cache_dir))
                updated_kernels += 1

            return updated_kernels, current_kernels

    def close(self):
        """Close the database connection."""
        self.db.close()
