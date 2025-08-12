"""
Service for indexing Triton kernels from cache into database.

This module provides the main service for scanning the cache and populating the database.
"""

from __future__ import annotations
from typing import Tuple
from .base import BaseService
from ..models.criteria import SearchCriteria
from ..data.cache_repo import VllmCacheRepository
from ..data.database import VllmDatabase


class IndexService(BaseService):
    """
    Build or update the kernel index.

    This service scans the Triton cache directory and inserts or updates
    kernel metadata in the database.
    """


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
            # Type guard: ensure we're using vLLM types
            assert isinstance(self.repo, VllmCacheRepository)
            assert isinstance(self.db, VllmDatabase)
            for vllm_hash, vllm_cache_root, kernel in self.repo.kernels():
                self.db.insert_kernel(kernel, vllm_cache_root, vllm_hash)
                updated_kernels += 1

            return updated_kernels, current_kernels

        criteria = SearchCriteria(cache_dir=self.cache_dir)
        current_kernels = len(self.db.search(criteria))

        updated_kernels = 0
        assert not isinstance(self.repo, VllmCacheRepository)
        assert not isinstance(self.db, VllmDatabase)
        for kernel in self.repo.kernels():
            self.db.insert_kernel(kernel, str(self.cache_dir))
            updated_kernels += 1

        return updated_kernels, current_kernels

    def close(self):
        """Close the database connection."""
        self.db.close()
