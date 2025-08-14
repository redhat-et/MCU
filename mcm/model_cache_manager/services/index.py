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
from ..utils.mcm_constants import MODE_VLLM


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
        criteria = SearchCriteria(cache_dir=self.cache_dir)
        current_kernels = len(self.db.search(criteria))

        if self.mode == MODE_VLLM:
            updated_kernels = self._reindex_vllm_mode()
        else:
            updated_kernels = self._reindex_triton_mode()

        return updated_kernels, current_kernels

    def _reindex_vllm_mode(self) -> int:
        """Reindex for vLLM mode."""
        # Type guard: ensure we're using vLLM types
        if not isinstance(self.repo, VllmCacheRepository):
            raise TypeError(
                f"Expected self.repo to be VllmCacheRepository, got {type(self.repo).__name__}"
            )
        if not isinstance(self.db, VllmDatabase):
            raise TypeError(
                f"Expected self.db to be VllmDatabase, got {type(self.db).__name__}"
            )

        updated_kernels = 0
        for vllm_hash, vllm_cache_root, kernel in self.repo.kernels():
            self.db.insert_kernel(kernel, vllm_cache_root, vllm_hash)
            updated_kernels += 1

        return updated_kernels

    def _reindex_triton_mode(self) -> int:
        """Reindex for Triton mode."""
        if isinstance(self.repo, VllmCacheRepository):
            raise TypeError(
                "self.repo should not be a VllmCacheRepository in this mode"
            )
        if isinstance(self.db, VllmDatabase):
            raise TypeError("self.db should not be a VllmDatabase in this mode")

        updated_kernels = 0
        for kernel in self.repo.kernels():
            self.db.insert_kernel(kernel, str(self.cache_dir))
            updated_kernels += 1

        return updated_kernels

    def close(self):
        """Close the database connection."""
        self.db.close()
