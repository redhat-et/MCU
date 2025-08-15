"""
Service for indexing Triton kernels from cache into database.

This module provides the main service for scanning the cache and populating the database.
"""

from __future__ import annotations
from typing import Tuple
from .base import BaseService
from ..models.criteria import SearchCriteria


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

        updated_kernels = self.strategy.reindex_kernels(self.repo, self.db)

        return updated_kernels, current_kernels


    def close(self):
        """Close the database connection."""
        self.db.close()
