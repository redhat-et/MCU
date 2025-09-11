"""
Strategy implementation for Triton cache mode.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

from .base import CacheModeStrategy, CacheConfig
from ..data.database import Database
from ..data.cache_repo import CacheRepository
from ..data.db_models import KernelOrm, KernelFileOrm
from ..utils.utils import KernelIdentifier, create_kernel_identifier


class TritonStrategy(CacheModeStrategy):
    """Strategy for handling Triton cache mode operations."""

    @property
    def config(self) -> CacheConfig:
        """Return Triton cache configuration."""
        return CacheConfig(
            orm_model=KernelOrm,
            file_orm_model=KernelFileOrm,
            hash_field="hash",
            primary_key_fields=["hash", "cache_dir"],
            additional_duplicate_fields=[]
        )

    def create_database(self):
        """Create Database instance for Triton mode."""
        return Database()

    def create_repository(self, cache_dir: Path):
        """Create CacheRepository instance for Triton mode."""
        return CacheRepository(cache_dir)

    def extract_identifiers_from_row(self, row: Dict[str, Any]) -> KernelIdentifier:
        """Extract kernel identifier from Triton database row."""
        return create_kernel_identifier(
            mode="triton",
            hash=row["hash"]
        )

    def reindex_kernels(self, repo, db) -> int:
        """Perform Triton-specific kernel reindexing."""
        updated_kernels = 0
        for kernel in repo.kernels():
            self.insert_kernel_strategy(db, kernel, str(repo.root))
            updated_kernels += 1
        return updated_kernels

    def insert_kernel_strategy(self, db, k_data, *args, **kwargs) -> None:
        """Strategy-specific kernel insertion for Triton."""
        cache_dir = args[0] if args else kwargs.get('cache_dir')
        db.insert_kernel(k_data, cache_dir)
