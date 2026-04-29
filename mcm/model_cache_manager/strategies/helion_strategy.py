"""
Strategy implementation for Helion cache mode.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

from .base import CacheModeStrategy, CacheConfig
from ..data.database import HelionDatabase
from ..data.cache_repo import HelionCacheRepository
from ..data.db_models import HelionKernelOrm, HelionKernelFileOrm
from ..utils.utils import (
    KernelIdentifier,
    create_kernel_identifier,
    process_kernels_in_batches,
)


class HelionStrategy(CacheModeStrategy):
    """Strategy for handling Helion cache mode operations."""

    @property
    def config(self) -> CacheConfig:
        """Return Helion cache configuration."""
        return CacheConfig(
            orm_model=HelionKernelOrm,
            file_orm_model=HelionKernelFileOrm,
            hash_field="triton_cache_key",
            primary_key_fields=["triton_cache_key", "cache_dir"],
        )

    def create_database(self):
        """Create HelionDatabase instance."""
        return HelionDatabase()

    def create_repository(self, cache_dir: Path):
        """Create HelionCacheRepository instance."""
        return HelionCacheRepository(cache_dir)

    def extract_identifiers_from_row(self, row: Dict[str, Any]) -> KernelIdentifier:
        """Extract kernel identifier from Helion database row."""
        return create_kernel_identifier(
            mode="helion", hash=row["triton_cache_key"]
        )

    def reindex_kernels(self, repo, db, batch_size: int = 1000) -> int:
        """Perform Helion-specific kernel reindexing using streaming bulk insert."""
        kernels_iterator = (
            (kernel, cache_dir, helion_hash, best_config, is_best)
            for cache_dir, _triton_cache_key, helion_hash,
            best_config, is_best, kernel in repo.kernels()
        )
        return process_kernels_in_batches(kernels_iterator, db, batch_size)

    def insert_kernel_strategy(self, db, k_data, *args, **kwargs) -> None:
        """Strategy-specific kernel insertion for Helion."""
        cache_dir = args[0] if len(args) > 0 else kwargs.get("cache_dir")
        helion_hash = args[1] if len(args) > 1 else kwargs.get("helion_hash")
        best_config = args[2] if len(args) > 2 else kwargs.get("best_config")
        is_best = args[3] if len(args) > 3 else kwargs.get("is_best", False)
        db.insert_kernel(k_data, cache_dir, helion_hash, best_config, is_best)
