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
from ..utils.utils import (
    KernelIdentifier,
    create_kernel_identifier,
    process_kernels_in_batches,
)


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
            additional_duplicate_fields=[],
        )

    def create_database(self):
        """Create Database instance for Triton mode."""
        return Database()

    def create_repository(self, cache_dir: Path):
        """Create CacheRepository instance for Triton mode."""
        return CacheRepository(cache_dir)

    def extract_identifiers_from_row(self, row: Dict[str, Any]) -> KernelIdentifier:
        """Extract kernel identifier from Triton database row."""
        return create_kernel_identifier(mode="triton", hash=row["hash"])

    def reindex_kernels(self, repo, db, batch_size: int = 1000) -> int:
        """Perform Triton-specific kernel reindexing using streaming bulk insert.

        Args:
            repo: Repository to read kernels from
            db: Database to insert kernels into
            batch_size: Number of kernels to accumulate before bulk inserting

        Returns:
            Total number of kernels inserted
        """
        cache_dir = str(repo.root)
        # Create generator that yields kernel data tuples
        kernels_iterator = ((kernel, cache_dir) for kernel in repo.kernels())
        return process_kernels_in_batches(kernels_iterator, db, batch_size)

    def insert_kernel_strategy(self, db, k_data, *args, **kwargs) -> None:
        """Strategy-specific kernel insertion for Triton."""
        cache_dir = args[0] if args else kwargs.get("cache_dir")
        db.insert_kernel(k_data, cache_dir)
