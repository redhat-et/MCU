"""
Strategy implementation for legacy vLLM cache mode.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

from .base import CacheModeStrategy, CacheConfig
from ..data.vllm_legacy_database import VllmLegacyDatabase
from ..data.cache_repo import VllmLegacyCacheRepository
from ..data.db_models import VllmLegacyKernelOrm, VllmLegacyKernelFileOrm
from ..utils.utils import (
    KernelIdentifier,
    create_kernel_identifier,
    process_kernels_in_batches,
)
from ..utils.strategy_constants import VLLM_COMMON_PRIMARY_FIELDS, VLLM_HASH_FIELD


class VllmLegacyStrategy(CacheModeStrategy):
    """Strategy for handling legacy vLLM cache mode operations."""

    @property
    def config(self) -> CacheConfig:
        """Return legacy vLLM cache configuration."""
        return CacheConfig(
            orm_model=VllmLegacyKernelOrm,
            file_orm_model=VllmLegacyKernelFileOrm,
            hash_field=VLLM_HASH_FIELD,
            primary_key_fields=VLLM_COMMON_PRIMARY_FIELDS,
            additional_duplicate_fields=["vllm_hash"],
        )

    def create_database(self):
        """Create VllmLegacyDatabase instance for legacy vLLM mode."""
        return VllmLegacyDatabase()

    def create_repository(self, cache_dir: Path):
        """Create VllmLegacyCacheRepository instance for legacy vLLM mode."""
        return VllmLegacyCacheRepository(cache_dir)

    def extract_identifiers_from_row(self, row: Dict[str, Any]) -> KernelIdentifier:
        """Extract kernel identifier from legacy vLLM database row."""
        return create_kernel_identifier(
            mode="vllm-legacy",
            vllm_hash=row["vllm_hash"],
            triton_cache_key=row["triton_cache_key"],
            rank_x_y=row["rank_x_y"],
        )

    def reindex_kernels(self, repo, db, batch_size: int = 1000) -> int:
        """Perform legacy vLLM-specific kernel reindexing using streaming bulk insert.

        Args:
            repo: Repository to read kernels from
            db: Database to insert kernels into
            batch_size: Number of kernels to accumulate before bulk inserting

        Returns:
            Total number of kernels inserted
        """
        # Create generator that yields kernel data tuples
        kernels_iterator = (
            (kernel, cache_dir, vllm_hash, rank_x_y)
            for vllm_hash, cache_dir, rank_x_y, kernel in repo.kernels()
        )
        return process_kernels_in_batches(kernels_iterator, db, batch_size)

    def insert_kernel_strategy(self, db, k_data, *args, **kwargs) -> None:
        """Strategy-specific kernel insertion for legacy vLLM."""
        cache_dir = args[0] if len(args) > 0 else kwargs.get("cache_dir")
        vllm_hash = args[1] if len(args) > 1 else kwargs.get("vllm_hash")
        rank_x_y = args[2] if len(args) > 2 else kwargs.get("rank_x_y")
        db.insert_kernel(k_data, cache_dir, vllm_hash, rank_x_y)
