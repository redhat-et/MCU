"""
Strategy implementation for new vLLM cache mode.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

from .base import CacheModeStrategy, CacheConfig
from ..data.database import VllmDatabase
from ..data.cache_repo import VllmCacheRepository
from ..data.db_models import VllmKernelOrm, VllmKernelFileOrm
from ..utils.utils import (
    KernelIdentifier,
    create_kernel_identifier,
    process_kernels_in_batches,
)
from ..models.kernel import VllmKernelMetadata
from ..utils.strategy_constants import VLLM_EXTENDED_PRIMARY_FIELDS, VLLM_HASH_FIELD


class VllmStrategy(CacheModeStrategy):
    """Strategy for handling new vLLM cache mode operations."""

    @property
    def config(self) -> CacheConfig:
        """Return new vLLM cache configuration."""
        return CacheConfig(
            orm_model=VllmKernelOrm,
            file_orm_model=VllmKernelFileOrm,
            hash_field=VLLM_HASH_FIELD,
            primary_key_fields=VLLM_EXTENDED_PRIMARY_FIELDS,
            additional_duplicate_fields=["vllm_hash", "artifact_compile_range"],
        )

    def create_database(self):
        """Create VllmDatabase instance for new vLLM mode."""
        return VllmDatabase()

    def create_repository(self, cache_dir: Path):
        """Create VllmCacheRepository instance for new vLLM mode."""
        return VllmCacheRepository(cache_dir)

    def extract_identifiers_from_row(self, row: Dict[str, Any]) -> KernelIdentifier:
        """Extract kernel identifier from new vLLM database row."""
        identifier = create_kernel_identifier(
            mode="vllm",
            vllm_hash=row["vllm_hash"],
            triton_cache_key=row["triton_cache_key"],
            rank_x_y=row["rank_x_y"],
        )
        # Add artifact_compile_range as an attribute if present
        if "artifact_compile_range" in row:
            identifier.artifact_compile_range = row["artifact_compile_range"]
        return identifier

    def reindex_kernels(self, repo, db, batch_size: int = 1000) -> int:
        """Perform new vLLM-specific kernel reindexing using streaming bulk insert.

        Args:
            repo: Repository to read kernels from
            db: Database to insert kernels into
            batch_size: Number of kernels to accumulate before bulk inserting

        Returns:
            Total number of kernels inserted
        """
        # Create generator that yields kernel data tuples
        kernels_iterator = (
            (kernel, cache_dir, vllm_hash, rank_x_y, artifact_compile_range, best_config)
            for vllm_hash, cache_dir, rank_x_y, artifact_compile_range,
            best_config, kernel in repo.kernels()
        )
        return process_kernels_in_batches(kernels_iterator, db, batch_size)

    def insert_kernel_strategy(self, db, k_data, *args, **kwargs) -> None:
        """Strategy-specific kernel insertion for new vLLM."""

        cache_dir = args[0] if len(args) > 0 else kwargs.get("cache_dir")
        vllm_hash = args[1] if len(args) > 1 else kwargs.get("vllm_hash")
        rank_x_y = args[2] if len(args) > 2 else kwargs.get("rank_x_y")
        artifact_compile_range = args[3] if len(args) > 3 else kwargs.get("artifact_compile_range")
        best_config = args[4] if len(args) > 4 else kwargs.get("best_config")

        vllm_meta = VllmKernelMetadata(
            vllm_hash=vllm_hash,
            rank_x_y=rank_x_y,
            artifact_compile_range=artifact_compile_range,
            best_config=best_config,
        )
        db.insert_kernel(k_data, cache_dir, vllm_meta)
