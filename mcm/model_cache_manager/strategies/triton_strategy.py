"""
Strategy implementation for Triton cache mode.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
from sqlalchemy import and_

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

    def insert_kernel_strategy(self, db, k_data, cache_dir: str) -> None:
        """Strategy-specific kernel insertion for Triton."""
        db.insert_kernel(k_data, cache_dir)

    def get_cache_dir_from_row(self, row: Dict[str, Any]) -> str:
        """Get cache directory from Triton database row."""
        return row.get("cache_dir", "")

    def build_search_filters(self, criteria, orm_class) -> List:
        """Build Triton-specific search filters."""
        active_filters = []

        equality_filter_configs = [
            ("cache_dir", orm_class.cache_dir, str),
            ("name", orm_class.name, None),
            ("backend", orm_class.backend, None),
            ("arch", orm_class.arch, str),
        ]

        for crit_attr, orm_column, transformer in equality_filter_configs:
            value = getattr(criteria, crit_attr, None)
            if value is not None:
                if transformer:
                    value = transformer(value)
                active_filters.append(orm_column == value)

        if criteria.cache_hit_lower is not None:
            active_filters.append(orm_class.runtime_hits < criteria.cache_hit_lower)

        if criteria.cache_hit_higher is not None:
            active_filters.append(orm_class.runtime_hits > criteria.cache_hit_higher)

        if criteria.older_than_timestamp is not None:
            active_filters.append(orm_class.modified_time < criteria.older_than_timestamp)

        if criteria.younger_than_timestamp is not None:
            active_filters.append(orm_class.modified_time > criteria.younger_than_timestamp)

        return active_filters
