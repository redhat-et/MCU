"""
Abstract base class for cache mode strategies.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Type, List, Dict, Any

from ..utils.utils import KernelIdentifier, build_common_search_filters


@dataclass
class CacheConfig:
    """Configuration for cache mode strategy."""

    orm_model: Type
    file_orm_model: Type
    hash_field: str
    primary_key_fields: List[str]
    additional_duplicate_fields: List[str] | None = None


class CacheModeStrategy(ABC):
    """Abstract strategy for handling cache mode-specific operations."""

    @property
    @abstractmethod
    def config(self) -> CacheConfig:
        """Return configuration for this cache mode."""

    @abstractmethod
    def create_database(self) -> Any:
        """Create appropriate database instance for this mode."""

    @abstractmethod
    def create_repository(self, cache_dir: Path) -> Any:
        """Create appropriate repository instance for this mode."""

    @abstractmethod
    def extract_identifiers_from_row(self, row: Dict[str, Any]) -> KernelIdentifier:
        """Extract kernel identifier from database row."""

    @abstractmethod
    def reindex_kernels(self, repo, db) -> int:
        """Perform mode-specific kernel reindexing."""

    @abstractmethod
    def insert_kernel_strategy(self, db, k_data, *args, **kwargs) -> None:
        """Strategy-specific kernel insertion."""

    def get_cache_dir_from_row(self, row: Dict[str, Any]) -> str:
        """Get cache directory path from database row."""
        return row.get("cache_dir", "")

    def build_search_filters(self, criteria, orm_class) -> List:
        """Build search filters for database queries.

        Default implementation for common filter configuration.
        Subclasses can override if they need different behavior.
        """
        equality_filter_configs = [
            ("cache_dir", orm_class.cache_dir, str),
            ("name", orm_class.name, None),
            ("backend", orm_class.backend, None),
            ("arch", orm_class.arch, str),
        ]

        return build_common_search_filters(criteria, orm_class, equality_filter_configs)
