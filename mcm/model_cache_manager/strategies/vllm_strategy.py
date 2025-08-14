"""
Strategy implementation for vLLM cache mode.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
from sqlalchemy import and_

from .base import CacheModeStrategy, CacheConfig
from ..data.database import VllmDatabase
from ..data.cache_repo import VllmCacheRepository  
from ..data.db_models import VllmKernelOrm, VllmKernelFileOrm
from ..utils.utils import KernelIdentifier, create_kernel_identifier


class VllmStrategy(CacheModeStrategy):
    """Strategy for handling vLLM cache mode operations."""
    
    @property
    def config(self) -> CacheConfig:
        """Return vLLM cache configuration."""
        return CacheConfig(
            orm_model=VllmKernelOrm,
            file_orm_model=VllmKernelFileOrm,
            hash_field="triton_cache_key",
            primary_key_fields=["vllm_cache_root", "vllm_hash", "triton_cache_key"],
            additional_duplicate_fields=["vllm_hash"]
        )
    
    def create_database(self):
        """Create VllmDatabase instance for vLLM mode."""
        return VllmDatabase()
    
    def create_repository(self, cache_dir: Path):
        """Create VllmCacheRepository instance for vLLM mode."""
        return VllmCacheRepository(cache_dir)
    
    def extract_identifiers_from_row(self, row: Dict[str, Any]) -> KernelIdentifier:
        """Extract kernel identifier from vLLM database row."""
        return create_kernel_identifier(
            mode="vllm",
            vllm_hash=row["vllm_hash"],
            triton_cache_key=row["triton_cache_key"]
        )
    
    def reindex_kernels(self, repo, db) -> int:
        """Perform vLLM-specific kernel reindexing."""
        updated_kernels = 0
        for vllm_hash, vllm_cache_root, kernel in repo.kernels():
            self.insert_kernel_strategy(db, kernel, vllm_cache_root, vllm_hash)
            updated_kernels += 1
        return updated_kernels
    
    def insert_kernel_strategy(self, db, k_data, vllm_cache_root: str, vllm_hash: str) -> None:
        """Strategy-specific kernel insertion for vLLM."""
        db.insert_kernel(k_data, vllm_cache_root, vllm_hash)
    
    def get_cache_dir_from_row(self, row: Dict[str, Any]) -> str:
        """Get cache directory from vLLM database row."""
        return row.get("vllm_cache_root", "")
    
    def build_search_filters(self, criteria, orm_class) -> List:
        """Build vLLM-specific search filters."""
        active_filters = []
        
        equality_filter_configs = [
            ("cache_dir", orm_class.vllm_cache_root, str),
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