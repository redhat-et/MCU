"""
Base service class for shared initialization logic.
"""

from pathlib import Path
from typing import Union

from ..data.cache_repo import CacheRepository, VllmCacheRepository
from ..data.database import Database, VllmDatabase
from ..utils.paths import get_cache_dir


class BaseService:
    """Base service with common initialization for cache and database."""

    def __init__(self, cache_dir: Path | None = None, mode: str = "triton"):
        """
        Initialize the service with cache and database based on mode.

        Args:
            cache_dir: Path to the cache directory. If None, uses the default.
            mode: Cache mode - 'triton' for standard Triton cache, 'vllm' for vLLM cache.
        """
        self.mode = mode
        if mode == "vllm":
            self.cache_dir = cache_dir or get_cache_dir(mode=mode)
            self.repo = VllmCacheRepository(self.cache_dir)
            self.db = VllmDatabase()
        else:
            self.cache_dir = cache_dir or get_cache_dir(mode=mode)
            self.repo = CacheRepository(self.cache_dir)
            self.db = Database()