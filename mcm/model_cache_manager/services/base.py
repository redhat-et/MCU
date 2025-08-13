"""
Base service class for shared initialization logic.
"""

from pathlib import Path
from typing import Union

from ..data.cache_repo import CacheRepository, VllmCacheRepository
from ..data.database import Database, VllmDatabase
from ..utils.paths import get_cache_dir
from ..utils.mcm_constants import MODE_TRITON, MODE_VLLM


class BaseService: # pylint: disable=too-few-public-methods
    """Base service with common initialization for cache and database."""

    def __init__(self, cache_dir: Path | None = None, mode: str = MODE_TRITON):
        """
        Initialize the service with cache and database based on mode.

        Args:
            cache_dir: Path to the cache directory. If None, uses the default.
            mode: Cache mode - 'triton' for standard Triton cache, 'vllm' for vLLM cache.
        """
        self.mode = mode
        self.cache_dir: Path
        self.repo: Union[CacheRepository, VllmCacheRepository]
        self.db: Union[Database, VllmDatabase]
        if mode == MODE_VLLM:
            self.cache_dir = cache_dir or get_cache_dir(mode=mode)
            self.repo = VllmCacheRepository(self.cache_dir)
            self.db = VllmDatabase()
        else:
            self.cache_dir = cache_dir or get_cache_dir(mode=mode)
            self.repo = CacheRepository(self.cache_dir)
            self.db = Database()
