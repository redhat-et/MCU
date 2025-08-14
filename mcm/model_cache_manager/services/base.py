"""
Base service class for shared initialization logic.
"""

from pathlib import Path
from ..strategies import TritonStrategy, VllmStrategy
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
        self.cache_dir = cache_dir or get_cache_dir(mode=mode)

        # Initialize strategy based on mode
        if mode == MODE_VLLM:
            self.strategy = VllmStrategy()
        else:
            self.strategy = TritonStrategy()

        # Create repository and database using strategy
        self.repo = self.strategy.create_repository(self.cache_dir)
        self.db = self.strategy.create_database()
