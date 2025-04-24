from __future__ import annotations
from pathlib import Path
from ..data.cache_repo import CacheRepository
from ..data.database import Database


class IndexService:
    """Build or update the kernel index."""

    def __init__(self, cache_dir: Path | None = None):
        self.repo = CacheRepository(cache_dir)
        self.db = Database()

    def reindex(self) -> int:
        total = 0
        for kernel in self.repo.kernels():
            self.db.insert_kernel(kernel)
            total += 1
        return total

    def close(self):
        self.db.close()
