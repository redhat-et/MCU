"""
Runtime tracking functionality for Triton Cache Manager.

This module provides cache hit/miss tracking that integrates with TCM's database.
"""

from typing import Any, Dict, List, Optional, TypedDict
import logging
import time
from threading import Lock
from pathlib import Path

from triton.runtime.cache import CacheManager, FileCacheManager
from triton import knobs

from triton_cache_manager.data.database import Database
from triton_cache_manager.data.db_models import KernelOrm
from triton_cache_manager.utils.paths import get_cache_dir


log = logging.getLogger(__name__)


class StatsDict(TypedDict):  # pylint: disable=too-few-public-methods
    """Typed dictionary for kernel statistics."""

    hits: int
    misses: int
    last_access: float

# pylint: disable=too-few-public-methods
class CacheAccessRecord:  # pylint: disable=too-many-instance-attributes
    """Record of a single cache access event."""

    # pylint: disable=too-many-positional-arguments
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        cache_key: str,
        kernel_name: str,
        hit: bool,
        timestamp: float,
        cache_dir: Optional[Path] = None,
    ):
        self.cache_key = cache_key
        self.kernel_name = kernel_name
        self.hit = hit
        self.timestamp = timestamp
        self.cache_dir = cache_dir or get_cache_dir()


class RuntimeStatsCollector:
    """Collects and persists runtime cache statistics to TCM database."""

    def __init__(self):
        self._lock = Lock()
        self._pending_records: List[CacheAccessRecord] = []
        self._db = Database()
        self._enabled = True

    def enable(self):
        """Enable statistics collection."""
        self._enabled = True

    def disable(self):
        """Disable statistics collection."""
        self._enabled = False

    def record_access(
        self,
        cache_key: str,
        kernel_name: str,
        hit: bool,
        cache_dir: Optional[Path] = None,
    ):
        """Record a cache access event."""
        if not self._enabled:
            return

        with self._lock:
            self._pending_records.append(
                CacheAccessRecord(cache_key, kernel_name, hit, time.time(), cache_dir)
            )
            self._persist_to_database()

    @staticmethod
    def _create_stats_dict() -> StatsDict:
        """Create a properly typed stats dictionary."""
        return StatsDict(hits=0, misses=0, last_access=0.0)

    def _persist_to_database(self):
        """Persist pending records to the TCM database."""
        if not self._pending_records:
            return
        cache_dir: Optional[Path] = None

        try:
            stats_by_key: Dict[str, StatsDict] = {}

            for record in self._pending_records:
                if record.cache_key not in stats_by_key:
                    stats_by_key[record.cache_key] = self._create_stats_dict()

                stats = stats_by_key[record.cache_key]
                if record.hit:
                    stats["hits"] += 1
                else:
                    stats["misses"] += 1
                stats["last_access"] = max(stats["last_access"], record.timestamp)
                cache_dir = record.cache_dir.parent

            session = self._db.get_session()
            try:
                for cache_key, stats in stats_by_key.items():
                    kernels = (
                        session.query(KernelOrm)
                        .filter(
                            KernelOrm.hash == cache_key,
                            KernelOrm.cache_dir == str(cache_dir),
                        )
                        .all()
                    )

                    for kernel in kernels:
                        current_hits: int = kernel.runtime_hits or 0
                        current_misses: int = kernel.runtime_misses or 0

                        kernel.runtime_hits = current_hits + stats["hits"]
                        kernel.runtime_misses = current_misses + stats["misses"]
                        kernel.last_access_time = stats["last_access"]

                session.commit()
            finally:
                session.close()

            self._pending_records.clear()

        except Exception as exc:  # pylint: disable=broad-except
            log.error("Failed to persist runtime stats: %s", exc, exc_info=False)

    def flush(self):
        """Force persist all pending records."""
        with self._lock:
            self._persist_to_database()

    def close(self):
        """Clean up resources."""
        self.flush()
        if self._db:
            self._db.close()


# Global collector instance
_runtime_collector = RuntimeStatsCollector()


class TCMTrackingCacheManager(CacheManager):
    """Cache manager that tracks runtime statistics for TCM."""

    def __init__(self, key: str, override: bool = False, dump: bool = False):
        self.full_cache_key = key

        # Determine the base cache manager to wrap
        base_cache_cls = FileCacheManager
        if knobs.cache.manager_class not in (TCMTrackingCacheManager, None):
            base_cache_cls = knobs.cache.manager_class

        self._base_manager = base_cache_cls(key, override=override, dump=dump)

        # Try to determine cache directory from the base manager
        if hasattr(self._base_manager, "cache_dir"):
            self._cache_dir = Path(self._base_manager.cache_dir)
        elif hasattr(knobs.cache, "dir"):
            self._cache_dir = Path(knobs.cache.dir)
        else:
            self._cache_dir = get_cache_dir()

    def get_file(self, filename: str) -> Optional[str]:
        """Intercepts file retrieval to track hits/misses."""
        return self._base_manager.get_file(filename)

    def put(self, data: Any, filename: str, binary: bool = True) -> str:
        """Passes through put operations to the base manager."""
        return self._base_manager.put(data, filename, binary)

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        """Intercepts group retrieval to track cache access."""
        group = self._base_manager.get_group(filename)
        kernel_name = filename.split(".")[0]

        _runtime_collector.record_access(
            cache_key=self.full_cache_key,
            kernel_name=kernel_name,
            hit=group is not None,
            cache_dir=self._cache_dir,
        )

        return group

    def put_group(self, filename: str, group: Dict[str, str]) -> None:
        """Passes through put_group operations to the base manager."""
        self._base_manager.put_group(filename, group)
