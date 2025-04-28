"""
Repository module for accessing and managing Triton kernel cache files.

This module provides functionality to scan and parse the Triton cache directory.
"""

from __future__ import annotations
from pathlib import Path
import os
import json
import logging
from typing import Iterable
from ..utils.paths import get_cache_dir
from ..models.kernel import Kernel
from ..plugins.discovery import discover_plugins
from .kernel_validator import deserialize_kernel

log = logging.getLogger(__name__)


class CacheRepository:
    # pylint: disable=too-few-public-methods
    """
    Repository for accessing and managing Triton kernel cache files.

    This class provides methods to iterate through kernels in the cache directory
    and extract their metadata and associated files.
    """

    def __init__(self, root: Path | None = None):
        """
        Initialize the cache repository.

        Args:
            root: Path to the Triton cache directory. If None, uses the default location.

        Raises:
            FileNotFoundError: If the cache directory doesn't exist.
        """
        self.root = root or get_cache_dir()
        if not self.root.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.root}")
        self.plugins = {p.backend: p for p in discover_plugins()}

    def _dirs(self):
        """
        Yield directories within the cache root.

        Returns:
            Iterator of Path objects for each directory.
        """

        with os.scandir(self.root) as it:
            for e in it:
                if e.is_dir():
                    yield Path(e.path)

    def kernels(self) -> Iterable[Kernel]:
        """
        Iterate through all kernels in the cache directory.

        Returns:
            Iterable of valid Kernel objects with metadata parsed from cache files.
            Invalid kernels are logged and skipped.
        """
        for d in self._dirs():
            meta = next(d.glob("*.json"), None)
            if not meta:
                continue

            try:
                data = json.loads(meta.read_text())
            except json.JSONDecodeError as e:
                log.error(
                    "Skipping kernel, failed to parse metadata JSON '%s': %s", meta, e
                )
                continue
            except OSError as e:
                log.error(
                    "Skipping kernel, OS error reading metadata file '%s': %s", meta, e
                )
                continue

            kernel = deserialize_kernel(data, d.name, d, self.plugins)
            if kernel:
                yield kernel
            else:
                log.warning("Skipping invalid kernel at '%s'", d)
