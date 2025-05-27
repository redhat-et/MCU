"""
Repository module for accessing and managing Triton kernel cache files.

This module provides functionality to scan and parse the Triton cache directory.
"""

from __future__ import annotations
from pathlib import Path
import os
import json
import logging
from typing import Iterable, Optional
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

    def _process_group_metadata_file(
        self, d: Path, group_meta_file: Path
    ) -> Optional[Path]:
        """
        Tries to load the actual metadata path from a single group metadata file.
        Returns the path to the actual metadata file if found, otherwise None.
        """
        log.debug(
            "Processing group metadata file: %s in directory %s.",
            group_meta_file,
            d,
        )
        try:
            group_data = json.loads(group_meta_file.read_text(encoding="utf-8"))
            child_paths = group_data.get("child_paths")

            if not child_paths or not isinstance(child_paths, dict):
                log.warning(
                    "Skipping kernel in '%s', group metadata '%s' "
                    "is missing 'child_paths' or it's not a dictionary.",
                    d,
                    group_meta_file.name,
                )
                return None

            for key, path_str in child_paths.items():
                if key.endswith(".json"):
                    candidate_actual_meta_path = Path(path_str)
                    if candidate_actual_meta_path.is_file():
                        log.debug(
                            "Derived actual metadata path %s from group file '%s' using key '%s'.",
                            candidate_actual_meta_path,
                            group_meta_file.name,
                            key,
                        )
                        return candidate_actual_meta_path
                    log.warning(
                        "Path '%s' for key '%s' in group file '%s' (dir '%s') "
                        "does not point to an existing file.",
                        path_str,
                        key,
                        group_meta_file.name,
                        d,
                    )
            log.warning(
                "Skipping kernel in '%s', no valid '*.json' entry in 'child_paths' "
                "of group metadata file '%s' pointed to an existing file.",
                d,
                group_meta_file.name,
            )
            return None

        except json.JSONDecodeError as e:
            log.error(
                "Skipping kernel in '%s', failed to parse group metadata JSON '%s': %s",
                d,
                group_meta_file.name,
                e,
            )
        except OSError as e:
            log.error(
                "Skipping kernel in '%s', OS error reading group metadata file '%s': %s",
                d,
                group_meta_file.name,
                e,
            )
        except Exception as e: # pylint: disable=broad-except
            # This broad except is a fallback for truly unexpected errors during this specific step.
            log.error(
                "Skipping kernel in '%s', unexpected error processing group "
                "metadata file '%s': %s",
                d,
                group_meta_file.name,
                e,
            )
        return None

    def _load_kernel_from_metadata(
        self, d: Path, meta_path_to_load: Path
    ) -> Optional[Kernel]:
        """Loads and deserializes a kernel from its metadata file."""
        try:
            data = json.loads(meta_path_to_load.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log.error(
                "Skipping kernel, failed to parse metadata JSON '%s': %s",
                meta_path_to_load,
                e,
            )
            return None
        except OSError as e:
            log.error(
                "Skipping kernel, OS error reading metadata file '%s': %s",
                meta_path_to_load,
                e,
            )
            return None
        except Exception as e: # pylint: disable=broad-except
            log.error(
                "Skipping kernel, unexpected error processing metadata file '%s': %s",
                meta_path_to_load,
                e,
            )
            return None

        kernel = deserialize_kernel(data, d.name, d, self.plugins)
        if kernel:
            return kernel

        log.warning(
            "Skipping invalid kernel at '%s' (metadata from '%s')",
            d,
            meta_path_to_load,
        )
        return None

    def kernels(self) -> Iterable[Kernel]:
        """
        Iterate through all kernels in the cache directory.

        Returns:
            Iterable of valid Kernel objects with metadata parsed from cache files.
            Invalid kernels are logged and skipped.
        """
        for d in self._dirs():
            meta_path_to_load: Optional[Path] = None

            all_json_files = list(d.glob("*.json"))
            group_json_files = [
                f for f in all_json_files if f.name.startswith("__grp__")
            ]

            if group_json_files:
                meta_path_to_load = self._process_group_metadata_file(
                    d, group_json_files[0]
                )
                if meta_path_to_load is None:
                    continue

            if not meta_path_to_load:
                if not group_json_files:
                    log.debug(
                        "No group metadata JSON found for kernel in directory %s.",
                        d,
                    )
                continue

            kernel = self._load_kernel_from_metadata(d, meta_path_to_load)
            if kernel:
                yield kernel
