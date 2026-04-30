"""
Repository module for accessing and managing kernel cache files.

This module provides functionality to scan and parse the model cache directory.
"""

from __future__ import annotations
from pathlib import Path
import json
import logging
import os
import threading
from typing import Iterable, Optional
from ..utils.paths import get_cache_dir
from ..utils.utils import iter_artifact_compile_range_dirs, resolve_helion_triton_dir
from ..utils.mcm_constants import MODE_HELION
from ..models.kernel import Kernel
from ..plugins.discovery import discover_plugins
from .kernel_validator import deserialize_kernel

log = logging.getLogger(__name__)

# Thread-safe lazy-loaded plugins to avoid import-time discovery failures
# pylint: disable=invalid-name
_PLUGINS_CACHE = None
_PLUGINS_LOCK = threading.Lock()


def _get_plugins():
    """Get plugins dictionary, loading them lazily on first access with thread safety."""
    global _PLUGINS_CACHE  # pylint: disable=global-statement
    if _PLUGINS_CACHE is None:
        with _PLUGINS_LOCK:
            # Double-check pattern to avoid race conditions
            if _PLUGINS_CACHE is None:
                _PLUGINS_CACHE = {p.backend: p for p in discover_plugins()}
    return _PLUGINS_CACHE


def _read_json(path: Path, ctx: str) -> Optional[dict]:
    """
    Read and parse JSON file with consistent error handling.

    Args:
        path: Path to the JSON file to read
        ctx: Context string for error messages (e.g., "metadata", "group")

    Returns:
        Parsed JSON data or None if reading/parsing failed
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        log.error("Failed to parse JSON (%s) '%s': %s", ctx, path, e)
    except OSError as e:
        log.error("OS error reading JSON (%s) '%s': %s", ctx, path, e)
    except Exception as e:  # pylint: disable=broad-except
        log.error("Unexpected error reading JSON (%s) '%s': %s", ctx, path, e)
    return None


def _resolve_group_metadata(cache_root: Path, kernel_dir: Path) -> Optional[Path]:
    """
    Resolve the actual metadata file path from group metadata.

    Args:
        cache_root: Root cache directory
        kernel_dir: Directory containing the kernel files

    Returns:
        Path to the actual metadata file, or None if not found
    """
    grp_files = [f for f in kernel_dir.glob("*.json") if f.name.startswith("__grp__")]
    if not grp_files:
        return None

    grp = grp_files[0]
    group_data = _read_json(grp, "group")
    if not group_data:
        return None

    child_paths = group_data.get("child_paths")
    if not isinstance(child_paths, dict):
        log.warning("Missing/invalid 'child_paths' in '%s'", grp)
        return None

    for key, path_str in child_paths.items():
        if not key.endswith(".json"):
            continue

        p = Path(path_str)
        if p.parent.name and p.name:
            derived = cache_root / p.parent.name / p.name
            if derived.is_file():
                return derived

        candidate = Path(path_str)
        if candidate.is_file():
            return candidate

        log.debug("Child path candidate does not exist: %s", path_str)

    log.warning("No valid '*.json' in 'child_paths' of '%s'", grp)
    return None


def iter_triton_kernels(cache_root: Path, plugins: dict) -> Iterable[Kernel]:
    """
    Iterate over Triton kernels in a cache directory.

    Args:
        cache_root: Root directory containing kernel subdirectories
        plugins: Dictionary of backend plugins

    Yields:
        Valid Kernel objects with metadata parsed from cache files
    """
    for kernel_dir in (p for p in cache_root.iterdir() if p.is_dir()):
        meta_path = _resolve_group_metadata(cache_root, kernel_dir)
        if not meta_path:
            log.debug("No group metadata JSON for %s", kernel_dir)
            continue

        data = _read_json(meta_path, "metadata")
        if not data:
            continue

        kernel = deserialize_kernel(
            data, kernel_dir.name, str(cache_root), kernel_dir, plugins
        )
        if kernel:
            yield kernel
        else:
            log.warning(
                "Skipping invalid kernel at '%s' (meta '%s')", kernel_dir, meta_path
            )


class CacheRepository:
    # pylint: disable=too-few-public-methods
    """
    Repository for accessing and managing kernel cache files.

    This class provides methods to iterate through kernels in the cache directory
    and extract their metadata and associated files.
    """

    def __init__(self, root: Path | None = None):
        """
        Initialize the cache repository.

        Args:
            root: Path to the model cache directory. If None, uses the default location.

        Raises:
            FileNotFoundError: If the cache directory doesn't exist.
        """
        self.root = root or get_cache_dir()
        if not self.root.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.root}")

    def kernels(self) -> Iterable[Kernel]:
        """
        Iterate through all kernels in the cache directory.

        Returns:
            Iterable of valid Kernel objects with metadata parsed from cache files.
            Invalid kernels are logged and skipped.
        """
        yield from iter_triton_kernels(self.root, _get_plugins())


class VllmLegacyCacheRepository:  # pylint: disable=too-few-public-methods
    """
    Repository for accessing and managing legacy vLLM kernel cache files.

    This class provides methods to iterate through kernels in the legacy vLLM cache directory
    structure and extract their metadata and associated files.
    """

    def __init__(self, root: Path | None = None):
        """
        Initialize the legacy vLLM cache repository.

        Args:
            root: Path to the vLLM cache directory. If None, uses ~/.cache/vllm.

        Raises:
            FileNotFoundError: If the cache directory doesn't exist.
        """
        self.root = root or (Path.home() / ".cache" / "vllm")
        if not self.root.exists():
            raise FileNotFoundError(
                f"Legacy vLLM cache directory not found: {self.root}"
            )

    def _find_torch_compile_cache_dirs(self) -> Iterable[tuple[str, Path]]:
        """
        Find torch compile cache directories in the vLLM cache root.

        Yields:
            Tuples of (vllm_hash, path_to_hash_directory)
        """
        torch_compile_cache = self.root / "torch_compile_cache"
        if not torch_compile_cache.exists():
            log.warning("No torch_compile_cache directory found in %s", self.root)
            return

        for hash_dir in torch_compile_cache.iterdir():
            if hash_dir.is_dir():
                yield hash_dir.name, hash_dir

    def _find_rank_dirs(self, hash_dir: Path) -> Iterable[tuple[str, Path]]:
        """
        Find rank directories within a legacy vLLM hash directory.

        Args:
            hash_dir: Path to the vLLM hash directory

        Yields:
            Tuples of (rank_x_y, triton_cache_path)
        """
        for rank_dir in hash_dir.iterdir():
            if rank_dir.is_dir() and rank_dir.name.startswith("rank"):
                triton_cache = rank_dir / "triton_cache"
                if triton_cache.exists():
                    yield rank_dir.name, triton_cache

    def kernels(self) -> Iterable[tuple[str, str, str, Kernel]]:
        """
        Iterate through all kernels in the legacy vLLM cache directory.

        Yields:
            Tuples of (vllm_hash, cache_root, rank_x_y, kernel)
            where each kernel contains metadata parsed from cache files.
        """
        plugins = _get_plugins()
        for vllm_hash, hash_dir in self._find_torch_compile_cache_dirs():
            for rank_x_y, triton_cache_dir in self._find_rank_dirs(hash_dir):
                for kernel in iter_triton_kernels(triton_cache_dir, plugins):
                    yield vllm_hash, str(self.root), rank_x_y, kernel


class VllmCacheRepository:  # pylint: disable=too-few-public-methods
    """
    Repository for accessing and managing new vLLM kernel cache files.

    This class provides methods to iterate through kernels in the new vLLM cache directory
    structure with artifact_compile_range directories and best config support.
    """

    def __init__(self, root: Path | None = None):
        """
        Initialize the new vLLM cache repository.

        Args:
            root: Path to the vLLM cache directory. If None, uses ~/.cache/vllm.

        Raises:
            FileNotFoundError: If the cache directory doesn't exist.
        """
        self.root = root or (Path.home() / ".cache" / "vllm")
        if not self.root.exists():
            raise FileNotFoundError(f"New vLLM cache directory not found: {self.root}")

    def _find_torch_compile_cache_dirs(self) -> Iterable[tuple[str, Path]]:
        """
        Find torch compile cache directories in the vLLM cache root.

        Yields:
            Tuples of (vllm_hash, path_to_hash_directory)
        """
        torch_compile_cache = self.root / "torch_compile_cache"
        if not torch_compile_cache.exists():
            log.warning("No torch_compile_cache directory found in %s", self.root)
            return

        for hash_dir in torch_compile_cache.iterdir():
            if hash_dir.is_dir():
                yield hash_dir.name, hash_dir

    def _find_best_config(self, processing_dir: Path) -> Optional[str]:
        """
        Find and read the best_config file in processing directory.

        Searches processing_dir and its immediate subdirectories for *.best_config files.

        Args:
            processing_dir: Path to the processing directory

        Returns:
            Content of best_config file or None if not found
        """
        # Defensive check: if a file path is passed instead of directory, skip
        if processing_dir.is_file():
            log.debug(
                "Expected directory but got file, cannot search for best_config: %s",
                processing_dir,
            )
            return None

        return self._search_best_config_in_dir(processing_dir)

    def _search_best_config_in_dir(self, search_dir: Path) -> Optional[str]:
        """
        Search for best_config file in a directory and its subdirectories.

        Args:
            search_dir: Directory to search in

        Returns:
            Content of best_config file or None if not found
        """
        # Exclude known directories that don't contain these files
        exclude_dirs = {"aotautograd", "fxgraph", "triton"}

        # Check search_dir root
        for config_path in search_dir.glob("*.best_config"):
            try:
                return config_path.read_text()
            except (OSError, IOError, PermissionError, UnicodeDecodeError) as e:
                log.debug("Could not read best config %s: %s", config_path, e)

        # Check immediate subdirectories (excluding known dirs)
        for subdir in search_dir.iterdir():
            if not subdir.is_dir() or subdir.name in exclude_dirs:
                continue
            for config_path in subdir.glob("*.best_config"):
                try:
                    return config_path.read_text()
                except (OSError, IOError, PermissionError, UnicodeDecodeError) as e:
                    log.debug("Could not read best config %s: %s", config_path, e)

        return None

    def _find_triton_dir(
        self, processing_dir: Path
    ) -> tuple[Optional[Path], Optional[str]]:
        """
        Find the triton directory within a processing directory.

        For unpacked artifacts, triton/ is directly under processing_dir.
        For extracted binary artifacts, the archive contents may include subdirectories
        that mirror the original cache structure. We search:
        1. processing_dir/triton (direct path)
        2. processing_dir/<TORCHINDUCTOR_CACHE_DIR_value>/triton - if env var is set,
           treat its value as a subdirectory name within the extracted contents
        3. processing_dir/torchinductor_$USER/triton - default inductor cache subdir name

        Note: If not found in these locations, returns (None, None). Caller should handle
        with an appropriate error message to the user.

        Args:
            processing_dir: Path to search for triton directory

        Returns:
            Tuple of (triton_dir, triton_subpath) where:
            - triton_dir: Path to triton directory if found, None otherwise
            - triton_subpath: Relative path from artifact_dir to triton's parent,
              None if triton is directly under artifact_dir
        """
        # 1. Check direct path first (unpacked artifacts)
        direct_triton = processing_dir / "triton"
        if direct_triton.exists() and direct_triton.is_dir():
            log.debug("Found triton dir at direct path: %s", direct_triton)
            return direct_triton, None

        # 2. Check TORCHINDUCTOR_CACHE_DIR if set
        # Note: If env var is absolute path, pathlib returns the absolute path directly,
        # effectively checking the system's torchinductor cache location as a fallback
        inductor_cache_dir = os.getenv("TORCHINDUCTOR_CACHE_DIR")
        if inductor_cache_dir:
            inductor_triton = processing_dir / inductor_cache_dir / "triton"
            if inductor_triton.exists() and inductor_triton.is_dir():
                log.debug(
                    "Found triton dir under TORCHINDUCTOR_CACHE_DIR: %s",
                    inductor_triton,
                )
                return inductor_triton, inductor_cache_dir

        # 3. Check default torchinductor_$USER location
        user = os.getenv("USER", "unknown")
        default_subpath = f"torchinductor_{user}"
        default_inductor_triton = processing_dir / default_subpath / "triton"
        if default_inductor_triton.exists() and default_inductor_triton.is_dir():
            log.debug(
                "Found triton dir at default inductor cache location: %s",
                default_inductor_triton,
            )
            return default_inductor_triton, default_subpath

        return None, None

    def _process_artifact_dir(
        self, artifact_dir: Path, plugins: dict
    ) -> Iterable[tuple[str, Optional[str], Optional[str], Kernel]]:
        """
        Process a single unpacked artifact directory for kernels.

        Args:
            artifact_dir: Path to the artifact directory (must contain triton/)
            plugins: Dictionary of plugins for kernel processing

        Yields:
            Tuples of (artifact_compile_range, best_config, triton_subpath, kernel)
        """
        best_config = self._find_best_config(artifact_dir)
        triton_dir, triton_subpath = self._find_triton_dir(artifact_dir)

        if triton_dir is None:
            log.debug("No triton directory in artifact '%s', skipping.", artifact_dir.name)
            return

        for sub_dir in triton_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            for kernel in iter_triton_kernels(sub_dir, plugins):
                yield artifact_dir.name, best_config, triton_subpath, kernel

    def _find_artifact_kernels(
        self, rank_dir: Path
    ) -> Iterable[tuple[str, Optional[str], Optional[str], Kernel]]:
        """
        Find kernels within artifact_compile_range directories for a rank.

        Args:
            rank_dir: Path to the rank directory

        Yields:
            Tuples of (artifact_compile_range, best_config, triton_subpath, kernel)
        """
        backbone_dir = rank_dir / "backbone"
        if not backbone_dir.exists():
            log.debug("Backbone directory does not exist: %s", backbone_dir)
            return

        plugins = _get_plugins()
        for artifact_dir in iter_artifact_compile_range_dirs(backbone_dir):
            yield from self._process_artifact_dir(artifact_dir, plugins)

    def _find_torch_aot_compile_dirs(self) -> Iterable[tuple[str, Path]]:
        """
        Find torch_aot_compile hash directories that contain triton kernels.

        Only yields directories that have inductor_cache/triton/ — incomplete
        or stale hashes are skipped so that the caller can safely treat a
        non-empty result as authoritative.

        Yields:
            Tuples of (vllm_hash, path_to_hash_directory)
        """
        aot_dir = self.root / "torch_compile_cache" / "torch_aot_compile"
        if not aot_dir.exists():
            return

        for hash_dir in aot_dir.iterdir():
            if hash_dir.is_dir() and (
                hash_dir / "inductor_cache" / "triton"
            ).is_dir():
                yield hash_dir.name, hash_dir

    def _iter_aot_kernels(
        self, hash_dir: Path, vllm_hash: str
    ) -> Iterable[tuple[str, str, str, str, Optional[str], Optional[str], Kernel]]:
        """
        Iterate kernels from a torch_aot_compile hash directory.

        The layout is ``<hash>/inductor_cache/triton/<sub>/<kernel_hash>/``.
        Rank information comes from sibling directories of inductor_cache.
        """
        inductor_cache = hash_dir / "inductor_cache"
        if not inductor_cache.is_dir():
            log.debug("No inductor_cache/ in %s, skipping", hash_dir)
            return

        # Determine rank from sibling rank_* directories
        rank_name = "rank_0_0"
        for item in hash_dir.iterdir():
            if item.is_dir() and item.name.startswith("rank"):
                rank_name = item.name
                break

        best_config = self._find_best_config(inductor_cache)
        triton_dir = inductor_cache / "triton"
        if not triton_dir.is_dir():
            log.debug("No triton/ in %s", inductor_cache)
            return

        plugins = _get_plugins()
        for sub_dir in triton_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            for kernel in iter_triton_kernels(sub_dir, plugins):
                yield (
                    vllm_hash,
                    str(self.root),
                    rank_name,
                    "inductor_cache",
                    best_config,
                    None,
                    kernel,
                )

    def _iter_torch_compile_kernels(
        self,
    ) -> Iterable[tuple[str, str, str, str, Optional[str], Optional[str], Kernel]]:
        """Iterate kernels from torch_compile_cache layout."""
        torch_compile_dirs = list(self._find_torch_compile_cache_dirs())
        log.debug("Found %d torch_compile_cache directories", len(torch_compile_dirs))

        for vllm_hash, hash_dir in torch_compile_dirs:
            rank_dirs = [
                d
                for d in hash_dir.iterdir()
                if d.is_dir() and d.name.startswith("rank")
            ]
            for rank_dir in rank_dirs:
                for (
                    artifact_compile_range,
                    best_config,
                    triton_subpath,
                    kernel,
                ) in self._find_artifact_kernels(rank_dir):
                    yield (
                        vllm_hash,
                        str(self.root),
                        rank_dir.name,
                        artifact_compile_range,
                        best_config,
                        triton_subpath,
                        kernel,
                    )

    def kernels(
        self,
    ) -> Iterable[tuple[str, str, str, str, Optional[str], Optional[str], Kernel]]:
        """
        Iterate through all kernels in the vLLM cache directory.

        Prefers torch_aot_compile (already-unpacked inductor cache) when present.
        Falls back to torch_compile_cache for older caches that lack the AOT layout.

        Yields:
            Tuples of (vllm_hash, cache_root, rank_x_y, artifact_compile_range,
                       best_config, triton_subpath, kernel)
            where each kernel contains metadata parsed from cache files.
        """
        log.info("Starting vLLM cache scan in: %s", self.root)

        aot_dirs = list(self._find_torch_aot_compile_dirs())
        if aot_dirs:
            log.debug("Found %d torch_aot_compile directories", len(aot_dirs))
            for vllm_hash, hash_dir in aot_dirs:
                yield from self._iter_aot_kernels(hash_dir, vllm_hash)
            return

        # Fallback: torch_compile_cache (older caches without torch_aot_compile)
        yield from self._iter_torch_compile_kernels()


HELION_KERNEL_PREFIX = "_helion_"


class HelionCacheRepository:  # pylint: disable=too-few-public-methods
    """Repository for accessing and managing Helion kernel cache files."""

    def __init__(self, root: Path | None = None):
        self.root = root or get_cache_dir(MODE_HELION)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Helion cache directory not found: {self.root}"
            )

    def _read_best_configs(self) -> dict[str, tuple[str, str]]:
        """Read all ``*.best_config`` files and build a lookup.

        Returns:
            Mapping of ``backend_cache_key`` to
            ``(helion_hash, raw_json_content)`` for each best_config file.
        """
        configs: dict[str, tuple[str, str]] = {}
        for path in sorted(self.root.glob("*.best_config")):
            helion_hash = path.stem
            try:
                raw = path.read_text(encoding="utf-8")
                data = json.loads(raw)
                backend_key = data.get("backend_cache_key")
                if backend_key:
                    if backend_key in configs:
                        log.warning(
                            "Duplicate backend_cache_key '%s' in %s; keeping first occurrence",
                            backend_key,
                            path,
                        )
                        continue
                    configs[backend_key] = (helion_hash, raw)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Could not read best_config %s: %s", path, exc)
        return configs

    def kernels(
        self,
    ) -> Iterable[tuple[str, str, Optional[str], Optional[str], bool, Kernel]]:
        """Iterate through Helion kernels in the cache directory.

        Only kernels whose name starts with ``_helion_`` are yielded;
        regular triton kernels that may share the same directory are skipped.

        Yields:
            Tuples of (cache_dir, triton_cache_key, helion_hash,
                       best_config, is_best, kernel)
        """
        triton_dir = resolve_helion_triton_dir(self.root)
        if triton_dir is None:
            log.warning("No triton directory found in %s", self.root)
            return

        best_configs = self._read_best_configs()
        plugins = _get_plugins()

        for sub_dir in triton_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            for kernel in iter_triton_kernels(sub_dir, plugins):
                if not kernel.name or not kernel.name.startswith(HELION_KERNEL_PREFIX):
                    continue
                match = best_configs.get(kernel.hash)
                if match:
                    helion_hash, best_config = match
                    is_best = True
                else:
                    helion_hash, best_config = None, None
                    is_best = False
                yield (
                    str(self.root),
                    kernel.hash,
                    helion_hash,
                    best_config,
                    is_best,
                    kernel,
                )
