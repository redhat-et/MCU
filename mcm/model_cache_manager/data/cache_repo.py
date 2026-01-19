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
from ..utils.utils import iter_artifact_compile_range_dirs
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

    def _find_best_config(self, artifact_dir: Path) -> Optional[str]:
        """
        Find and read the best_config file in artifact directory.

        For unpacked artifacts, best_config is directly under artifact_dir or subdirs.
        For extracted binary artifacts, best_config may be under:
        1. artifact_dir (direct or subdirs)
        2. artifact_dir/$TORCHINDUCTOR_CACHE_DIR/... (if env var set)
        3. artifact_dir/torchinductor_$USER/... (default inductor cache location)

        Args:
            artifact_dir: Path to the artifact directory (or binary file)

        Returns:
            Content of best_config file or None if not found
        """
        # Defensive check: if a file path is passed instead of directory, skip
        if artifact_dir.is_file():
            log.debug(
                "Expected directory but got file, cannot search for best_config: %s",
                artifact_dir,
            )
            return None

        # Try to find best_config in the given directory
        result = self._search_best_config_in_dir(artifact_dir)
        if result:
            return result

        # Check TORCHINDUCTOR_CACHE_DIR if set
        inductor_cache_dir = os.getenv("TORCHINDUCTOR_CACHE_DIR")
        if inductor_cache_dir:
            inductor_path = artifact_dir / inductor_cache_dir
            if inductor_path.exists():
                result = self._search_best_config_in_dir(inductor_path)
                if result:
                    return result

        # Check default torchinductor_$USER location
        user = os.getenv("USER", "unknown")
        default_inductor_path = artifact_dir / f"torchinductor_{user}"
        if default_inductor_path.exists():
            result = self._search_best_config_in_dir(default_inductor_path)
            if result:
                return result

        return None

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
        For extracted binary artifacts, triton/ may be under:
        1. processing_dir/triton (direct path)
        2. processing_dir/$TORCHINDUCTOR_CACHE_DIR/triton (if env var set)
        3. processing_dir/torchinductor_$USER/triton (default inductor cache location)

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
        if direct_triton.exists():
            log.debug("Found triton dir at direct path: %s", direct_triton)
            return direct_triton, None

        # 2. Check TORCHINDUCTOR_CACHE_DIR if set
        inductor_cache_dir = os.getenv("TORCHINDUCTOR_CACHE_DIR")
        if inductor_cache_dir:
            # The env var value could be absolute or relative, but in extracted
            # artifacts it will be a subdirectory name under processing_dir
            inductor_triton = processing_dir / inductor_cache_dir / "triton"
            if inductor_triton.exists():
                log.debug(
                    "Found triton dir under TORCHINDUCTOR_CACHE_DIR: %s",
                    inductor_triton,
                )
                return inductor_triton, inductor_cache_dir

        # 3. Check default torchinductor_$USER location
        user = os.getenv("USER", "unknown")
        default_subpath = f"torchinductor_{user}"
        default_inductor_triton = processing_dir / default_subpath / "triton"
        if default_inductor_triton.exists():
            log.debug(
                "Found triton dir at default inductor cache location: %s",
                default_inductor_triton,
            )
            return default_inductor_triton, default_subpath

        return None, None

    # pylint: disable=too-many-locals
    def _process_artifact_dir(
        self, artifact_dir: Path, plugins: dict, vllm_hash: str, rank_name: str
    ) -> Iterable[tuple[str, Optional[str], Optional[str], Kernel]]:
        """
        Process a single artifact directory for kernels.

        Handles both unpacked (triton/ subdirectory) and binary (single file) artifacts.
        Binary artifacts are temporarily extracted to /tmp for processing.

        Args:
            artifact_dir: Path to the artifact directory
            plugins: Dictionary of plugins for kernel processing
            vllm_hash: vLLM hash identifier (for temp dir naming)
            rank_name: Rank name (e.g., 'rank_0_0') (for temp dir naming)

        Yields:
            Tuples of (artifact_compile_range, best_config, triton_subpath, kernel)
        """
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from .binary_artifact_extractor import (
            TemporaryExtractedArtifact,
            BinaryArtifactExtractionError,
        )

        log.debug(
            "Processing artifact directory: %s (vllm_hash=%s, rank=%s)",
            artifact_dir,
            vllm_hash,
            rank_name,
        )

        try:
            # Context manager handles detection and extraction if needed
            with TemporaryExtractedArtifact(
                artifact_dir, vllm_hash, rank_name
            ) as processing_dir:
                log.debug(
                    "Processing dir resolved to: %s (original: %s)",
                    processing_dir,
                    artifact_dir,
                )
                # Find best_config in processing_dir - for binary artifacts, best_config
                # is embedded inside and only available after extraction to processing_dir
                best_config = self._find_best_config(processing_dir)

                # processing_dir is either:
                # - Original artifact_dir (if unpacked)
                # - Temp extracted dir (if binary)
                triton_dir, triton_subpath = self._find_triton_dir(processing_dir)

                if triton_dir is None:
                    user = os.getenv("USER", "unknown")
                    inductor_cache_dir = os.getenv("TORCHINDUCTOR_CACHE_DIR")
                    searched_locations = [
                        f"  - {processing_dir}/triton",
                        f"  - {processing_dir}/torchinductor_{user}/triton",
                    ]
                    if inductor_cache_dir:
                        searched_locations.insert(
                            1, f"  - {processing_dir}/{inductor_cache_dir}/triton"
                        )
                    log.error(
                        "No triton directory found in artifact '%s'.\n"
                        "Searched locations:\n%s\n"
                        "Ensure the artifact was created with a compatible "
                        "vLLM/PyTorch version and the TORCHINDUCTOR_CACHE_DIR "
                        "environment variable matches the one used during creation.",
                        artifact_dir.name,
                        "\n".join(searched_locations),
                    )
                    return

                for sub_dir in triton_dir.iterdir():
                    if not sub_dir.is_dir():
                        continue
                    for kernel in iter_triton_kernels(sub_dir, plugins):
                        # Yield artifact_dir.name (original), not processing_dir
                        yield artifact_dir.name, best_config, triton_subpath, kernel

        except BinaryArtifactExtractionError as e:
            log.warning(
                "Failed to extract binary artifact %s: %s. Skipping.",
                artifact_dir.name,
                e,
            )
            return
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error(
                "Unexpected error processing artifact %s: %s. Skipping.",
                artifact_dir.name,
                e,
                exc_info=True,
            )
            return

    def _find_artifact_kernels(
        self, rank_dir: Path, rank_name: str, vllm_hash: str
    ) -> Iterable[tuple[str, Optional[str], Optional[str], Kernel]]:
        """
        Find kernels within artifact_compile_range directories for a rank.

        Args:
            rank_dir: Path to the rank directory
            rank_name: Name of the rank directory (e.g., 'rank_0_0')
            vllm_hash: vLLM hash for this cache group

        Yields:
            Tuples of (artifact_compile_range, best_config, triton_subpath, kernel)
        """
        backbone_dir = rank_dir / "backbone"
        if not backbone_dir.exists():
            log.debug("Backbone directory does not exist: %s", backbone_dir)
            return

        log.debug("Finding artifacts in backbone: %s", backbone_dir)
        plugins = _get_plugins()
        artifact_count = 0
        for artifact_dir in iter_artifact_compile_range_dirs(backbone_dir):
            artifact_count += 1
            log.debug(
                "Found artifact %d: %s (is_file=%s)",
                artifact_count,
                artifact_dir,
                artifact_dir.is_file(),
            )
            # Pass vllm_hash and rank_name to support binary extraction
            yield from self._process_artifact_dir(
                artifact_dir, plugins, vllm_hash, rank_name
            )
        log.debug("Total artifacts found in %s: %d", backbone_dir, artifact_count)

    def kernels(
        self,
    ) -> Iterable[tuple[str, str, str, str, Optional[str], Optional[str], Kernel]]:
        """
        Iterate through all kernels in the new vLLM cache directory.

        Handles both unpacked and binary artifacts transparently.

        Yields:
            Tuples of (vllm_hash, cache_root, rank_x_y, artifact_compile_range,
                       best_config, triton_subpath, kernel)
            where each kernel contains metadata parsed from cache files.
        """
        log.info("Starting vLLM cache scan in: %s", self.root)
        torch_compile_dirs = list(self._find_torch_compile_cache_dirs())
        log.info("Found %d torch_compile_cache directories", len(torch_compile_dirs))

        for vllm_hash, hash_dir in torch_compile_dirs:
            log.info("Processing vllm_hash=%s at %s", vllm_hash, hash_dir)
            rank_dirs = [
                d
                for d in hash_dir.iterdir()
                if d.is_dir() and d.name.startswith("rank")
            ]
            log.info(
                "Found %d rank directories for vllm_hash=%s", len(rank_dirs), vllm_hash
            )

            for rank_dir in rank_dirs:
                log.info("Processing rank directory: %s", rank_dir.name)
                # Pass vllm_hash to support binary artifact extraction
                kernel_count = 0
                for (
                    artifact_compile_range,
                    best_config,
                    triton_subpath,
                    kernel,
                ) in self._find_artifact_kernels(rank_dir, rank_dir.name, vllm_hash):
                    kernel_count += 1
                    yield (
                        vllm_hash,
                        str(self.root),
                        rank_dir.name,
                        artifact_compile_range,
                        best_config,
                        triton_subpath,
                        kernel,
                    )
                log.info("Yielded %d kernels from rank %s", kernel_count, rank_dir.name)
