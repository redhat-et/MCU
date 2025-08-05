"""
Repository module for accessing and managing kernel cache files.

This module provides functionality to scan and parse the model cache directory.
"""

from __future__ import annotations
from pathlib import Path
import json
import logging
from typing import Iterable, Optional
from ..utils.paths import get_cache_dir
from ..models.kernel import Kernel
from ..plugins.discovery import discover_plugins
from .kernel_validator import deserialize_kernel

log = logging.getLogger(__name__)

# Lazy-loaded plugins to avoid import-time discovery failures
_PLUGINS = None


def _get_plugins():
    """Get plugins dictionary, loading them lazily on first access."""
    global _PLUGINS
    if _PLUGINS is None:
        _PLUGINS = {p.backend: p for p in discover_plugins()}
    return _PLUGINS


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

        # Try same-dir derivation first (robust across host/container moves)
        p = Path(path_str)
        if p.parent.name and p.name:
            derived = cache_root / p.parent.name / p.name
            if derived.is_file():
                return derived

        # Try path as given in group file
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

        kernel = deserialize_kernel(data, kernel_dir.name, str(cache_root), kernel_dir, plugins)
        if kernel:
            yield kernel
        else:
            log.warning("Skipping invalid kernel at '%s' (meta '%s')", kernel_dir, meta_path)


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


class VllmCacheRepository:
    """
    Repository for accessing and managing vLLM kernel cache files.
    
    This class provides methods to iterate through kernels in the vLLM cache directory
    structure and extract their metadata and associated files.
    """

    def __init__(self, root: Path | None = None):
        """
        Initialize the vLLM cache repository.

        Args:
            root: Path to the vLLM cache directory. If None, uses ~/.cache/vllm.

        Raises:
            FileNotFoundError: If the cache directory doesn't exist.
        """
        self.root = root or (Path.home() / ".cache" / "vllm")
        if not self.root.exists():
            raise FileNotFoundError(f"vLLM cache directory not found: {self.root}")

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

    def _find_rank_dirs(self, hash_dir: Path) -> Iterable[Path]:
        """
        Find rank directories within a vLLM hash directory.
        
        Args:
            hash_dir: Path to the vLLM hash directory
            
        Yields:
            Paths to rank directories (rank<x>_<y>)
        """
        for rank_dir in hash_dir.iterdir():
            if rank_dir.is_dir() and rank_dir.name.startswith("rank"):
                triton_cache = rank_dir / "triton_cache"
                if triton_cache.exists():
                    yield triton_cache

    def kernels(self) -> Iterable[tuple[str, str, Kernel]]:
        """
        Iterate through all kernels in the vLLM cache directory.

        Yields:
            Tuples of (vllm_hash, cache_root, kernel)
            where each kernel contains metadata parsed from cache files.
        """
        plugins = _get_plugins()
        for vllm_hash, hash_dir in self._find_torch_compile_cache_dirs():
            for triton_cache_dir in self._find_rank_dirs(hash_dir):
                for kernel in iter_triton_kernels(triton_cache_dir, plugins):
                    yield vllm_hash, str(self.root), kernel
