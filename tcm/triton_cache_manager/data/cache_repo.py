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
from ..models.kernel import Kernel, KernelFile
from ..plugins.discovery import discover_plugins

_COMMON = {".json": "metadata", ".ttir": "ttir", ".ttgir": "ttgir", ".llir": "llir"}

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
            Iterable of Kernel objects with metadata parsed from cache files.
        """

        for d in self._dirs():
            meta = next(d.glob("*.json"), None)
            if not meta:
                continue
            try:
                m = json.loads(meta.read_text())
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
            tgt = m.get("target", {})
            backend = tgt.get("backend", "")
            arch = tgt.get("arch", "")
            warp_size = tgt.get("warp_size", 0)
            k = Kernel(
                hash=d.name,
                backend=backend,
                arch=arch,
                warp_size=warp_size,
                num_warps=m.get("num_warps", 0),
                num_stages=m.get("num_stages", 0),
                name=m.get("name", ""),
                num_ctas=m.get("num_ctas", 0),
                maxnreg=m.get("maxnreg", 0),
                cluster_dims=m.get("cluster_dims", []),
                ptx_version=m.get("ptx_version", None),
                enable_fp_fusion=m.get("enable_fp_fusion", False),
                launch_cooperative_grid=m.get("launch_cooperative_grid", False),
                supported_fp8_dtypes=m.get("supported_fp8_dtypes", []),
                deprecated_fp8_dtypes=m.get("deprecated_fp8_dtypes", []),
                default_dot_input_precision=m.get("default_dot_input_precision", ""),
                allowed_dot_input_precisions=m.get("allowed_dot_input_precisions", []),
                max_num_imprecise_acc_default=m.get("max_num_imprecise_acc_default", 0),
                extern_libs=m.get("extern_libs", []),
                debug=m.get("debug", False),
                backend_name=m.get("backend_name", ""),
                sanitize_overflow=m.get("sanitize_overflow", False),
                triton_version=m.get("triton_version", ""),
                shared=m.get("shared", 0),
                tmem_size=m.get("tmem_size", 0),
                global_scratch_size=m.get("global_scratch_size", 0),
                global_scratch_align=m.get("global_scratch_align", 0),
                waves_per_eu=m.get("waves_per_eu", None),
                kpack=m.get("kpack", None),
                matrix_instr_nonkdim=m.get("matrix_instr_nonkdim", None),
                metadata=m,
                files=[],
            )
            plugin = self.plugins.get(backend)
            for f in d.iterdir():
                ft = None
                if f.suffix in _COMMON:
                    ft = _COMMON[f.suffix]
                elif plugin and f.suffix in plugin.relevant_extensions():
                    ft = plugin.relevant_extensions()[f.suffix]
                if ft:
                    k.files.append(KernelFile(ft, f, f.stat().st_size))
            yield k
