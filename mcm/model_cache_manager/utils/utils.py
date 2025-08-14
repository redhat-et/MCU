"""
Utilities.
"""

import re
import logging
import shutil
from datetime import timedelta, datetime, timezone
from typing import Optional, Tuple, List, Union, Any, Dict
from pathlib import Path
from dataclasses import dataclass
import rich
import typer


def format_size(size_bytes: int | float) -> str:
    """
    Format a file size in a human-readable way.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def parse_duration(duration_str: Optional[str]) -> Optional[timedelta]:
    """
    Parses a duration string (e.g., '7d', '2w') into a timedelta object.
    Returns None if the string is invalid or None.
    """
    if not duration_str:
        return None

    match = re.match(r"(\d+)([dw])$", duration_str.lower())
    if not match:
        rich.print(
            f"[red]Invalid duration format: '{duration_str}'. "
            f"Use 'Xd' for days or 'Xw' for weeks.[/red]"
        )
        raise typer.Exit(code=1)

    value, unit = match.groups()
    value = int(value)

    if unit == "d":
        return timedelta(days=value)
    if unit == "w":
        return timedelta(weeks=value)
    return None


def mod_time_handle(mod_time_unix) -> str:
    """
    Convert an optional UNIX timestamp into a formatted date string.

    Args:
        timestamp: An optional float representing the UNIX timestamp.

    Returns:
        A string formatted as 'YYYY-MM-DD HH:MM:SS',
        'Invalid Date' if the timestamp causes an error during conversion,
        or 'N/A' if the timestamp is None.
    """
    if mod_time_unix is not None:
        try:
            dt_obj = datetime.fromtimestamp(mod_time_unix)
            return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError, OSError):
            return "Invalid Date"
    return "N/A"


def get_older_younger(
    older_than: str | None, younger_than: str | None
) -> Tuple[float | None, float | None]:
    """
    Calculates cutoff timestamps based on "older than" and "younger than" duration strings.

    Args:
        older_than: A duration string (e.g., "7d") indicating the minimum
            age.
        younger_than: A duration string (e.g., "1d") indicating the maximum
            age.
    Returns:
        A tuple containing two float or None values:
        (older_than_timestamp, younger_than_timestamp).
    """
    older_than_timestamp: Optional[float] = None
    younger_than_timestamp: Optional[float] = None
    now = datetime.now(timezone.utc)

    try:
        if older_than:
            delta = parse_duration(older_than)
            if delta:
                older_than_timestamp = (now - delta).timestamp()
        if younger_than:
            delta = parse_duration(younger_than)
            if delta:
                younger_than_timestamp = (now - delta).timestamp()
    except Exception as exc:
        raise typer.Exit(1) from exc

    if (
        older_than_timestamp is not None
        and younger_than_timestamp is not None
        and older_than_timestamp < younger_than_timestamp
    ):
        rich.print(
            "[red]Error: --older-than timestamp cannot be more recent than"
            "--younger-than timestamp.[/red]"
        )
        raise typer.Exit(1)
    return older_than_timestamp, younger_than_timestamp


def check_hits_num(higher: int | None, lower: int | None) -> bool:
    """Check if cache hit bounds are valid (higher should not be greater than lower)."""
    if higher is not None and lower is not None:
        if higher > lower:
            return False
    return True


def _has_vllm_cache_structure(cache_dir: Path) -> bool:
    """Check if directory has vLLM cache structure."""
    torch_compile_cache = cache_dir / "torch_compile_cache"
    if not (torch_compile_cache.exists() and torch_compile_cache.is_dir()):
        return False

    # Look for hash directories containing rank subdirectories
    for hash_dir in torch_compile_cache.iterdir():
        if not hash_dir.is_dir():
            continue

        # Look for rank directories pattern rank<x>_<y>
        for rank_dir in hash_dir.iterdir():
            if not (rank_dir.is_dir() and rank_dir.name.startswith("rank")):
                continue

            # Check for triton_cache subdirectory
            triton_cache = rank_dir / "triton_cache"
            if triton_cache.exists():
                return True
    return False


def detect_cache_mode(cache_dir: Path) -> str:
    """
    Auto-detect cache mode based on directory structure.

    Args:
        cache_dir: Path to the cache directory

    Returns:
        'vllm' if vLLM cache structure detected, 'triton' otherwise
    """
    if not cache_dir.exists():
        return "triton"

    # Check for vLLM cache structure:
    # $VLLM_CACHE_ROOT/torch_compile_cache/<hash>/rank<x>_<y>/
    if _has_vllm_cache_structure(cache_dir):
        return "vllm"

    # Check for direct triton cache structure
    # Look for triton kernel files in the directory
    for item in cache_dir.rglob("*.json"):
        # Triton kernels typically have .json metadata files
        if item.parent.name.startswith("triton_"):
            return "triton"

    return "triton"  # Default to triton mode


# Kernel operations utilities
log = logging.getLogger(__name__)


@dataclass
class KernelIdentifier:
    """Unified identifier for kernels across different modes."""
    mode: str
    hash_key: str  # "hash" for triton, "triton_cache_key" for vllm
    vllm_hash: Optional[str] = None  # Only used for vLLM mode

    def __str__(self) -> str:
        if self.mode == "vllm":
            return f"vllm_hash={self.vllm_hash}, triton_cache_key={self.hash_key}"
        return self.hash_key

    def to_tuple(self) -> Union[str, Tuple[str, str]]:
        """Convert to the format expected by existing code."""
        if self.mode == "vllm":
            return (self.vllm_hash, self.hash_key)
        return self.hash_key


def find_vllm_kernel_dirs(cache_dir: Path, vllm_hash: str, triton_cache_key: str) -> List[Path]:
    """Find kernel directories for a given vLLM hash and triton cache key."""
    vllm_root_dir = cache_dir / "torch_compile_cache" / vllm_hash
    kernel_dirs = []

    if vllm_root_dir.exists():
        for rank_dir in vllm_root_dir.iterdir():
            if rank_dir.is_dir() and rank_dir.name.startswith("rank"):
                triton_cache_dir = rank_dir / "triton_cache"
                kernel_dir = triton_cache_dir / triton_cache_key
                if kernel_dir.exists():
                    kernel_dirs.append(kernel_dir)
    return kernel_dirs


def get_kernel_directories(cache_dir: Path, mode: str, identifier: KernelIdentifier) -> List[Path]:
    """Get list of directories containing kernel files for any mode."""
    if mode == "vllm":
        return find_vllm_kernel_dirs(cache_dir, identifier.vllm_hash, identifier.hash_key)
    return [cache_dir / identifier.hash_key]


def delete_ir_files_from_dirs(kernel_dirs: List[Path], ir_extensions: set) -> Tuple[int, List[str]]:
    """Delete IR files from kernel directories. Returns (bytes_freed, deleted_file_names)."""
    freed = 0
    deleted_file_names = []

    for k_dir in kernel_dirs:
        files = list(k_dir.iterdir()) if k_dir.exists() else []
        for p in files:
            if p.suffix in ir_extensions and p.is_file():
                try:
                    freed += p.stat().st_size
                    p.unlink()
                    deleted_file_names.append(p.name)
                    log.debug("Deleted IR file: %s", p)
                except OSError as err:
                    log.warning("Could not delete %s: %s", p, err)
    return freed, deleted_file_names


def delete_kernel_directories(kernel_dirs: List[Path]) -> int:
    """Delete entire kernel directories. Returns bytes freed."""
    freed = 0
    for k_dir in kernel_dirs:
        if k_dir.exists():
            try:
                freed += sum(
                    p.stat().st_size for p in k_dir.rglob("*") if p.is_file()
                )
                shutil.rmtree(k_dir)
                log.debug("Deleted kernel directory: %s", k_dir)
            except OSError as err:
                log.error("Failed to remove %s: %s", k_dir, err, exc_info=True)
    return freed


def create_kernel_identifier(mode: str, **kwargs) -> KernelIdentifier:
    """Factory function to create kernel identifiers."""
    if mode == "vllm":
        return KernelIdentifier(
            mode=mode,
            hash_key=kwargs.get('triton_cache_key'),
            vllm_hash=kwargs.get('vllm_hash')
        )
    return KernelIdentifier(
        mode=mode,
        hash_key=kwargs.get('hash')
    )


def extract_identifiers_from_groups(mode: str,
                                    duplicate_groups: List[List[Dict[str, Any]]]
                                    ) -> List[KernelIdentifier]:
    """Extract kernel identifiers from duplicate groups, excluding the newest in each group."""
    identifiers = []

    for group in duplicate_groups:
        if len(group) > 1:
            # Prune all but the newest (last) kernel in each group
            for kernel_dict in group[:-1]:
                if mode == "vllm":
                    identifier = create_kernel_identifier(
                        mode=mode,
                        vllm_hash=kernel_dict["vllm_hash"],
                        triton_cache_key=kernel_dict["triton_cache_key"]
                    )
                else:
                    identifier = create_kernel_identifier(
                        mode=mode,
                        hash=kernel_dict["hash"]
                    )
                identifiers.append(identifier)

    return identifiers
