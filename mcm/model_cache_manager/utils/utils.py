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

from model_cache_manager.utils.mcm_constants import (
    MODE_TRITON,
    MODE_VLLM,
    MODE_VLLM_LEGACY,
    ARTIFACT_COMPILE_RANGE_PREFIX,
)


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


def get_temp_extraction_dir() -> Path:
    """
    Get the base directory for temporary artifact extraction.

    Returns:
        Path to /tmp/mcm_{username}/ directory (created if it doesn't exist)
    """
    import getpass  # pylint: disable=import-outside-toplevel

    username = getpass.getuser()
    temp_base = Path("/tmp") / f"mcm_{username}"
    temp_base.mkdir(parents=True, exist_ok=True)
    return temp_base


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


def _has_vllm_legacy_cache_structure(cache_dir: Path) -> bool:
    """Check if directory has legacy vLLM cache structure."""
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

            # Check for triton_cache subdirectory (legacy structure)
            triton_cache = rank_dir / "triton_cache"
            if triton_cache.exists():
                return True
    return False


def iter_artifact_compile_range_dirs(backbone_dir: Path):
    """Iterate through artifact_compile_range directories and binary files in backbone.

    This function yields both unpacked artifacts (directories) and binary artifacts (files)
    that match the artifact_compile_range naming pattern.

    Args:
        backbone_dir: Path to the backbone directory

    Yields:
        Path objects for each artifact_compile_range directory or binary file
    """
    for item in backbone_dir.iterdir():
        # Yield both directories (unpacked) and files (binary) that match the pattern
        # Explicitly check for dir or file to exclude symlinks and other special items
        if item.name.startswith(ARTIFACT_COMPILE_RANGE_PREFIX) and (
            item.is_dir() or item.is_file()
        ):
            yield item


def _has_artifact_compile_range_with_triton(backbone_dir: Path) -> bool:
    """Check if backbone directory has artifact_compile_range subdirectories with triton."""
    for artifact_dir in iter_artifact_compile_range_dirs(backbone_dir):
        triton_dir = artifact_dir / "triton"
        if triton_dir.exists():
            return True
    return False


def _has_valid_rank_structure(hash_dir: Path) -> bool:
    """Check if hash directory has valid rank structure with new vLLM format."""
    for rank_dir in hash_dir.iterdir():
        if not (rank_dir.is_dir() and rank_dir.name.startswith("rank")):
            continue

        backbone = rank_dir / "backbone"
        if backbone.exists() and _has_artifact_compile_range_with_triton(backbone):
            return True
    return False


def _has_vllm_cache_structure(cache_dir: Path) -> bool:
    """Check if directory has new vLLM cache structure."""
    torch_compile_cache = cache_dir / "torch_compile_cache"
    if not (torch_compile_cache.exists() and torch_compile_cache.is_dir()):
        return False

    # Look for hash directories containing rank subdirectories
    for hash_dir in torch_compile_cache.iterdir():
        if hash_dir.is_dir() and _has_valid_rank_structure(hash_dir):
            return True
    return False


def detect_cache_mode(cache_dir: Path) -> str:
    """
    Auto-detect cache mode based on directory structure.

    Args:
        cache_dir: Path to the cache directory

    Returns:
        'vllm' for new vLLM structure, 'vllm-legacy' for old vLLM structure,
        'triton' otherwise
    """
    if not cache_dir.exists():
        return MODE_TRITON

    # Check for new vLLM cache structure (with backbone/artifact_compile_range_* directories)
    if _has_vllm_cache_structure(cache_dir):
        return MODE_VLLM

    # Check for legacy vLLM cache structure (with direct triton_cache)
    if _has_vllm_legacy_cache_structure(cache_dir):
        return MODE_VLLM_LEGACY

    # Check for direct triton cache structure
    # Look for triton kernel files in the directory
    for item in cache_dir.rglob("*.json"):
        # Triton kernels typically have .json metadata files
        if item.parent.name.startswith("triton_"):
            return MODE_TRITON

    return MODE_TRITON  # Default to triton mode


# Kernel operations utilities
log = logging.getLogger(__name__)


@dataclass
class KernelIdentifier:
    """Unified identifier for kernels across different modes."""

    mode: str
    hash_key: str  # "hash" for triton, "triton_cache_key" for vllm
    vllm_hash: Optional[str] = None  # Only used for vLLM mode
    rank_x_y: Optional[str] = None  # Only used for vLLM mode
    artifact_compile_range: Optional[str] = None  # Only used for new vLLM mode
    triton_subpath: Optional[str] = None  # Relative path from artifact_dir to triton's parent

    def __str__(self) -> str:
        if self.mode in (MODE_VLLM, MODE_VLLM_LEGACY):
            return f"vllm_hash={self.vllm_hash}, triton_cache_key={self.hash_key}"
        return self.hash_key

    def to_tuple(self) -> Union[str, Tuple[Optional[str], str]]:
        """Convert to the format expected by existing code."""
        if self.mode in (MODE_VLLM, MODE_VLLM_LEGACY):
            return (self.vllm_hash, self.hash_key)
        return self.hash_key


def find_vllm_legacy_kernel_dirs(
    cache_dir: Path, vllm_hash: str, triton_cache_key: str
) -> List[Path]:
    """Find kernel directories for legacy vLLM structure."""
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


def _find_kernel_dirs_in_triton(triton_dir: Path, triton_cache_key: str) -> List[Path]:
    """Find kernel directories within a triton directory."""
    kernel_dirs = []
    for sub_dir in triton_dir.iterdir():
        if sub_dir.is_dir():
            kernel_dir = sub_dir / triton_cache_key
            if kernel_dir.exists():
                kernel_dirs.append(kernel_dir)
    return kernel_dirs


def _process_specific_artifact_compile_range(
    backbone_dir: Path,
    artifact_compile_range: str,
    triton_cache_key: str,
    triton_subpath: Optional[str] = None,
) -> List[Path]:
    """Process a specific artifact_compile_range directory.

    Args:
        backbone_dir: Path to the backbone directory
        artifact_compile_range: Name of the artifact_compile_range directory
        triton_cache_key: Triton cache key to search for
        triton_subpath: Relative path from artifact_dir to triton's parent.
            None means triton is directly under artifact_dir.
    """
    artifact_dir = backbone_dir / artifact_compile_range
    if not artifact_dir.exists() or not artifact_dir.is_dir():
        return []

    # Build path to triton directory using triton_subpath if provided
    if triton_subpath:
        triton_dir = artifact_dir / triton_subpath / "triton"
    else:
        triton_dir = artifact_dir / "triton"

    if not triton_dir.exists():
        return []

    return _find_kernel_dirs_in_triton(triton_dir, triton_cache_key)


def _process_all_artifact_compile_ranges(
    backbone_dir: Path,
    triton_cache_key: str,
    triton_subpath: Optional[str] = None,
) -> List[Path]:
    """Process all artifact_compile_range directories in backbone.

    Args:
        backbone_dir: Path to the backbone directory
        triton_cache_key: Triton cache key to search for
        triton_subpath: Relative path from artifact_dir to triton's parent.
            None means triton is directly under artifact_dir.
    """
    kernel_dirs = []
    for artifact_dir in iter_artifact_compile_range_dirs(backbone_dir):
        # Build path to triton directory using triton_subpath if provided
        if triton_subpath:
            triton_dir = artifact_dir / triton_subpath / "triton"
        else:
            triton_dir = artifact_dir / "triton"

        if triton_dir.exists():
            kernel_dirs.extend(
                _find_kernel_dirs_in_triton(triton_dir, triton_cache_key)
            )
    return kernel_dirs


def _process_rank_directory(
    rank_dir: Path,
    triton_cache_key: str,
    artifact_compile_range: Optional[str],
    triton_subpath: Optional[str] = None,
) -> List[Path]:
    """Process a single rank directory.

    Args:
        rank_dir: Path to the rank directory
        triton_cache_key: Triton cache key to search for
        artifact_compile_range: Name of specific artifact_compile_range, or None for all
        triton_subpath: Relative path from artifact_dir to triton's parent.
            None means triton is directly under artifact_dir.
    """
    if not (rank_dir.is_dir() and rank_dir.name.startswith("rank")):
        return []

    backbone_dir = rank_dir / "backbone"
    if not backbone_dir.exists():
        return []

    if artifact_compile_range:
        return _process_specific_artifact_compile_range(
            backbone_dir, artifact_compile_range, triton_cache_key, triton_subpath
        )
    return _process_all_artifact_compile_ranges(
        backbone_dir, triton_cache_key, triton_subpath
    )


def find_vllm_kernel_dirs(
    cache_dir: Path,
    vllm_hash: str,
    triton_cache_key: str,
    artifact_compile_range: Optional[str] = None,
    triton_subpath: Optional[str] = None,
) -> List[Path]:
    """Find kernel directories for new vLLM structure.

    Args:
        cache_dir: Root cache directory
        vllm_hash: vLLM hash identifier
        triton_cache_key: Triton cache key to search for
        artifact_compile_range: Name of specific artifact_compile_range, or None for all
        triton_subpath: Relative path from artifact_dir to triton's parent.
            None means triton is directly under artifact_dir.
    """
    vllm_root_dir = cache_dir / "torch_compile_cache" / vllm_hash
    if not vllm_root_dir.exists():
        return []

    kernel_dirs = []
    for rank_dir in vllm_root_dir.iterdir():
        kernel_dirs.extend(
            _process_rank_directory(
                rank_dir, triton_cache_key, artifact_compile_range, triton_subpath
            )
        )
    return kernel_dirs


def get_kernel_directories(
    cache_dir: Path, mode: str, identifier: KernelIdentifier
) -> List[Path]:
    """Get list of directories containing kernel files for any mode."""
    if mode == MODE_VLLM_LEGACY:
        if identifier.vllm_hash is None:
            raise ValueError("vllm_hash cannot be None for VLLM mode")
        return find_vllm_legacy_kernel_dirs(
            cache_dir, identifier.vllm_hash, identifier.hash_key
        )
    if mode == MODE_VLLM:
        if identifier.vllm_hash is None:
            raise ValueError("vllm_hash cannot be None for VLLM mode")
        # Use artifact_compile_range and triton_subpath if available for precise lookup
        return find_vllm_kernel_dirs(
            cache_dir,
            identifier.vllm_hash,
            identifier.hash_key,
            identifier.artifact_compile_range,
            identifier.triton_subpath,
        )
    return [cache_dir / identifier.hash_key]


def delete_ir_files_from_dirs(
    kernel_dirs: List[Path], ir_extensions: set
) -> Tuple[int, List[str]]:
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
                freed += sum(p.stat().st_size for p in k_dir.rglob("*") if p.is_file())
                shutil.rmtree(k_dir)
                log.debug("Deleted kernel directory: %s", k_dir)
            except OSError as err:
                log.error("Failed to remove %s: %s", k_dir, err, exc_info=True)
    return freed


def create_kernel_identifier(mode: str, **kwargs) -> KernelIdentifier:
    """Factory function to create kernel identifiers."""
    if mode in (MODE_VLLM, MODE_VLLM_LEGACY):
        triton_cache_key = kwargs.get("triton_cache_key")
        vllm_hash = kwargs.get("vllm_hash")
        rank_x_y = kwargs.get("rank_x_y")
        to_check = [triton_cache_key, vllm_hash, rank_x_y]
        if any(not v for v in to_check):
            raise ValueError(
                f"triton_cache_key, vllm_hash and rank_x_y are required for {mode} mode"
            )
        return KernelIdentifier(
            mode=mode,
            hash_key=triton_cache_key,
            vllm_hash=vllm_hash,
            rank_x_y=rank_x_y,
        )
    hash_key = kwargs.get("hash")
    if hash_key is None:
        raise ValueError("hash is required for Triton mode")
    return KernelIdentifier(mode=mode, hash_key=hash_key)


def extract_identifiers_from_groups(
    mode: str, duplicate_groups: List[List[Dict[str, Any]]]
) -> List[KernelIdentifier]:
    """Extract kernel identifiers from duplicate groups, excluding the newest in each group."""
    identifiers = []

    for group in duplicate_groups:
        if len(group) > 1:
            # Prune all but the newest (last) kernel in each group
            for kernel_dict in group[:-1]:
                if mode == MODE_VLLM:
                    identifier = create_kernel_identifier(
                        mode=mode,
                        vllm_hash=kernel_dict["vllm_hash"],
                        triton_cache_key=kernel_dict["triton_cache_key"],
                    )
                else:
                    identifier = create_kernel_identifier(
                        mode=mode, hash=kernel_dict["hash"]
                    )
                identifiers.append(identifier)

    return identifiers


def build_common_search_filters(
    criteria, orm_class, equality_filter_configs: List[Tuple[str, Any, Any]]
) -> List:
    """Build common search filters that are shared between strategies."""
    active_filters = []

    for crit_attr, orm_column, transformer in equality_filter_configs:
        value = getattr(criteria, crit_attr, None)
        if value is not None:
            if transformer:
                value = transformer(value)
            active_filters.append(orm_column == value)

    if criteria.cache_hit_lower is not None:
        active_filters.append(orm_class.runtime_hits < criteria.cache_hit_lower)

    if criteria.cache_hit_higher is not None:
        active_filters.append(orm_class.runtime_hits > criteria.cache_hit_higher)

    if criteria.older_than_timestamp is not None:
        active_filters.append(orm_class.modified_time < criteria.older_than_timestamp)

    if criteria.younger_than_timestamp is not None:
        active_filters.append(orm_class.modified_time > criteria.younger_than_timestamp)

    return active_filters


def process_kernels_in_batches(kernels_iterator, db, batch_size: int = 1000) -> int:
    """Process kernels from iterator in batches and bulk insert them.

    Args:
        kernels_iterator: Iterator yielding kernel data tuples
        db: Database to insert kernels into
        batch_size: Number of kernels to accumulate before bulk inserting

    Returns:
        Total number of kernels inserted
    """
    kernels_batch = []
    total_inserted = 0

    for kernel_data in kernels_iterator:
        kernels_batch.append(kernel_data)

        # Insert batch when it reaches the batch size
        if len(kernels_batch) >= batch_size:
            total_inserted += db.bulk_insert_kernels(kernels_batch)
            kernels_batch = []  # Reset batch

    # Insert any remaining kernels
    if kernels_batch:
        total_inserted += db.bulk_insert_kernels(kernels_batch)

    return total_inserted
