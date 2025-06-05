"""
Data models for Triton kernels and related files.

This module defines the data structures used to represent kernels and their files.
"""

from dataclasses import dataclass
from pathlib import Path
import datetime
import logging
from typing import Mapping, Any, Optional


log = logging.getLogger(__name__)


@dataclass(slots=True)
class KernelFile:
    """
    Represents a file associated with a Triton kernel.

    Attributes:
        file_type: The type of file (e.g., 'ptx', 'cubin', 'metadata')
        path: Path to the file
        size: Size of the file in bytes
    """

    file_type: str
    path: Path
    size: int


@dataclass(slots=True)
class Kernel:
    """
    Represents a Triton kernel with its metadata and associated files.

    This class contains all attributes extracted from a kernel's metadata file.
    """

    # pylint: disable=too-many-instance-attributes
    hash: str
    cache_dir: str
    backend: str
    arch: int
    warp_size: int
    num_warps: int
    num_stages: int
    name: str
    num_ctas: int
    maxnreg: int
    cluster_dims: list[int]
    ptx_version: Optional[str]
    enable_fp_fusion: bool
    launch_cooperative_grid: bool
    supported_fp8_dtypes: list[str]
    deprecated_fp8_dtypes: list[str]
    default_dot_input_precision: str
    allowed_dot_input_precisions: list[str]
    max_num_imprecise_acc_default: int
    extern_libs: list[list[str]]
    debug: bool
    backend_name: str
    sanitize_overflow: bool
    triton_version: str
    shared: int
    tmem_size: int
    global_scratch_size: int
    global_scratch_align: int
    metadata: Mapping[str, Any]
    waves_per_eu: Optional[int]
    kpack: Optional[int]
    matrix_instr_nonkdim: Optional[int]
    files: list[KernelFile]
    modified_time: Optional[float] = None

    @property
    def modified_datetime(self) -> Optional[datetime.datetime]:
        """Return modification time as a datetime object."""
        if self.modified_time is None:
            return None
        try:
            return datetime.datetime.fromtimestamp(self.modified_time)
        except (ValueError, TypeError):
            log.debug("Value or Type error for modified_time")
            return None
