from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Any, Optional


@dataclass(slots=True)
class KernelFile:
    file_type: str
    path: Path
    size: int


@dataclass(slots=True)
class Kernel:
    hash: str
    backend: str
    arch: str
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
