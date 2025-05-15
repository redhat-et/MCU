"""
This module defines Pydantic models for validating Triton kernel metadata
and utilities for deserializing kernel metadata from JSON.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from pydantic import BaseModel, Field, ValidationError, model_validator

from ..models.kernel import Kernel, KernelFile
from ..plugins.base import KernelBackendPlugin


log = logging.getLogger(__name__)


class KernelTarget(BaseModel):
    """Validation model for the 'target' field of kernel metadata."""

    backend: str = ""
    arch: int | str = 0
    warp_size: int = 0


class KernelMetadata(BaseModel):
    """
    Pydantic model for validating kernel metadata.

    This model ensures that all required fields exist and have the correct types.
    """

    name: str = ""
    target: KernelTarget = Field(default_factory=KernelTarget)
    num_warps: int = 0
    num_stages: int = 0
    num_ctas: int = 0
    maxnreg: Any = 0
    cluster_dims: List[int] = Field(default_factory=list)
    ptx_version: Optional[str] = None
    enable_fp_fusion: bool = False
    launch_cooperative_grid: bool = False
    supported_fp8_dtypes: List[str] = Field(default_factory=list)
    deprecated_fp8_dtypes: List[str] = Field(default_factory=list)
    default_dot_input_precision: str = ""
    allowed_dot_input_precisions: List[str] = Field(default_factory=list)
    max_num_imprecise_acc_default: int = 0
    extern_libs: List[List[str]] = Field(default_factory=list)
    debug: bool = False
    backend_name: str = ""
    sanitize_overflow: bool = False
    triton_version: str = ""
    shared: int = 0
    tmem_size: int = 0
    global_scratch_size: int = 0
    global_scratch_align: int = 0
    waves_per_eu: Optional[int] = None
    kpack: Optional[int] = None
    matrix_instr_nonkdim: Optional[int] = None

    @model_validator(mode="after")
    def check_backend_consistency(self) -> "KernelMetadata":
        """Ensure backend_name is set if provided in target."""
        # Cast to KernelTarget to help pylint understand the type
        target = cast(KernelTarget, self.target)

        # pylint: disable=no-member
        if not self.backend_name and target.backend:
            self.backend_name = target.backend
        return self


def is_kernel_related(data: Dict[str, Any]) -> bool:
    """
    Determine if the JSON data appears to be kernel metadata.

    This function checks for the presence of kernel-specific fields or patterns
    to determine if the JSON is likely to represent a kernel.

    Args:
        data: Dictionary of JSON data to check

    Returns:
        True if the data appears to be kernel-related, False otherwise
    """
    kernel_specific_fields = [
        "name",
        "triton_version",
        "num_warps",
        "num_stages",
        "debug",
        "shared",
        "cluster_dims",
    ]

    for field in kernel_specific_fields:
        if field not in data:
            return False

    if "target" in data and isinstance(data["target"], dict):
        if "backend" not in data["target"]:
            return False

    return True


def deserialize_kernel(
    data: Dict[str, Any],
    hash_value: str,
    directory: Path,
    plugins: Dict[str, KernelBackendPlugin],
) -> Optional[Kernel]:
    """
    Deserialize kernel metadata from JSON into a Kernel object.

    Args:
        data: Dictionary containing kernel metadata
        hash_value: The hash identifier for the kernel
        directory: The directory containing the kernel files
        plugins: Dictionary mapping backend names to plugin instances

    Returns:
        A Kernel object if valid, None if invalid or not kernel-related
    """
    try:
        if not is_kernel_related(data):
            log.warning(
                "JSON data for '%s' does not appear to be kernel metadata", hash_value
            )
            return None

        # Validate with Pydantic
        metadata = KernelMetadata.model_validate(data)

        mod_time: Optional[float] = None
        try:
            mod_time = directory.stat().st_mtime
        except OSError as e:
            log.warning("Could not get stat for directory '%s': %s", directory, e)

        target = cast(KernelTarget, metadata.target)

        # pylint: disable=no-member
        kernel = Kernel(
            hash=hash_value,
            backend=target.backend,
            arch=target.arch,
            warp_size=target.warp_size,
            num_warps=metadata.num_warps,
            num_stages=metadata.num_stages,
            name=metadata.name,
            num_ctas=metadata.num_ctas,
            maxnreg=metadata.maxnreg,
            cluster_dims=metadata.cluster_dims,
            ptx_version=metadata.ptx_version,
            enable_fp_fusion=metadata.enable_fp_fusion,
            launch_cooperative_grid=metadata.launch_cooperative_grid,
            supported_fp8_dtypes=metadata.supported_fp8_dtypes,
            deprecated_fp8_dtypes=metadata.deprecated_fp8_dtypes,
            default_dot_input_precision=metadata.default_dot_input_precision,
            allowed_dot_input_precisions=metadata.allowed_dot_input_precisions,
            max_num_imprecise_acc_default=metadata.max_num_imprecise_acc_default,
            extern_libs=metadata.extern_libs,
            debug=metadata.debug,
            backend_name=metadata.backend_name,
            sanitize_overflow=metadata.sanitize_overflow,
            triton_version=metadata.triton_version,
            shared=metadata.shared,
            tmem_size=metadata.tmem_size,
            global_scratch_size=metadata.global_scratch_size,
            global_scratch_align=metadata.global_scratch_align,
            waves_per_eu=metadata.waves_per_eu,
            kpack=metadata.kpack,
            matrix_instr_nonkdim=metadata.matrix_instr_nonkdim,
            metadata=data,
            files=[],
            modified_time=mod_time,
        )

        common_extensions = {
            ".json": "metadata",
            ".ttir": "ttir",
            ".ttgir": "ttgir",
            ".llir": "llir",
        }

        plugin = None
        if target.backend:  # pylint: disable=no-member
            plugin = plugins.get(target.backend)  # pylint: disable=no-member

        for f in directory.iterdir():
            ft = None
            if f.suffix in common_extensions:
                ft = common_extensions[f.suffix]
            elif plugin and f.suffix in plugin.relevant_extensions():
                ft = plugin.relevant_extensions()[f.suffix]
            if ft:
                try:
                    file_stat = f.stat()
                    kernel.files.append(KernelFile(ft, f, file_stat.st_size))
                except OSError as e:
                    log.warning("Could not get stat for file '%s': %s", f, e)
                    kernel.files.append(KernelFile(ft, f, 0))

        return kernel

    except ValidationError as e:
        log.error("Invalid kernel metadata for '%s': %s", hash_value, e)
        return None
    except (TypeError, KeyError, AttributeError) as e:
        log.error("Error processing kernel '%s': %s", hash_value, e)
        return None
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Keep the broad exception as a fallback, but disable the warning
        log.error("Unexpected error processing kernel '%s': %s", hash_value, e)
        return None
