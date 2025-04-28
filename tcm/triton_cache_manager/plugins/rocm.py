"""
ROCm backend plugin for Triton kernel cache.

This module provides support for ROCm-specific kernel files.
"""

from .base import KernelBackendPlugin


class RocmPlugin(KernelBackendPlugin):
    """
    Plugin for ROCm backend support.

    Handles ROCm-specific file types like AMDGCN and HSACO.
    """

    backend = "rocm"

    def relevant_extensions(self):
        """
        Get file extensions relevant to ROCm backend.

        Returns:
            Dictionary mapping file extensions to file type names.
        """
        return {".amdgcn": "amdgcn", ".hsaco": "hsaco"}
