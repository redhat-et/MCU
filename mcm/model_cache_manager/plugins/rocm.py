"""
ROCm backend plugin for Triton kernel cache.

This module provides support for ROCm-specific kernel files.
"""

from .base import KernelBackendPlugin


class RocmPlugin(KernelBackendPlugin):
    # pylint: disable=too-few-public-methods
    """
    Plugin for ROCm backend support.

    Handles ROCm-specific file types like AMDGCN and HSACO.
    """

    backend = "hip"

    def relevant_extensions(self):
        """
        Get file extensions relevant to ROCm backend.

        Returns:
            Dictionary mapping file extensions to file type names.
        """
        return {".amdgcn": "amdgcn", ".hsaco": "hsaco"}
