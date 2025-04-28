"""
CUDA backend plugin for Triton kernel cache.

This module provides support for CUDA-specific kernel files.
"""

from .base import KernelBackendPlugin


class CudaPlugin(KernelBackendPlugin):
    """
    Plugin for CUDA backend support.

    Handles CUDA-specific file types like PTX and cubin.
    """

    backend = "cuda"

    def relevant_extensions(self):
        """
        Get file extensions relevant to CUDA backend.

        Returns:
            Dictionary mapping file extensions to file type names.
        """
        return {".ptx": "ptx", ".cubin": "cubin"}
