"""
Plugin discovery for Triton kernel backend plugins.

This module handles discovering and loading backend plugins.
"""

from typing import List

from .base import KernelBackendPlugin
from .cuda import CudaPlugin
from .rocm import RocmPlugin


def discover_plugins() -> List[KernelBackendPlugin]:
    """
    Discover and instantiate all available backend plugins.

    Returns:
        List of instantiated backend plugin objects.
    """
    return [CudaPlugin(), RocmPlugin()]
