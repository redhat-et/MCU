"""
Base plugin interface for Triton kernel backends.

This module defines the protocol that backend plugins must implement.
"""

from typing import Protocol, Dict, List


class KernelBackendPlugin(Protocol):
    """
    Protocol for Triton kernel backend plugins.

    Backend plugins are responsible for recognizing and handling
    backend-specific file types in the kernel cache.
    """

    backend: str

    def relevant_extensions(self) -> Dict[str, str]:
        """
        Get the file extensions relevant to this backend.

        Returns:
            Dictionary mapping file extensions to file type names.
        """
        ...

def discover_plugins() -> List[KernelBackendPlugin]:
    """
    Discover and instantiate all available backend plugins.
    
    Returns:
        List of instantiated backend plugin objects.
    """
    # Import here to avoid circular dependencies
    from .cuda import CudaPlugin  # pylint: disable=import-outside-toplevel
    from .rocm import RocmPlugin  # pylint: disable=import-outside-toplevel

    return [CudaPlugin(), RocmPlugin()]
