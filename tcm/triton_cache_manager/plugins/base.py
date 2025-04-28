"""
Base plugin interface for Triton kernel backends.

This module defines the protocol that backend plugins must implement.
"""

from typing import Protocol, Dict


class KernelBackendPlugin(Protocol):
    # pylint: disable=too-few-public-methods
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
        ...  # pylint: disable=unnecessary-ellipsis
