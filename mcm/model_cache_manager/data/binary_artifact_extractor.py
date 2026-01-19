"""
Binary artifact extractor for vLLM Inductor CompiledArtifacts.

This module provides utilities for detecting and extracting binary vLLM cache artifacts
(Inductor CompiledArtifacts) into unpacked directory form for processing by MCM.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class BinaryArtifactExtractionError(Exception):
    """Exception raised when binary artifact extraction fails."""


def check_pytorch_support() -> tuple[bool, str]:
    """
    Check if PyTorch version supports binary artifact extraction.

    Returns:
        Tuple of (supported, error_message). If supported is True, error_message is empty.
    """
    try:
        # pylint: disable=import-outside-toplevel
        import torch

        if not hasattr(torch, "compiler") or not hasattr(
            torch.compiler, "load_cache_artifacts"
        ):
            return (
                False,
                "PyTorch does not expose torch.compiler.load_cache_artifacts().",
            )

        # pylint: disable=protected-access
        if not hasattr(torch, "_inductor") or not hasattr(
            torch._inductor, "CompiledArtifact"
        ):
            return (
                False,
                "PyTorch does not expose torch._inductor.CompiledArtifact.",
            )

        return True, ""
    except ImportError:
        return False, "PyTorch is not installed."


def _find_binary_file_in_dir(artifact_dir: Path) -> Optional[Path]:
    """
    Find a binary artifact file in the given directory.

    Args:
        artifact_dir: Path to search for binary files

    Returns:
        Path to binary file if found, None otherwise
    """
    # If artifact_dir itself is a file, return it
    if artifact_dir.is_file():
        return artifact_dir

    # Look for a file with no extension (typical for binary artifacts)
    if artifact_dir.is_dir():
        for item in artifact_dir.iterdir():
            if item.is_file() and item.suffix == "":
                return item

    return None


def is_binary_artifact(artifact_dir: Path) -> bool:
    """
    Check if an artifact directory contains a binary artifact.

    Binary artifacts are detected by:
    1. No 'triton/' subdirectory exists
    2. The artifact_dir itself is a file (not a directory)
       OR contains a file matching the artifact name

    Args:
        artifact_dir: Path to the artifact directory

    Returns:
        True if binary artifact, False if unpacked format
    """
    # Check if there's a triton subdirectory (unpacked format)
    if artifact_dir.is_dir():
        triton_dir = artifact_dir / "triton"
        if triton_dir.exists():
            return False

    # Check if there's a binary file
    if not artifact_dir.exists():
        return False

    return _find_binary_file_in_dir(artifact_dir) is not None


def extract_artifact_bytes_via_hook(input_file: Path) -> bytes:
    """
    Load the CompiledArtifact while intercepting the internal call to
    torch.compiler.load_cache_artifacts(artifact_bytes) to capture
    the serialized blob.

    Args:
        input_file: Path to the binary artifact file

    Returns:
        Captured artifact bytes

    Raises:
        BinaryArtifactExtractionError: If extraction fails
    """
    # Check PyTorch support using the centralized function
    supported, msg = check_pytorch_support()
    if not supported:
        raise BinaryArtifactExtractionError(msg)

    # pylint: disable=import-outside-toplevel
    import torch

    captured: dict[str, bytes] = {}
    orig = torch.compiler.load_cache_artifacts

    def hooked_load_cache_artifacts(artifact_bytes: bytes, *args, **kwargs):
        # Save the blob; CompiledArtifact.load() will call this
        captured["artifact_bytes"] = artifact_bytes
        return orig(artifact_bytes, *args, **kwargs)

    # Monkeypatch
    torch.compiler.load_cache_artifacts = hooked_load_cache_artifacts  # type: ignore[assignment]
    try:
        # This will parse the binary file and (normally) load cache artifacts.
        # We let it proceed because load() may depend on artifacts being loaded.
        # pylint: disable=protected-access
        _ = torch._inductor.CompiledArtifact.load(path=str(input_file), format="binary")
    finally:
        torch.compiler.load_cache_artifacts = orig  # type: ignore[assignment]

    if "artifact_bytes" not in captured:
        raise BinaryArtifactExtractionError(
            "Could not capture artifact_bytes. This may be an AOTCompiledArtifact "
            "(which does not use torch.compiler.load_cache_artifacts), or the file "
            "is not a CacheCompiledArtifact binary."
        )

    return captured["artifact_bytes"]


def unpack_binary_artifact_to_dir(artifact_bytes: bytes, output_dir: Path) -> None:
    """
    Unpack binary artifact bytes to a directory.

    Args:
        artifact_bytes: Binary artifact bytes captured from CompiledArtifact
        output_dir: Directory to extract artifacts into

    Raises:
        BinaryArtifactExtractionError: If unpacking fails
    """
    try:
        # pylint: disable=import-outside-toplevel
        import torch
        from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir
    except ImportError as e:
        raise BinaryArtifactExtractionError(
            "PyTorch or required modules not available"
        ) from e

    # This context redirects cache dirs so load_cache_artifacts "unpacks" into output_dir
    try:
        with temporary_cache_dir(str(output_dir)):
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)
            if cache_info is None:
                raise BinaryArtifactExtractionError(
                    "torch.compiler.load_cache_artifacts returned None (unexpected)."
                )
    except Exception as e:
        if isinstance(e, BinaryArtifactExtractionError):
            raise
        raise BinaryArtifactExtractionError(f"Unpacking failed: {e}") from e


class TemporaryExtractedArtifact:
    """
    Context manager for temporarily extracting binary artifacts.

    Automatically detects if an artifact is in binary format and extracts it
    to a temporary directory. Returns the path to process (either the original
    unpacked path or the temporary extracted path). Cleans up temp files on exit.

    Usage:
        with TemporaryExtractedArtifact(artifact_dir, vllm_hash, rank_name) as processing_dir:
            # processing_dir is either artifact_dir (if unpacked) or temp dir (if binary)
            triton_dir = processing_dir / "triton"
            # ... process kernels ...
        # Temp files are automatically cleaned up here
    """

    def __init__(self, artifact_dir: Path, vllm_hash: str, rank_name: str):
        """
        Initialize the context manager.

        Args:
            artifact_dir: Path to the artifact directory to process
            vllm_hash: vLLM hash identifier (for temp dir naming)
            rank_name: Rank name (e.g., 'rank_0_0') (for temp dir naming)
        """
        self.artifact_dir = artifact_dir
        self.vllm_hash = vllm_hash
        self.rank_name = rank_name
        self._temp_dir: Optional[Path] = None
        self._is_binary = False

    def __enter__(self) -> Path:
        """
        Enter context: detect and extract if needed.

        Returns:
            Path to process (either original or temp extracted path)

        Raises:
            BinaryArtifactExtractionError: If extraction fails
        """
        # Check if it's a binary artifact
        self._is_binary = is_binary_artifact(self.artifact_dir)

        if not self._is_binary:
            # Unpacked format, return original path
            log.debug("Artifact %s is already unpacked", self.artifact_dir.name)
            return self.artifact_dir

        # Binary format - need to extract
        log.debug("Detected binary artifact: %s", self.artifact_dir.name)

        # Check PyTorch support
        supported, msg = check_pytorch_support()
        if not supported:
            log.warning(
                "Binary artifact detected but PyTorch doesn't support extraction: %s. "
                "Treating as unpacked (will likely fail).",
                msg,
            )
            return self.artifact_dir

        # Find the binary artifact file
        artifact_file = self._find_binary_file()
        if artifact_file is None:
            raise BinaryArtifactExtractionError(
                f"Could not find binary artifact file in {self.artifact_dir}"
            )

        # Create temp directory
        try:
            # pylint: disable=import-outside-toplevel
            from ..utils.utils import get_temp_extraction_dir

            temp_base = get_temp_extraction_dir()
            # Create unique temp dir with random suffix
            temp_dir = tempfile.mkdtemp(
                prefix=f"vllm_{self.vllm_hash}_{self.rank_name}_{self.artifact_dir.name}_",
                dir=str(temp_base),
            )
            self._temp_dir = Path(temp_dir)
            log.debug("Created temp extraction dir: %s", self._temp_dir)
        except Exception as e:
            raise BinaryArtifactExtractionError(
                f"Failed to create temp directory: {e}"
            ) from e

        # Extract artifact
        try:
            log.info("Extracting binary artifact %s to %s", artifact_file.name, self._temp_dir)
            artifact_bytes = extract_artifact_bytes_via_hook(artifact_file)
            unpack_binary_artifact_to_dir(artifact_bytes, self._temp_dir)
            log.debug("Extraction complete for %s", artifact_file.name)
            return self._temp_dir
        except Exception as e:
            # Clean up temp dir on failure
            if self._temp_dir and self._temp_dir.exists():
                try:
                    shutil.rmtree(self._temp_dir, ignore_errors=True)
                # pylint: disable=broad-exception-caught
                except Exception as cleanup_err:
                    log.warning(
                        "Failed to clean up temp dir %s after extraction failure: %s",
                        self._temp_dir,
                        cleanup_err,
                    )
            self._temp_dir = None
            raise BinaryArtifactExtractionError(
                f"Failed to extract binary artifact: {e}"
            ) from e

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """
        Exit context: clean up temp directory if created.
        """
        if self._temp_dir and self._temp_dir.exists():
            try:
                log.debug("Cleaning up temp extraction dir: %s", self._temp_dir)
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            # pylint: disable=broad-exception-caught
            except Exception as e:
                # Don't raise - cleanup failure shouldn't break indexing
                log.warning(
                    "Failed to clean up temp extraction dir %s: %s", self._temp_dir, e
                )
        return False  # Don't suppress exceptions

    def _find_binary_file(self) -> Optional[Path]:
        """
        Find the binary artifact file in the artifact directory.

        Returns:
            Path to binary file, or None if not found
        """
        return _find_binary_file_in_dir(self.artifact_dir)
