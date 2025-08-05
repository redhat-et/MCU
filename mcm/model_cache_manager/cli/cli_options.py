"""
Shared CLI options for the Model Cache Manager.

This module centralizes common Typer options to reduce duplication across commands.
"""

from pathlib import Path
from typing import Optional
import typer


def get_common_search_options():
    """Return dictionary of common search options for Typer commands."""
    return {
        "name": typer.Option(
            None, "--name", "-n", help="Filter by kernel name (exact match)."
        ),
        "backend": typer.Option(
            None, "--backend", "-b", help="Filter by backend (e.g., 'cuda', 'rocm')."
        ),
        "arch": typer.Option(
            None, "--arch", "-a", help="Filter by architecture (e.g., '120', 'gfx90a')."
        ),
        "older_than": typer.Option(
            None,
            "--older-than",
            help="Show kernels older than specified duration (e.g., '7d', '2w').",
        ),
        "younger_than": typer.Option(
            None,
            "--younger-than",
            help="Show kernels younger than specified duration (e.g., '14d', '1w').",
        ),
        "cache_hit_lower": typer.Option(
            None,
            "--cache-hit-lower",
            help="Show kernels with cache hits lower than "
            "specified number (e.g., '1', '10').",
        ),
        "cache_hit_higher": typer.Option(
            None,
            "--cache-hit-higher",
            help="Show kernels with cache hits higher than "
            "specified number (e.g., '1', '10').",
        ),
        "cache_dir": typer.Option(
            None,
            help="Specify the Model cache directory. Uses default if not provided.",
        ),
        "mode": typer.Option(
            None,
            "--mode",
            help="Cache mode: 'triton' for standard Triton cache, "
            "'vllm' for vLLM cache structure. Auto-detected if not specified.",
        ),
    }


CommonSearchParams = {
    "name": Optional[str],
    "backend": Optional[str],
    "arch": Optional[str],
    "older_than": Optional[str],
    "younger_than": Optional[str],
    "cache_hit_lower": Optional[int],
    "cache_hit_higher": Optional[int],
    "cache_dir": Optional[Path],
    "mode": Optional[str],
}
