"""
CLI helper functions for the Model Cache Manager.

This module provides shared utilities to reduce code duplication across CLI commands.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator, Any
import typer
import rich
from ..utils.paths import get_cache_dir, get_db_path
from ..utils.utils import detect_cache_mode

VALID_MODES = ("triton", "vllm")


def resolve_mode(explicit: Optional[str], cache_dir: Optional[Path]) -> str:
    """Return validated or auto-detected mode, raise ValueError on bad input."""
    if explicit and explicit not in VALID_MODES:
        raise ValueError(f"Invalid mode '{explicit}'. Must be {VALID_MODES}.")

    if explicit:
        return explicit

    # Auto-detect mode
    actual_cache_dir = cache_dir or get_cache_dir()
    mode = detect_cache_mode(actual_cache_dir)
    rich.print(f"[blue]Auto-detected cache mode: {mode}[/blue]")
    return mode


def ensure_db(mode: str) -> None:
    """Abort command early if DB doesn't exist."""
    db_path = get_db_path(mode)
    if not db_path.exists():
        rich.print(
            f"[red]DB was not found for {mode} mode. "
            f"Run `mcm index --mode {mode}` first.[/red]"
        )
        raise typer.Exit(code=1)


@contextmanager
def service_ctx(cls, *args, **kwargs) -> Generator[Any, None, None]:
    """Context-manager wrapper for all *Service classes."""
    svc = cls(*args, **kwargs)
    try:
        yield svc
    finally:
        try:
            svc.close()
        except Exception as exc:  # pylint: disable=broad-except
            rich.print(f"[yellow]Warning while closing: {exc}[/yellow]")
