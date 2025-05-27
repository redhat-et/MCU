"""
CLI interface for the Triton Cache Manager.

This module provides command-line commands to interact with the Triton kernel cache.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
import rich
from rich.table import Table
from ..services.index import IndexService
from ..utils.logger import configure_logging
from ..utils.paths import get_db_path
from ..utils.utils import (
    format_size,
    mod_time_handle,
    get_older_younger,
)
from ..models.criteria import SearchCriteria
from ..services.prune import PruningService, PruneStats

log = logging.getLogger(__name__)
app = typer.Typer(help="Triton Kernel Cache Manager CLI")


@app.callback()
def base(
    verbose: int = typer.Option(
        0, "-v", "--verbose", count=True, help="Increase logging verbosity."
    )
):
    """
    Base callback to configure logging level.
    """
    log_level = ["ERROR", "WARNING", "INFO", "DEBUG"][min(verbose, 3)]
    configure_logging(log_level)


@app.command()
def index(
    cache_dir: Optional[Path] = typer.Option(
        None,
        help="Specify the Triton cache directory to index. Uses default if not provided.",
    )
):
    """
    Index kernels found in the cache directory and store metadata in the database.
    """
    svc = None
    try:
        svc = IndexService(cache_dir=cache_dir)
        rich.print(f"Starting indexing process for cache directory: {svc.repo.root}...")
        n = svc.reindex()
        rich.print(f"[green]Kernels Numbers\n\t before: {n[1]}\n\t now: {n[0]}[/green]")
    except FileNotFoundError as e:
        missing_path = (
            e.filename if hasattr(e, "filename") else (cache_dir or "default location")
        )
        rich.print(f"[red]Error: Cache directory not found at '{missing_path}'.[/red]")
    except IndexError as e:
        rich.print(f"[red]Index error during indexing: {e}[/red]")
    except ValueError as e:
        rich.print(f"[red]Value error during indexing: {e}[/red]")
    except KeyError as e:
        rich.print(f"[red]Key error during indexing: {e}[/red]")
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Acceptable use of broad exception to prevent CLI crashes
        rich.print(f"[red]An unexpected error occurred during indexing: {e}[/red]")
    finally:
        if svc:
            try:
                svc.close()
                rich.print("Database connection closed.")
            except Exception as e_close:  # pylint: disable=broad-exception-caught
                # Acceptable use of broad exception in cleanup code
                rich.print(
                    f"[yellow]Warning: Error closing database connection: {e_close}[/yellow]"
                )


def _cache_db_exists() -> bool:
    """
    Check if the cache database file exists.

    Returns:
        bool: True if the database file exists, False otherwise
    """
    db_path = get_db_path()
    exists = db_path.exists()

    if exists:
        log.debug("Cache database found at %s", db_path)
    else:
        log.debug("Cache database not found at %s", db_path)

    return exists


def _display_kernels_table(rows: List[Dict[str, Any]]):
    """
    Helper function to display kernel data (list of dicts) in a rich Table.
    """
    if not rows:
        rich.print(
            "[yellow]No kernels found matching the criteria.\
                   Have you used `tcm index` first?[/yellow]"
        )
        return

    table = Table(
        title=f"Kernel(s) Found {len(rows)}",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Hash", style="dim", width=15, overflow="fold")
    table.add_column("Name", style="cyan", min_width=20, overflow="fold")
    table.add_column("Modified", style="magenta", width=18)
    table.add_column("Backend", style="green", width=5)
    table.add_column("Arch", style="blue", width=5)
    table.add_column("Version", style="yellow", width=5)
    table.add_column("Warp size", style="dim", width=5)
    table.add_column("Warps", style="dim", width=5)
    table.add_column("Stages", style="dim", width=5)
    table.add_column("Shared", style="dim", width=8)
    table.add_column("Size", style="cyan", width=10)
    table.add_column("Debug", style="dim", width=8)

    for row in rows:
        row_dict = dict(row)
        total_size_bytes = row_dict.get("total_size", 0)
        total_size_str = format_size(total_size_bytes)
        shared_size_bytes = row_dict.get("shared", 0)
        shared_size_str = format_size(shared_size_bytes)
        mod_time_unix = row_dict.get("modified_time")
        mod_time_str = mod_time_handle(mod_time_unix)
        table.add_row(
            row_dict.get("hash", "N/A")[:12] + "...",
            row_dict.get("name", "N/A"),
            mod_time_str,
            row_dict.get("backend", "N/A"),
            row_dict.get("arch", "N/A"),
            row_dict.get("triton_version", "N/A"),
            str(row_dict.get("warp_size", "N/A")),
            str(row_dict.get("num_warps", "N/A")),
            str(row_dict.get("num_stages", "N/A")),
            shared_size_str,
            total_size_str,
            str(row_dict.get("debug", "N/A")),
        )

    rich.print(table)


@app.command(name="list")
def search(
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Filter by kernel name (exact match)."
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Filter by backend (e.g., 'cuda', 'rocm')."
    ),
    arch: Optional[str] = typer.Option(
        None, "--arch", "-a", help="Filter by architecture (e.g., '120', 'gfx90a')."
    ),
    older_than: Optional[str] = typer.Option(
        None,
        "--older-than",
        help="Show kernels older than specified duration (e.g., '7d', '2w').",
    ),
    younger_than: Optional[str] = typer.Option(
        None,
        "--younger-than",
        help="Show kernels younger than specified duration (e.g., '14d', '1w').",
    ),
):
    """
    Search for indexed kernels based on various SearchCriteria.
    """
    if not _cache_db_exists():
        rich.print("[red]DB was not found. Have you used `tcm index` first?[/red]")
        return

    older_younger = get_older_younger(older_than, younger_than)

    criteria = SearchCriteria(
        name=name,
        backend=backend,
        arch=arch,
        older_than_timestamp=older_younger[0],
        younger_than_timestamp=older_younger[1],
    )

    svc = None
    try:
        svc = IndexService()
        rich.print(
            f"Searching for kernels with: Name='{name or 'any'}', "
            f"Backend='{backend or 'any'}', Arch='{arch or 'any'}', "
            f"OlderThan='{older_than or 'N/A'}', YoungerThan='{younger_than or 'N/A'}'..."
        )
        rows = svc.db.search(criteria)
        _display_kernels_table(rows)
    except Exception as e:  # pylint: disable=broad-exception-caught
        rich.print(f"[red]An error occurred during search: {e}[/red]")
        log.exception("Search command failed")
    finally:
        if svc:
            try:
                svc.close()
            except Exception as e_close:  # pylint: disable=broad-exception-caught
                rich.print(
                    f"[yellow]Warning: Error closing database connection: {e_close}[/yellow]"
                )


# pylint: disable=too-many-positional-arguments
@app.command()
def prune(  # pylint: disable=too-many-arguments
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Filter by kernel name (exact match)."
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Filter by backend (e.g., 'cuda', 'rocm')."
    ),
    arch: Optional[str] = typer.Option(
        None, "--arch", "-a", help="Filter by architecture (e.g., '120', 'gfx90a')."
    ),
    older_than: Optional[str] = typer.Option(
        None,
        "--older-than",
        help="Show kernels older than specified duration (e.g., '7d', '2w').",
    ),
    younger_than: Optional[str] = typer.Option(
        None,
        "--younger-than",
        help="Show kernels younger than specified duration (e.g., '14d', '1w').",
    ),
    full: bool = typer.Option(
        False, "--full", help="Remove the entire kernel directory."
    ),
    deduplicate: bool = typer.Option(
        False,
        "--deduplicate",
        help="Delete older duplicate kernels, keeping only the newest of each set.",
    ),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation prompt."),
):
    """
    Delete intermediate‑representation files (default) or whole kernel
    directories that satisfy the given filters.
    """
    if not _cache_db_exists():
        rich.print("[red]DB was not found. Have you used `tcm index` first?[/red]")
        return

    svc = None
    try:
        svc = PruningService()
        stats: Optional[PruneStats] = None
        if deduplicate:
            filter_options_used = [name, backend, arch, older_than, younger_than, full]
            if any(opt is not None and opt is not False for opt in filter_options_used):
                rich.print(
                    "[yellow]Warning: Filter options and --full are ignored"
                    + "when --deduplicate is used.[/yellow]"
                )
            rich.print("Starting kernel deduplication process...")
            stats = svc.deduplicate_kernels(auto_confirm=yes)
        else:
            older_younger = get_older_younger(older_than, younger_than)

            criteria = SearchCriteria(
                name=name,
                backend=backend,
                arch=arch,
                older_than_timestamp=older_younger[0],
                younger_than_timestamp=older_younger[1],
            )
            stats = svc.prune(
                criteria,
                delete_ir_only=not full,
                auto_confirm=yes,
            )

        if stats is None:
            rich.print("[yellow]Prune cancelled by user.[/yellow]")
            return
        if stats.pruned == 0:
            rich.print("[yellow]No kernels matched the given filters.[/yellow]")
            return

        if not full:
            rich.print(f"[green]Pruned IRs of {stats.pruned} kernel(s).[/green]")
        else:
            rich.print(f"[green]Pruned fully {stats.pruned} kernel(s).[/green]")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        rich.print(f"[red]Prune failed: {exc}[/red]")
        log.exception("Prune command failed")
    finally:
        if svc:
            svc.close()


def run():
    """Entry point for the Typer application."""
    app()


if __name__ == "__main__":
    run()
