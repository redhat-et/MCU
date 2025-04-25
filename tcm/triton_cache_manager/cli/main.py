import typer
import rich
from rich.table import Table
from pathlib import Path
from typing import Optional, List, Dict, Any
from ..services.index import IndexService
from ..utils.logger import configure_logging

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
        rich.print(f"[green]Successfully indexed {n} kernels.[/green]")
    except FileNotFoundError as e:
        missing_path = (
            e.filename if hasattr(e, "filename") else (cache_dir or "default location")
        )
        rich.print(f"[red]Error: Cache directory not found at '{missing_path}'.[/red]")
    except Exception as e:
        rich.print(f"[red]An unexpected error occurred during indexing: {e}[/red]")
    finally:
        if svc:
            try:
                svc.close()
                rich.print("Database connection closed.")
            except Exception as e_close:
                rich.print(
                    f"[yellow]Warning: Error closing database connection: {e_close}[/yellow]"
                )


def _display_kernels_table(rows: List[Dict[str, Any]]):
    """
    Helper function to display kernel data (list of dicts) in a rich Table.
    Compatible with the original database.py's search result format.
    """
    if not rows:
        rich.print("[yellow]No kernels found matching the criteria.[/yellow]")
        return

    table = Table(
        title="Kernel Search Results",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Hash", style="dim", width=15, overflow="fold")
    table.add_column("Name", style="cyan", min_width=20, overflow="fold")
    table.add_column("Backend", style="green", width=5)
    table.add_column("Arch", style="blue", width=5)
    table.add_column("Version", style="yellow", width=5)
    table.add_column("Warps", style="dim", width=5)
    table.add_column("Stages", style="dim", width=5)
    table.add_column("Shared", style="dim", width=8)

    for row in rows:
        row_dict = dict(row)
        table.add_row(
            row_dict.get("hash", "N/A")[:12] + "...",
            row_dict.get("name", "N/A"),
            row_dict.get("backend", "N/A"),
            row_dict.get("arch", "N/A"),
            row_dict.get("triton_version", "N/A"),
            str(row_dict.get("num_warps", "N/A")),
            str(row_dict.get("num_stages", "N/A")),
            str(row_dict.get("shared", "N/A")),
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
):
    """
    Search for indexed kernels based on name, backend, or architecture.
    """
    svc = None
    try:
        svc = IndexService()
        rich.print(
            f"Searching for kernels with: Name='{name or 'any'}', Backend='{backend or 'any'}', Arch='{arch or 'any'}'..."
        )
        rows = svc.db.search(name=name, backend=backend, arch=arch)
        _display_kernels_table(rows)
    except Exception as e:
        rich.print(f"[red]An error occurred during search: {e}[/red]")
    finally:
        if svc:
            try:
                svc.close()
            except Exception as e_close:
                rich.print(
                    f"[yellow]Warning: Error closing database connection: {e_close}[/yellow]"
                )


def run():
    """Entry point for the Typer application."""
    app()


if __name__ == "__main__":
    run()
