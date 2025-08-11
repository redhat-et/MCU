"""
CLI interface for the Model Cache Manager.

This module provides command-line commands to interact with the Triton kernel cache.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
import rich
from rich.table import Table
from ..services.index import IndexService
from ..services.search import SearchService
from ..utils.logger import configure_logging
from ..utils.utils import (
    format_size,
    mod_time_handle,
    get_older_younger,
    check_hits_num,
)
from ..models.criteria import SearchCriteria
from ..services.prune import PruningService, PruneStats
from ..services.warm import WarmupService
from .cli_helpers import resolve_mode, ensure_db, service_ctx
from .cli_options import get_common_search_options

log = logging.getLogger(__name__)
app = typer.Typer(help="Model Kernel Cache Manager CLI")

# Constants for container images
DEFAULT_CUDA_IMAGE = "quay.io/rh-ee-asangior/vllm-0.9.2-tcm-warm:0.0.2"
DEFAULT_ROCM_IMAGE = "quay.io/rh-ee-asangior/vllm-0.9.1-tcm-warm-rocm:0.0.1"

# Log level mapping
LOG_LEVELS = {0: "ERROR", 1: "WARNING", 2: "INFO", 3: "DEBUG"}


@app.callback()
def base(
    verbose: int = typer.Option(
        0, "-v", "--verbose", count=True, help="Increase logging verbosity."
    )
):
    """
    Base callback to configure logging level.
    """
    log_level = LOG_LEVELS[min(verbose, 3)]
    configure_logging(log_level)


@app.command()
def index(
    cache_dir: Optional[Path] = typer.Option(
        None,
        help="Specify the Model cache directory to index. Uses default if not provided.",
    ),
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        help="Cache mode: 'triton' for standard Triton cache, 'vllm' for "
        "vLLM cache structure. Auto-detected if not specified.",
    ),
):
    """
    Index kernels found in the cache directory and store metadata in the database.
    """
    try:
        mode = resolve_mode(mode, cache_dir)

        with service_ctx(IndexService, cache_dir=cache_dir, mode=mode) as svc:
            rich.print(
                f"Starting indexing process for cache directory: {svc.repo.root}..."
            )
            n = svc.reindex()
            rich.print(
                f"[green]Number of kernels in {svc.cache_dir}\n\tbefore: {n[1]}\n\tnow: {n[0]}[/green]"
            )
    except (ValueError, FileNotFoundError) as exc:
        rich.print(f"[red]{exc}[/red]")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        rich.print(f"[red]Unexpected error: {exc}[/red]")


def _display_kernels_table(rows: List[Dict[str, Any]], mode: str = "triton"):
    """
    Helper function to display kernel data (list of dicts) in a rich Table.
    """
    if not rows:
        rich.print(
            "[yellow]No kernels found matching the criteria.\
                   Have you used `mcm index` first?[/yellow]"
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
    table.add_column("Hits", style="green", min_width=5, overflow="fold")
    table.add_column("Last Access", style="magenta", width=18)
    table.add_column("Backend", style="green", width=5)
    table.add_column("Arch", style="blue", width=5)
    table.add_column("Version", style="yellow", width=5)
    table.add_column("Warps", style="dim", width=5)
    table.add_column("Total Size", style="dim", width=5)
    table.add_column("Dir", style="dim", width=15)

    for row in rows:
        row_dict = dict(row)
        total_size_bytes = row_dict.get("total_size", 0)
        total_size_str = format_size(total_size_bytes)
        last_time_unix = row_dict.get("last_access_time")
        last_time_str = mod_time_handle(last_time_unix)
        num_hits = row_dict.get("runtime_hits", 0)
        num_hits_str = str(num_hits)
        if mode != "vllm":
            hash_mode = row_dict.get("hash", "N/A")[:12] + "..."
        else:
            hash_mode = row_dict.get("vllm_hash", "N/A")

        table.add_row(
            hash_mode,
            row_dict.get("name", "N/A"),
            num_hits_str,
            last_time_str,
            row_dict.get("backend", "N/A"),
            row_dict.get("arch", "N/A"),
            row_dict.get("triton_version", "N/A"),
            str(row_dict.get("num_warps", "N/A")),
            total_size_str,
            str(row_dict.get("cache_dir", "N/A")),
        )

    rich.print(table)


@app.command(name="list")
def search(
    name: Optional[str] = get_common_search_options()["name"],
    backend: Optional[str] = get_common_search_options()["backend"],
    arch: Optional[str] = get_common_search_options()["arch"],
    older_than: Optional[str] = get_common_search_options()["older_than"],
    younger_than: Optional[str] = get_common_search_options()["younger_than"],
    cache_hit_lower: Optional[int] = get_common_search_options()["cache_hit_lower"],
    cache_hit_higher: Optional[int] = get_common_search_options()["cache_hit_higher"],
    cache_dir: Optional[Path] = get_common_search_options()["cache_dir"],
    mode: Optional[str] = get_common_search_options()["mode"],
):
    """
    Search for indexed kernels based on various SearchCriteria.
    """
    try:
        mode = resolve_mode(mode, cache_dir)
        ensure_db(mode)

        if not check_hits_num(cache_hit_higher, cache_hit_lower):
            raise ValueError("--cache-hit-lower cannot exceed --cache-hit-higher")

        older_ts, younger_ts = get_older_younger(older_than, younger_than)

        criteria = SearchCriteria(
            cache_dir=cache_dir,
            name=name,
            backend=backend,
            arch=arch,
            older_than_timestamp=older_ts,
            younger_than_timestamp=younger_ts,
            cache_hit_lower=cache_hit_lower,
            cache_hit_higher=cache_hit_higher,
        )

        with service_ctx(SearchService, criteria=criteria, mode=mode) as svc:
            rich.print(
                f"Searching for kernels for {mode} with: Name='{name or 'any'}', "
                f"Cache_dir='{cache_dir or 'any'}', "
                f"Backend='{backend or 'any'}', Arch='{arch or 'any'}', "
                f"OlderThan='{older_than or 'N/A'}', YoungerThan='{younger_than or 'N/A'}'..."
            )
            rows = svc.search()
        _display_kernels_table(rows=rows, mode=mode)
    except (ValueError, FileNotFoundError) as exc:
        rich.print(f"[red]{exc}[/red]")
    except typer.Exit:
        # Re-raise typer.Exit to allow proper exit code handling
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.exception("search failed")
        rich.print(f"[red]Unexpected error: {exc}[/red]")


@app.command()
def prune(
    name: Optional[str] = get_common_search_options()["name"],
    backend: Optional[str] = get_common_search_options()["backend"],
    arch: Optional[str] = get_common_search_options()["arch"],
    older_than: Optional[str] = get_common_search_options()["older_than"],
    younger_than: Optional[str] = get_common_search_options()["younger_than"],
    cache_hit_lower: Optional[int] = get_common_search_options()["cache_hit_lower"],
    cache_hit_higher: Optional[int] = get_common_search_options()["cache_hit_higher"],
    full: bool = typer.Option(
        False, "--full", help="Remove the entire kernel directory."
    ),
    deduplicate: bool = typer.Option(
        False,
        "--deduplicate",
        help="Delete older duplicate kernels, keeping only the newest of each set.",
    ),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation prompt."),
    cache_dir: Optional[Path] = get_common_search_options()["cache_dir"],
    mode: Optional[str] = get_common_search_options()["mode"],
):
    """
    Delete intermediate‑representation files (default) or whole kernel
    directories that satisfy the given filters.
    """
    try:
        mode = resolve_mode(mode, cache_dir)
        ensure_db(mode)

        if not check_hits_num(cache_hit_higher, cache_hit_lower):
            raise ValueError("--cache-hit-lower cannot exceed --cache-hit-higher")

        with service_ctx(PruningService, cache_dir=cache_dir, mode=mode) as svc:
            stats: Optional[PruneStats] = None
            if deduplicate:
                filter_options_used = [
                    name,
                    backend,
                    arch,
                    older_than,
                    younger_than,
                    full,
                ]
                if any(
                    opt is not None and opt is not False for opt in filter_options_used
                ):
                    rich.print(
                        "[yellow]Warning: Filter options and --full are ignored"
                        + "when --deduplicate is used.[/yellow]"
                    )
                rich.print("Starting kernel deduplication process...")
                stats = svc.deduplicate_kernels(auto_confirm=yes)
            else:
                older_ts, younger_ts = get_older_younger(older_than, younger_than)

                criteria = SearchCriteria(
                    name=name,
                    backend=backend,
                    arch=arch,
                    older_than_timestamp=older_ts,
                    younger_than_timestamp=younger_ts,
                    cache_hit_higher=cache_hit_higher,
                    cache_hit_lower=cache_hit_lower,
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
    except (ValueError, FileNotFoundError) as exc:
        rich.print(f"[red]{exc}[/red]")
    except typer.Exit:
        # Re-raise typer.Exit to allow proper exit code handling
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        rich.print(f"[red]Prune failed: {exc}[/red]")
        log.exception("Prune command failed")


@app.command()
def warm(
    model: str = typer.Option(
        "facebook/opt-125m",
        "--model",
        "-m",
        help="The model to use for warming the cache.",
    ),
    output_file: Path = typer.Option(
        "warmed_cache.tar.gz",
        "--output",
        "-o",
        help="The path to save the packaged cache archive.",
    ),
    host_cache_dir: str = typer.Option(
        "./",
        "--host-cache-dir",
        help="Specify the vLLM cache directory to use on the host",
    ),
    hug_face_token: str = typer.Option(
        None, "--hugging-face-token", help="Add HF Token"
    ),
    vllm_cache_dir: str = typer.Option(
        "/root/.cache/vllm/",
        "--vllm_cache_dir",
        help="Specify the vLLM cache directory to use on the container",
    ),
    tarball: bool = typer.Option(
        False, "--tarball", help="Create a tarball of the vLLM cache"
    ),
    rocm: bool = typer.Option(
        False, "--rocm", help="Warm vLLM cache for rocm. Default cuda."
    ),
):
    """
    Warms up the Model cache using a specified container image and
    optionally packages the result into a tarball.
    """
    image = DEFAULT_ROCM_IMAGE if rocm else DEFAULT_CUDA_IMAGE

    svc = None
    try:
        rich.print(f"Starting cache warm for '{model}'...")
        svc = WarmupService(model, hug_face_token, vllm_cache_dir, host_cache_dir)
        success = svc.warmup(image, output_file, tarball, rocm)

        if success:
            rich.print(
                f"[green]Cache warmup successful! Saved to: {host_cache_dir}[/green]"
            )
        else:
            rich.print(
                "[red]Cache warmup failed. Check the logs for more details.[/red]"
            )
            raise typer.Exit(code=1)

    except Exception as e:
        rich.print(f"[red]An unexpected error occurred during warmup: {e}[/red]")
        raise typer.Exit(code=1)


def run():
    """Entry point for the Typer application."""
    app()


if __name__ == "__main__":
    run()
