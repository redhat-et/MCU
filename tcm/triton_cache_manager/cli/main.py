import typer, rich
from pathlib import Path
from ..services.index import IndexService
from ..utils.logger import configure_logging

app = typer.Typer()


@app.callback()
def base(verbose: int = typer.Option(0, "-v", "--verbose", count=True)):
    configure_logging(["ERROR", "WARNING", "INFO", "DEBUG"][min(verbose, 3)])


@app.command()
def index(cache_dir: Path = typer.Option(None)):
    svc = IndexService(cache_dir)
    try:
        n = svc.reindex()
        rich.print(f"[green]Indexed {n} kernels[/green]")
    finally:
        svc.close()


@app.command()
def search(name: str = typer.Option(None), backend: str = typer.Option(None), arch: str = typer.Option(None)):
    svc = IndexService()
    try:
        rows = svc.db.search(name=name, backend=backend, arch=arch)
        rich.print(rows)
    finally:
        svc.close()


def run():
    app()


if __name__ == "__main__":
    run()
