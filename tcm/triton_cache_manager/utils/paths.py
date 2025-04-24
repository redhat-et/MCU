from pathlib import Path
from .config import settings


def get_cache_dir() -> Path:
    return settings.triton_cache_dir


def get_db_path() -> Path:
    return settings.data_dir / settings.db_filename
