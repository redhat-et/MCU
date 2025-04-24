from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
import platform, os


class Settings(BaseSettings):
    triton_cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".triton" / "cache"
    )
    data_dir: Path = Field(
        default_factory=lambda: (
            Path.home()
            / (
                ".local/share"
                if platform.system() == "Linux"
                else (
                    "Library/Application Support"
                    if platform.system() == "Darwin"
                    else os.environ.get("LOCALAPPDATA", "C:/Temp")
                )
            )
            / "triton-cache-manager"
        )
    )
    db_filename: str = "cache.db"
    log_level: str = "INFO"

    class Config:
        env_prefix = "TCM_"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
