"""
Configuration settings for the Model Cache Manager.

This module provides configuration settings including paths and defaults.
"""

import platform
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # pylint: disable=too-few-public-methods
    """
    Configuration settings for the Model Cache Manager.

    This class defines the default paths and settings used throughout the application.
    It supports overriding via environment variables prefixed with MCM_.
    """

    model_cache_dir: Path = Field(
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
            / "model-cache-manager"
        )
    )
    db_filename: str = "cache.db"
    log_level: str = "INFO"

    class Config:
        """Configuration for the Settings class."""

        env_prefix = "MCM_"


settings = Settings()
Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
