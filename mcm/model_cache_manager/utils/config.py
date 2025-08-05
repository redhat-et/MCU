"""
Configuration settings for the Model Cache Manager.

This module provides configuration settings including paths and defaults.
"""

import platform
import os
from pathlib import Path


class Settings:
    # pylint: disable=too-few-public-methods
    """
    Configuration settings for the Model Cache Manager.

    This class defines the default paths and settings used throughout the application.
    It supports overriding via environment variables prefixed with MCM_.
    """

    def __init__(self):
        self.model_cache_dir = Path.home() / ".triton" / "cache"
        self.model_cache_dir_vllm = Path.home() / ".cache" / "vllm"
        self.data_dir = (
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
        self.db_filename = "cache.db"
        self.log_level = "INFO"
        
        # Override with environment variables if present
        if "MCM_MODEL_CACHE_DIR" in os.environ:
            self.model_cache_dir = Path(os.environ["MCM_MODEL_CACHE_DIR"])
        if "MCM_DATA_DIR" in os.environ:
            self.data_dir = Path(os.environ["MCM_DATA_DIR"])
        if "MCM_DB_FILENAME" in os.environ:
            self.db_filename = os.environ["MCM_DB_FILENAME"]
        if "MCM_LOG_LEVEL" in os.environ:
            self.log_level = os.environ["MCM_LOG_LEVEL"]


settings = Settings()
Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
