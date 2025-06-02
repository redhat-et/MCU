"""
Data models for Triton kernels criteria.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SearchCriteria:
    """Data structure to hold kernel search filter criteria."""

    name: Optional[str] = None
    cache_dir: Optional[Path] = None
    backend: Optional[str] = None
    arch: Optional[str] = None
    older_than_timestamp: Optional[float] = None
    younger_than_timestamp: Optional[float] = None
