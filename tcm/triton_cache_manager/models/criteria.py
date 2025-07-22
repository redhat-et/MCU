"""
Data models for Triton kernels criteria.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# pylint: disable=too-many-instance-attributes
@dataclass
class SearchCriteria:
    """Data structure to hold kernel search filter criteria."""

    name: Optional[str] = None
    cache_dir: Optional[Path] = None
    backend: Optional[str] = None
    arch: Optional[str] = None
    older_than_timestamp: Optional[float] = None
    younger_than_timestamp: Optional[float] = None
    cache_hit_lower: Optional[int] = None
    cache_hit_higher: Optional[int] = None
