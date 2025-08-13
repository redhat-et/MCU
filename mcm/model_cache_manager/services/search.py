"""
Service for search Triton kernels based on criteria.
"""

from __future__ import annotations
from typing import Any, Dict, List
from ..data.database import Database, VllmDatabase
from ..models.criteria import SearchCriteria
from ..utils.mcm_constants import MODE_VLLM


class SearchService:
    """
    Search Triton kernels based on filters
    """

    def __init__(self, criteria: SearchCriteria, mode: str = "triton"):
        """
        Initialize the search service.

        Args:
            criteria: Search criteria for filtering kernels
            mode: Cache mode - 'triton' for standard Triton cache, 'vllm' for vLLM cache
        """
        self.mode = mode
        if mode == MODE_VLLM:
            self.db = VllmDatabase()
        else:
            self.db = Database()
        self.criteria = criteria

    def search(self) -> List[Dict[str, Any]]:
        """
        Searches for kernels matching criteria.

        Returns:
            A list of dictionaries, each representing a matching kernel.
        """
        found = self.db.search(self.criteria)

        return found

    def close(self):
        """Close the database connection."""
        self.db.close()
