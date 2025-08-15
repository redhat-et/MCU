"""
Service for pruning kernel cache artefacts based on criteria
"""

from __future__ import annotations

import logging
from typing import Optional, List, Any, Tuple, Dict
from dataclasses import dataclass
from rich.prompt import Confirm
from .base import BaseService
from ..models.criteria import SearchCriteria
from ..utils.mcm_constants import IR_EXTS, MODE_VLLM
from ..utils.utils import (
    KernelIdentifier,
    create_kernel_identifier,
    extract_identifiers_from_groups,
    get_kernel_directories,
    delete_ir_files_from_dirs,
    delete_kernel_directories,
    find_vllm_kernel_dirs
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PruneStats:
    "Stats for PruningService"

    pruned: int
    reclaimed: float
    aborted: bool = False
    empty: bool = False


class PruningService(BaseService):  # pylint: disable=too-few-public-methods
    """
    Remove files from the cache and keep the DB consistent.
    """

    def prune(
        self,
        criteria: SearchCriteria,
        delete_ir_only: bool = True,
        auto_confirm: bool = False,
    ) -> Optional[PruneStats]:
        """
        Prune kernels matching criteria.

        Args:
            criteria:
                Filters to select which kernels are affected.
            delete_ir_only:
                True  – partial prune (keep binary + metadata).
                False – full prune (remove whole dir & DB row).
            auto_confirm:
                Skip the interactive - Are you sure? - prompt (used by `-y`).

        Returns:
            Tuple[pruned_kernel_count, reclaimed_bytes]
        """
        criteria.cache_dir = self.cache_dir
        rows = self.db.search(criteria)

        # Extract identifiers using strategy pattern
        identifiers = [self.strategy.extract_identifiers_from_row(row) for row in rows]
        # For estimate_space, use the correct hash field based on mode
        if self.mode == "vllm":
            hash_list = [identifier.hash_key for identifier in identifiers]  # triton_cache_key
        else:
            hash_list = [identifier.hash_key for identifier in identifiers]  # hash

        if not hash_list:
            log.info("No kernels matched pruning criteria – nothing to do.")
            return PruneStats(0, 0)

        reclaimed = self.db.estimate_space(hash_list, IR_EXTS)
        if not auto_confirm and not self._confirm(
            len(hash_list), reclaimed, delete_ir_only
        ):
            return None

        freed = 0
        with self.db.get_session() as session:
            for identifier in identifiers:
                freed += self._delete_kernel_unified(identifier, session, delete_ir_only)
            session.commit()

        log.info("Pruned %d kernels – reclaimed %.1f MB", len(hash_list), freed / 2**20)
        return PruneStats(len(hash_list), (freed / 2**20))

    def close(self) -> None:
        """Dispose DB connections"""
        self.db.close()

    @staticmethod
    def _confirm(n: int, bytes_to_free: int, ir_only: bool) -> bool:
        """Confirm to prune"""
        human = f"{bytes_to_free / 2**20:.1f} MB"
        mode = "partial (IR‑only)" if ir_only else "FULL"
        return Confirm.ask(
            f"[yellow]About to {mode} prune {n} kernel(s), freeing {human}. Continue?[/yellow]"
        )

    def _get_hashes_and_estimated_space_for_deduplication(
        self, duplicate_groups: List[List[Dict[str, Any]]]
    ) -> Tuple[List[KernelIdentifier], int]:
        """
        Identifies kernel identifiers of older duplicate kernels to prune and estimates space.
        Works unified for both triton and vLLM modes using utility functions.

        Args:
            duplicate_groups: List of groups, where each group is a list of kernel dicts
                              (sorted oldest to newest).

        Returns:
            A tuple containing:
                - List of KernelIdentifier objects to prune.
                - Estimated total bytes to be reclaimed.
        """
        # Use utility function to extract identifiers
        identifiers_to_prune = extract_identifiers_from_groups(self.mode, duplicate_groups)

        # Calculate total estimated space
        total_reclaimed_bytes_estimate = 0
        for group in duplicate_groups:
            if len(group) > 1:
                # Sum up sizes of all but the newest kernel in each group
                for kernel_dict in group[:-1]:
                    total_reclaimed_bytes_estimate += kernel_dict.get("total_size", 0)

        log.debug("Found %d older duplicate kernels to prune", len(identifiers_to_prune))
        return identifiers_to_prune, total_reclaimed_bytes_estimate

    def _perform_deduplication_deletions(
        self, session, identifiers_to_delete: List[KernelIdentifier]
    ) -> Tuple[int, int]:
        """
        Deletes the specified kernels from disk and DB using unified deletion logic.

        Args:
            session: The SQLAlchemy session.
            identifiers_to_delete: List of KernelIdentifier objects to delete.

        Returns:
            A tuple containing:
                - Total bytes freed.
                - Count of kernels successfully pruned.
        """
        freed_bytes_total = 0
        pruned_count = 0

        for identifier in identifiers_to_delete:
            freed_bytes, success = self._delete_single_kernel_safely(identifier, session)
            freed_bytes_total += freed_bytes
            if success:
                pruned_count += 1

        return freed_bytes_total, pruned_count

    def _delete_single_kernel_safely(
        self, identifier: KernelIdentifier, session
    ) -> Tuple[int, bool]:
        """
        Safely delete a single kernel with error handling.

        Args:
            identifier: KernelIdentifier object to delete
            session: Database session

        Returns:
            A tuple containing:
                - Bytes freed (0 if deletion failed)
                - Success flag
        """
        try:
            freed_for_kernel = self._delete_kernel_unified(identifier, session, ir_only=False)
            log.debug("Successfully deleted kernel: %s", identifier)
            return freed_for_kernel, True

        except (FileNotFoundError, PermissionError, OSError) as e:
            log.error(
                "Failed to delete kernel %s during deduplication: %s",
                identifier,
                e,
                exc_info=True,
            )
            return 0, False
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error(
                "Unexpected error deleting kernel %s during deduplication: %s",
                identifier,
                e,
                exc_info=True,
            )
            return 0, False

    def _delete_kernel_unified(self, identifier: KernelIdentifier, session, ir_only: bool) -> int:
        """
        Unified kernel deletion method that works for both triton and vLLM modes.

        Args:
            identifier: KernelIdentifier object
            session: Database session
            ir_only: If True, only delete IR files; if False, delete entire kernel

        Returns:
            Bytes freed
        """
        # Get kernel directories using utility function
        kernel_dirs = get_kernel_directories(self.cache_dir, self.mode, identifier)

        # Get kernel record from database
        kernel_record = self._get_kernel_record(session, identifier)

        if ir_only:
            freed = self._delete_ir_only_unified(session, identifier, kernel_dirs, kernel_record)
        else:
            freed = self._delete_full_kernel_unified(session, kernel_dirs, kernel_record)

        return freed

    def _get_kernel_record(self, session, identifier: KernelIdentifier):
        """Get kernel record from database using strategy pattern."""
        config = self.strategy.config
        if self.mode == MODE_VLLM:
            # For vLLM: (vllm_cache_root, vllm_hash, triton_cache_key)
            return session.get(
                config.orm_model,
                (str(self.cache_dir), identifier.vllm_hash, identifier.hash_key),
            )
        # For Triton: (hash, cache_dir)
        return session.get(
            config.orm_model,
            (identifier.hash_key, str(self.cache_dir)),
        )

    def _delete_ir_only_unified(self, session, identifier: KernelIdentifier,
                                kernel_dirs: List, kernel_record) -> int:
        """Delete only IR files using unified logic."""
        freed, deleted_file_names = delete_ir_files_from_dirs(kernel_dirs, IR_EXTS)

        # Update database records
        if deleted_file_names:
            self._delete_ir_file_records(session, identifier, deleted_file_names)

        # Update kernel total size
        if kernel_record:
            kernel_record.total_size = sum(f.size or 0 for f in kernel_record.files)

        return freed

    def _delete_full_kernel_unified(self, session, kernel_dirs: List, kernel_record) -> int:
        """Delete entire kernel using unified logic."""
        freed = delete_kernel_directories(kernel_dirs)

        # Delete kernel record from database
        if kernel_record:
            session.delete(kernel_record)

        return freed

    def _delete_ir_file_records(self, session, identifier: KernelIdentifier,
                                ir_file_names: List[str]) -> None:
        """Delete IR file records from database using strategy pattern."""
        config = self.strategy.config

        if self.mode == MODE_VLLM:
            ir_rows = (
                session.query(config.file_orm_model)
                .filter(
                    config.file_orm_model.vllm_cache_root == str(self.cache_dir),
                    config.file_orm_model.vllm_hash == identifier.vllm_hash,
                    config.file_orm_model.triton_cache_key == identifier.hash_key,
                    config.file_orm_model.rel_path.in_(ir_file_names),
                )
                .all()
            )
        else:
            ir_rows = (
                session.query(config.file_orm_model)
                .filter(
                    config.file_orm_model.kernel_hash == identifier.hash_key,
                    config.file_orm_model.rel_path.in_(ir_file_names),
                )
                .all()
            )

        for r in ir_rows:
            session.delete(r)

    def deduplicate_kernels(self, auto_confirm: bool = False) -> Optional[PruneStats]:
        """
        Finds duplicate kernels and prunes the older ones, keeping only the
        newest instance of each set. Kernels are fully deleted.
        """
        duplicate_groups = self.db.find_duplicates()
        if not duplicate_groups:
            log.info("No duplicate kernels found – nothing to do.")
            return PruneStats(pruned=0, reclaimed=0.0, empty=True)

        kernels_to_prune_identifiers, total_reclaimed_bytes_estimate = (
            self._get_hashes_and_estimated_space_for_deduplication(duplicate_groups)
        )

        if not kernels_to_prune_identifiers:
            log.info("No older duplicate kernel instances to prune after analysis.")
            return PruneStats(pruned=0, reclaimed=0.0, empty=True)

        num_kernels_to_prune = len(kernels_to_prune_identifiers)
        if not auto_confirm:
            if not self._confirm(
                num_kernels_to_prune, total_reclaimed_bytes_estimate, ir_only=False
            ):
                log.info("Deduplication prune cancelled by user.")
                return PruneStats(pruned=0, reclaimed=0.0, aborted=True)

        freed_bytes_total = 0
        pruned_count = 0
        with self.db.get_session() as session:
            freed_bytes_total, pruned_count = self._perform_deduplication_deletions(
                session, kernels_to_prune_identifiers
            )
            session.commit()
        reclaimed_mb = freed_bytes_total / (1024 * 1024)
        log.info(
            "Deduplication pruned %d older kernel instance(s), reclaiming %.1f MB.",
            pruned_count,
            reclaimed_mb,
        )
        return PruneStats(pruned=pruned_count, reclaimed=reclaimed_mb)


    def _find_vllm_kernel_dirs(self, vllm_hash: str, triton_cache_key: str) -> list:
        """Find kernel directories for a given vLLM hash and triton cache key.
        Wrapper method for backward compatibility - uses utility function."""
        return find_vllm_kernel_dirs(self.cache_dir, vllm_hash, triton_cache_key)

    def _delete_vllm_kernel(
        self, vllm_hash: str, triton_cache_key: str, session, ir_only: bool
    ) -> int:
        """
        Delete vLLM kernel files on disk and update db. Returns bytes freed.
        Wrapper method for backward compatibility - uses unified deletion logic.

        Args:
            vllm_hash: The vLLM hash (used for directory structure)
            triton_cache_key: The Triton cache key (used for DB lookups)
            session: Database session
            ir_only: If True, only delete IR files; if False, delete entire kernel
        """
        identifier = create_kernel_identifier(
            mode=self.mode,
            vllm_hash=vllm_hash,
            triton_cache_key=triton_cache_key
        )
        return self._delete_kernel_unified(identifier, session, ir_only)
