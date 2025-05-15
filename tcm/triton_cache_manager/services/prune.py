"""
Service for pruning Triton‑kernel cache artefacts based on criteria
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional, List, Any, Tuple, Dict
from dataclasses import dataclass
from rich.prompt import Confirm
from ..data.cache_repo import CacheRepository
from ..data.database import Database
from ..models.criteria import SearchCriteria
from ..data.db_models import KernelOrm, KernelFileOrm
from ..utils.tcm_constants import IR_EXTS

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PruneStats:
    "Stats for PruningService"

    pruned: int
    reclaimed: float
    aborted: bool = False
    empty: bool = False


class PruningService:  # pylint: disable=too-few-public-methods
    """
    Remove files from the Triton cache and keep the DB consistent.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.repo = CacheRepository(cache_dir)
        self.db = Database()

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
        rows = self.db.search(criteria)
        hashes: List[str] = [row["hash"] for row in rows]

        if not hashes:
            log.info("No kernels matched pruning criteria – nothing to do.")
            return PruneStats(0, 0)

        reclaimed = self.db.estimate_space(hashes, IR_EXTS)
        if not auto_confirm and not self._confirm(
            len(hashes), reclaimed, delete_ir_only
        ):
            return None

        freed = 0
        with self.db.get_session() as session:
            for h in hashes:
                freed += self._delete_kernel(h, session, delete_ir_only)
            session.commit()

        log.info("Pruned %d kernels – reclaimed %.1f MB", len(hashes), freed / 2**20)
        return PruneStats(len(hashes), (freed / 2**20))

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
    ) -> Tuple[List[str], int]:
        """
        Identifies hashes of older duplicate kernels to prune and estimates reclaimable space.

        Args:
            duplicate_groups: List of groups, where each group is a list of kernel dicts
                              (sorted oldest to newest).

        Returns:
            A tuple containing:
                - List of kernel hashes to prune.
                - Estimated total bytes to be reclaimed.
        """
        kernels_to_prune_hashes: List[str] = []
        total_reclaimed_bytes_estimate = 0

        kernel_info_map: Dict[str, Dict[str, Any]] = {}
        for group in duplicate_groups:
            for kernel_dict in group:
                kernel_info_map[kernel_dict["hash"]] = kernel_dict

        for group in duplicate_groups:
            log.debug("Processing duplicate group: %s", [k.get("hash") for k in group])
            if len(group) > 1:
                for kernel_to_prune_dict in group[:-1]:
                    h_hash = kernel_to_prune_dict["hash"]
                    kernels_to_prune_hashes.append(h_hash)
                    total_reclaimed_bytes_estimate += kernel_to_prune_dict.get(
                        "total_size", 0
                    )

        return kernels_to_prune_hashes, total_reclaimed_bytes_estimate

    def _perform_deduplication_deletions(
        self, session, hashes_to_delete: List[str]
    ) -> Tuple[int, int]:
        """
        Deletes the specified kernels from disk and DB.

        Args:
            session: The SQLAlchemy session.
            hashes_to_delete: List of kernel hashes to delete.

        Returns:
            A tuple containing:
                - Total bytes freed.
                - Count of kernels successfully pruned.
        """
        freed_bytes_total = 0
        pruned_count = 0
        for h_to_delete in hashes_to_delete:
            try:
                freed_for_kernel = self._delete_kernel(
                    h_to_delete, session, ir_only=False
                )
                freed_bytes_total += freed_for_kernel
                pruned_count += 1
            except FileNotFoundError as e_fnf:
                log.error(
                    "Kernel directory or file not found for %s during deduplication: %s. ",
                    h_to_delete,
                    e_fnf,
                    exc_info=True,
                )
            except PermissionError as e_perm:
                log.error(
                    "Permission denied while trying to delete kernel %s during deduplication: %s.",
                    h_to_delete,
                    e_perm,
                    exc_info=True,
                )
            except OSError as e_os:
                log.error(
                    "OS error while deleting kernel %s during deduplication: %s.",
                    h_to_delete,
                    e_os,
                    exc_info=True,
                )
            # pylint: disable=broad-exception-caught
            except Exception as e_del:
                log.error(
                    "Failed to delete kernel %s during deduplication: %s",
                    h_to_delete,
                    e_del,
                    exc_info=True,
                )
        return freed_bytes_total, pruned_count

    def deduplicate_kernels(self, auto_confirm: bool = False) -> Optional[PruneStats]:
        """
        Finds duplicate kernels and prunes the older ones, keeping only the
        newest instance of each set. Kernels are fully deleted.
        """
        duplicate_groups = self.db.find_duplicates()
        if not duplicate_groups:
            log.info("No duplicate kernels found – nothing to do.")
            return PruneStats(pruned=0, reclaimed=0.0, empty=True)

        kernels_to_prune_hashes, total_reclaimed_bytes_estimate = (
            self._get_hashes_and_estimated_space_for_deduplication(duplicate_groups)
        )

        if not kernels_to_prune_hashes:
            log.info("No older duplicate kernel instances to prune after analysis.")
            return PruneStats(pruned=0, reclaimed=0.0, empty=True)

        num_kernels_to_prune = len(kernels_to_prune_hashes)
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
                session, kernels_to_prune_hashes
            )
            session.commit()
        reclaimed_mb = freed_bytes_total / (1024 * 1024)
        log.info(
            "Deduplication pruned %d older kernel instance(s), reclaiming %.1f MB.",
            pruned_count,
            reclaimed_mb,
        )
        return PruneStats(pruned=pruned_count, reclaimed=reclaimed_mb)

    def _delete_kernel(self, h: str, session, ir_only: bool) -> int:
        """
        Delete files on disk and update db. Returns bytes freed.
        """
        k_dir = self.repo.root / h
        freed = 0

        if ir_only:
            files = list(k_dir.iterdir()) if k_dir.exists() else []
            for p in files:
                if p.suffix in IR_EXTS and p.is_file():
                    try:
                        freed += p.stat().st_size
                        p.unlink()
                    except OSError as err:
                        log.warning("Could not delete %s: %s", p, err)

            ir_rows = (
                session.query(KernelFileOrm)
                .filter(
                    KernelFileOrm.kernel_hash == h,
                    KernelFileOrm.rel_path.in_(
                        [p.name for p in files if p.suffix in IR_EXTS]
                    ),
                )
                .all()
            )
            for r in ir_rows:
                session.delete(r)

            kernel_row: KernelOrm | None = session.get(KernelOrm, h)
            if kernel_row:
                kernel_row.total_size = sum(f.size or 0 for f in kernel_row.files)
        else:
            if k_dir.exists():
                try:
                    freed = sum(
                        p.stat().st_size for p in k_dir.rglob("*") if p.is_file()
                    )
                    shutil.rmtree(k_dir)
                except OSError as err:
                    log.error("Failed to remove %s: %s", k_dir, err, exc_info=True)
            if (k := session.get(KernelOrm, h)) is not None:
                session.delete(k)

        return freed
