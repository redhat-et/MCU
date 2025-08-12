"""
Service for pruning kernel cache artefacts based on criteria
"""

from __future__ import annotations

import logging
import shutil
from typing import Optional, List, Any, Tuple, Dict
from dataclasses import dataclass
from rich.prompt import Confirm
from .base import BaseService
from ..models.criteria import SearchCriteria
from ..data.db_models import KernelOrm, KernelFileOrm, VllmKernelOrm, VllmKernelFileOrm
from ..utils.mcm_constants import IR_EXTS

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
        if self.mode == "vllm":
            # For vLLM mode, we need both vllm_hash and triton_cache_key for deletion
            vllm_data: List[Tuple[str, str]] = [
                (row["vllm_hash"], row["triton_cache_key"]) for row in rows
            ]
            hash_list = [row["triton_cache_key"] for row in rows]  # For estimate_space
        else:
            hashes: List[str] = [row["hash"] for row in rows]
            hash_list = hashes

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
            if self.mode == "vllm":
                for vllm_hash, triton_cache_key in vllm_data:
                    freed += self._delete_vllm_kernel(
                        vllm_hash, triton_cache_key, session, delete_ir_only
                    )
            else:
                for h in hashes:
                    freed += self._delete_kernel(h, session, delete_ir_only)
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
    ) -> Tuple[List[str] | List[Tuple[str, str]], int]:
        """
        Identifies hashes of older duplicate kernels to prune and estimates reclaimable space.

        Args:
            duplicate_groups: List of groups, where each group is a list of kernel dicts
                              (sorted oldest to newest).

        Returns:
            A tuple containing:
                - List of kernel identifiers to prune (either hash strings for triton mode
                  or (vllm_hash, triton_cache_key) tuples for vllm mode).
                - Estimated total bytes to be reclaimed.
        """
        if self.mode == "vllm":
            kernels_to_prune_identifiers: List[Tuple[str, str]] = []
            hash_field = "triton_cache_key"
        else:
            kernels_to_prune_identifiers: List[str] = []
            hash_field = "hash"

        total_reclaimed_bytes_estimate = 0

        kernel_info_map: Dict[str, Dict[str, Any]] = {}
        for group in duplicate_groups:
            for kernel_dict in group:
                kernel_info_map[kernel_dict[hash_field]] = kernel_dict

        for group in duplicate_groups:
            log.debug(
                "Processing duplicate group: %s", [k.get(hash_field) for k in group]
            )
            if len(group) > 1:
                for kernel_to_prune_dict in group[:-1]:
                    if self.mode == "vllm":
                        vllm_hash = kernel_to_prune_dict["vllm_hash"]
                        triton_cache_key = kernel_to_prune_dict["triton_cache_key"]
                        kernels_to_prune_identifiers.append(
                            (vllm_hash, triton_cache_key)
                        )
                    else:
                        h_hash = kernel_to_prune_dict["hash"]
                        kernels_to_prune_identifiers.append(h_hash)
                    total_reclaimed_bytes_estimate += kernel_to_prune_dict.get(
                        "total_size", 0
                    )

        return kernels_to_prune_identifiers, total_reclaimed_bytes_estimate

    def _perform_deduplication_deletions(
        self, session, identifiers_to_delete: List[str] | List[Tuple[str, str]]
    ) -> Tuple[int, int]:
        """
        Deletes the specified kernels from disk and DB.

        Args:
            session: The SQLAlchemy session.
            identifiers_to_delete: List of kernel identifiers to delete
                                   (either hash strings for triton mode or
                                   (vllm_hash, triton_cache_key) tuples for vllm mode).

        Returns:
            A tuple containing:
                - Total bytes freed.
                - Count of kernels successfully pruned.
        """
        freed_bytes_total = 0
        pruned_count = 0
        for identifier in identifiers_to_delete:
            # Set identifier_str early for error handling
            if self.mode == "vllm" and isinstance(identifier, tuple):
                vllm_hash, triton_cache_key = identifier
                identifier_str = (
                    f"vllm_hash={vllm_hash}, triton_cache_key={triton_cache_key}"
                )
            else:
                identifier_str = str(identifier)

            try:
                if self.mode == "vllm" and isinstance(identifier, tuple):
                    vllm_hash, triton_cache_key = identifier
                    freed_for_kernel = self._delete_vllm_kernel(
                        vllm_hash, triton_cache_key, session, ir_only=False
                    )
                else:
                    freed_for_kernel = self._delete_kernel(
                        identifier, session, ir_only=False
                    )
                freed_bytes_total += freed_for_kernel
                pruned_count += 1
            except FileNotFoundError as e_fnf:
                log.error(
                    "Kernel directory or file not found for %s during deduplication: %s. ",
                    identifier_str,
                    e_fnf,
                    exc_info=True,
                )
            except PermissionError as e_perm:
                log.error(
                    "Permission denied while trying to delete kernel %s during deduplication: %s.",
                    identifier_str,
                    e_perm,
                    exc_info=True,
                )
            except OSError as e_os:
                log.error(
                    "OS error while deleting kernel %s during deduplication: %s.",
                    identifier_str,
                    e_os,
                    exc_info=True,
                )
            # pylint: disable=broad-exception-caught
            except Exception as e_del:
                log.error(
                    "Failed to delete kernel %s during deduplication: %s",
                    identifier_str,
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

    def _delete_kernel(self, h: str, session, ir_only: bool) -> int:
        """
        Delete files on disk and update db. Returns bytes freed.
        """
        k_dir = self.repo.root / h
        freed = 0

        kernel_row: KernelOrm | None = session.get(
            KernelOrm,
            (h, str(self.cache_dir)),
        )
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

            if kernel_row:
                session.delete(kernel_row)

        return freed

    def _find_vllm_kernel_dirs(self, vllm_hash: str, triton_cache_key: str) -> list:
        """Find kernel directories for a given vLLM hash and triton cache key."""
        vllm_root_dir = self.repo.root / "torch_compile_cache" / vllm_hash
        kernel_dirs = []

        if vllm_root_dir.exists():
            for rank_dir in vllm_root_dir.iterdir():
                if rank_dir.is_dir() and rank_dir.name.startswith("rank"):
                    triton_cache_dir = rank_dir / "triton_cache"
                    kernel_dir = triton_cache_dir / triton_cache_key
                    if kernel_dir.exists():
                        kernel_dirs.append(kernel_dir)
        return kernel_dirs

    def _delete_vllm_ir_files(self, kernel_dirs: list, vllm_hash: str,
                              triton_cache_key: str, session) -> int:
        """Delete IR files and update database records. Returns bytes freed."""
        freed = 0
        for k_dir in kernel_dirs:
            files = list(k_dir.iterdir()) if k_dir.exists() else []
            for p in files:
                if p.suffix in IR_EXTS and p.is_file():
                    try:
                        freed += p.stat().st_size
                        p.unlink()
                    except OSError as err:
                        log.warning("Could not delete %s: %s", p, err)

            # Remove IR file records from database
            ir_file_names = [p.name for p in files if p.suffix in IR_EXTS]
            ir_rows = (
                session.query(VllmKernelFileOrm)
                .filter(
                    VllmKernelFileOrm.vllm_cache_root == str(self.cache_dir),
                    VllmKernelFileOrm.vllm_hash == vllm_hash,
                    VllmKernelFileOrm.triton_cache_key == triton_cache_key,
                    VllmKernelFileOrm.rel_path.in_(ir_file_names),
                )
                .all()
            )
            for r in ir_rows:
                session.delete(r)
        return freed

    def _delete_vllm_kernel_dirs(self, kernel_dirs: list) -> int:
        """Delete entire kernel directories. Returns bytes freed."""
        freed = 0
        for k_dir in kernel_dirs:
            if k_dir.exists():
                try:
                    freed += sum(
                        p.stat().st_size for p in k_dir.rglob("*") if p.is_file()
                    )
                    shutil.rmtree(k_dir)
                except OSError as err:
                    log.error("Failed to remove %s: %s", k_dir, err, exc_info=True)
        return freed

    def _delete_vllm_kernel(
        self, vllm_hash: str, triton_cache_key: str, session, ir_only: bool
    ) -> int:
        """
        Delete vLLM kernel files on disk and update db. Returns bytes freed.

        Args:
            vllm_hash: The vLLM hash (used for directory structure)
            triton_cache_key: The Triton cache key (used for DB lookups)
            session: Database session
            ir_only: If True, only delete IR files; if False, delete entire kernel
        """
        kernel_dirs = self._find_vllm_kernel_dirs(vllm_hash, triton_cache_key)

        # Get the kernel record from the database
        kernel_row: VllmKernelOrm | None = session.get(
            VllmKernelOrm,
            (str(self.cache_dir), vllm_hash, triton_cache_key),
        )

        if ir_only:
            freed = self._delete_vllm_ir_files(
                kernel_dirs, vllm_hash, triton_cache_key, session
            )
            # Update kernel total size
            if kernel_row:
                kernel_row.total_size = sum(f.size or 0 for f in kernel_row.files)
        else:
            freed = self._delete_vllm_kernel_dirs(kernel_dirs)
            # Delete kernel record from database
            if kernel_row:
                session.delete(kernel_row)

        return freed
