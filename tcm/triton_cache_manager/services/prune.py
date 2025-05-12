"""
Service for pruning Triton‑kernel cache artefacts based on criteria
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from ..data.cache_repo import CacheRepository
from ..data.database import Database
from ..models.criteria import SearchCriteria
from ..data.db_models import KernelOrm, KernelFileOrm
from ..utils.utils import estimate_space
from ..utils.tcm_constants import IR_EXTS

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PruneStats:
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
                Filters to select which kernels are affected (same shape as `tcm list`).
            delete_ir_only:
                True  – partial prune (keep CUBIN + metadata).
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

        reclaimed = estimate_space(self.db, hashes, delete_ir_only)
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
        from rich.prompt import Confirm

        human = f"{bytes_to_free / 2**20:.1f} MB"
        mode = "partial (IR‑only)" if ir_only else "FULL"
        return Confirm.ask(
            f"[yellow]About to {mode} prune {n} kernel(s), freeing {human}. Continue?[/yellow]"
        )

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
