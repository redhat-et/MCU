"""
Legacy vLLM Database module for managing kernel metadata.
"""

from typing import List, Set, Dict, Any, Iterable
import logging
from sqlalchemy import exc, or_, func

from .db_config import create_engine_and_session, DB_PATH
from .db_models import (
    Base,
    VllmLegacyKernelOrm,
    VllmLegacyKernelFileOrm,
    SqlaSession,
)
from ..models.kernel import Kernel
from ..utils.mcm_constants import IR_EXTS
from . import database as db_base

log = logging.getLogger(__name__)


class VllmLegacyDatabase:
    """
    Manages database interactions for legacy vLLM kernel metadata.
    """

    def __init__(self) -> None:
        """Initializes DB engine, session factory, and ensures schema exists."""
        # pylint: disable=invalid-name
        self.engine, self.SessionLocal = create_engine_and_session(
            "vllm-legacy"
        )  # pylint: disable=invalid-name
        self._ensure_schema()
        log.info("Legacy vLLM Database service interface initialized successfully.")

    def estimate_space(self, hashes: Iterable[str], f_ext: Set[str] | None) -> int:
        """Sum the sizes of artefacts that would be deleted."""
        size = 0
        session = self.get_session()
        try:
            q = session.query(func.sum(VllmLegacyKernelFileOrm.size)).filter(
                VllmLegacyKernelFileOrm.triton_cache_key.in_(hashes)
            )

            if f_ext:
                q = q.filter(
                    or_(
                        *[VllmLegacyKernelFileOrm.rel_path.like(f"%{ext}") for ext in IR_EXTS]
                    )
                )

            size = q.scalar() or 0
        finally:
            session.close()
        return size

    def _ensure_schema(self) -> None:
        """Ensures database schema (tables, indexes) exists."""
        try:
            Base.metadata.create_all(bind=self.engine)
            log.info("Database schema verified/created at %s.", DB_PATH)
        except Exception as e:  # pylint: disable=broad-except
            log.error("Fatal error creating database schema: %s", e, exc_info=True)
            raise

    def get_session(self) -> SqlaSession:
        """Returns a new database session."""
        return self.SessionLocal()

    def insert_kernel(
        self, k_data: Kernel, cache_dir: str, vllm_hash: str, rank_x_y: str
    ) -> None:
        """Upserts a legacy vLLM kernel and its associated files into the database."""
        session = self.get_session()
        try:
            VllmLegacyKernelOrm.upsert_from_dto(
                session, k_data, cache_dir, vllm_hash, rank_x_y
            )
            session.commit()
            log.info(
                "Legacy vLLM Kernel %s with cache_dir %s vllm_hash %s and "
                "rank_x_y %s upserted into DB.",
                k_data.hash,
                cache_dir,
                vllm_hash,
                rank_x_y,
            )
        except exc.IntegrityError as e:
            session.rollback()
            log.error(
                "Failed to upsert legacy vLLM kernel %s with cache_dir %s "
                "vllm_hash %s and rank_x_y %s due to a constraint violation: %s",
                k_data.hash,
                cache_dir,
                vllm_hash,
                rank_x_y,
                e,
                exc_info=True,
            )
            raise
        except exc.OperationalError as e:
            session.rollback()
            log.error(
                "Failed to upsert legacy vLLM kernel %s with cache_dir %s "
                "vllm_hash %s and rank_x_y %s due to a db operation issue: %s",
                k_data.hash,
                cache_dir,
                vllm_hash,
                rank_x_y,
                e,
                exc_info=True,
            )
            raise
        except Exception:  # pylint: disable=broad-except
            session.rollback()
            log.error(
                "DB Error: Failed to upsert legacy vLLM kernel %s with cache_dir %s "
                "vllm_hash %s and rank_x_y %s",
                k_data.hash,
                cache_dir,
                vllm_hash,
                rank_x_y,
                exc_info=True,
            )
            raise
        finally:
            session.close()

    def find_duplicate_kernels(self) -> List[List[Dict[str, Any]]]:
        """Finds and groups duplicate vLLM kernels."""
        return db_base.Database.find_duplicates_generic(
            self.SessionLocal,
            VllmLegacyKernelOrm,
            hash_field="triton_cache_key",
            additional_fields=["vllm_hash"],
        )

    def delete_by_hash(
        self, hashes: List[str] | Set[str], ir_only: bool = False
    ) -> int:
        """
        Deletes kernel or IR files from the database.

        Args:
            hashes: List/set of triton cache keys to delete
            ir_only: If True, only delete IR files; else delete entire kernels

        Returns:
            Number of kernels or files deleted
        """
        if not hashes:
            return 0

        session = self.get_session()
        try:
            if ir_only:
                # Delete only IR files
                query = session.query(VllmLegacyKernelFileOrm).filter(
                    VllmLegacyKernelFileOrm.triton_cache_key.in_(hashes),
                    or_(
                        *[
                            VllmLegacyKernelFileOrm.rel_path.like(f"%{ext}")
                            for ext in IR_EXTS
                        ]
                    ),
                )
                count = query.count()
                query.delete(synchronize_session="fetch")
            else:
                # Delete entire kernels
                query = session.query(VllmLegacyKernelOrm).filter(
                    VllmLegacyKernelOrm.triton_cache_key.in_(hashes)
                )
                count = query.count()
                query.delete(synchronize_session="fetch")

            session.commit()
            log.info("Deleted %d %s from database", count, "files" if ir_only else "kernels")
            return count
        finally:
            session.close()

    def close(self) -> None:
        """Closes the database engine's connection pool."""
        if self.engine:
            self.engine.dispose()
            log.info("Database engine connection pool disposed.")