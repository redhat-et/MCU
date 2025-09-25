"""
Legacy vLLM Database module for managing kernel metadata.
"""

from typing import List, Set, Dict, Any, Iterable, Tuple
import logging
from sqlalchemy import exc, or_, func, and_, tuple_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .db_config import create_engine_and_session, DB_PATH
from .db_models import (
    Base,
    VllmLegacyKernelOrm,
    VllmLegacyKernelFileOrm,
    SqlaSession,
)
from ..models.kernel import Kernel
from ..models.criteria import SearchCriteria
from ..utils.mcm_constants import IR_EXTS
from ..utils.utils import build_common_search_filters
from . import database_utils
from .database_utils import create_file_orm_dict

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
        database_utils.ensure_schema(self.engine)

    def get_session(self) -> SqlaSession:
        """Returns a new database session."""
        return self.SessionLocal()

    def insert_kernel(
        self, k_data: Kernel, cache_dir: str, vllm_hash: str, rank_x_y: str
    ) -> None:
        """Upserts a legacy vLLM kernel and its associated files into the database."""
        session = self.get_session()

        def operation():
            VllmLegacyKernelOrm.upsert_from_dto(
                session, k_data, cache_dir, vllm_hash, rank_x_y
            )

        database_utils.handle_kernel_insert(
            session, operation, k_data, cache_dir, vllm_hash, rank_x_y,
            error_prefix="Legacy vLLM "
        )

    def search(self, criteria: SearchCriteria) -> List[Dict[str, Any]]:
        """
        Searches for legacy vLLM kernels matching criteria.

        Args:
            criteria: `SearchCriteria` object with filter values.

        Returns:
            A list of dictionaries, each representing a matching kernel.
        """
        session = self.get_session()
        try:
            query = session.query(VllmLegacyKernelOrm)
            equality_filter_configs = [
                ("cache_dir", VllmLegacyKernelOrm.cache_dir, str),
                ("name", VllmLegacyKernelOrm.name, None),
                ("backend", VllmLegacyKernelOrm.backend, None),
                ("arch", VllmLegacyKernelOrm.arch, str),
            ]

            active_filters = build_common_search_filters(
                criteria, VllmLegacyKernelOrm, equality_filter_configs
            )

            if active_filters:
                query = query.filter(and_(*active_filters))
            query = query.order_by(VllmLegacyKernelOrm.modified_time.desc())
            results_orm = query.all()
            log.debug(
                "Legacy vLLM DB Search: Found %d results for criteria: %s.",
                len(results_orm),
                criteria,
            )

            results = [
                {
                    "hash": r.triton_cache_key,
                    "vllm_hash": r.vllm_hash,
                    "rank_x_y": r.rank_x_y,
                    "cache_dir": r.cache_dir,
                    "name": r.name,
                    "backend": r.backend,
                    "arch": r.arch,
                    "modified_time": r.modified_time,
                    "created_time": r.created,  # Use 'created' instead of 'created_time'
                    "total_size": r.total_size,
                }
                for r in results_orm
            ]

            return results
        finally:
            session.close()

    def find_duplicate_kernels(self) -> List[List[Dict[str, Any]]]:
        """Finds and groups duplicate vLLM kernels."""
        return database_utils.find_duplicates_generic(
            self.SessionLocal,
            VllmLegacyKernelOrm,
            hash_field="triton_cache_key",
            additional_fields=["vllm_hash"],
        )

    def _prepare_legacy_batch_data(
        self, batch: List[Tuple[Kernel, str, str, str]]
    ) -> Tuple[List, List]:
        """Prepare legacy vLLM kernel IDs and file values for batch processing."""
        kernel_ids_to_clear = []
        file_values_list = []

        for k_data, cache_dir, vllm_hash, rank_x_y in batch:
            kernel_ids_to_clear.append((cache_dir, vllm_hash, k_data.hash, rank_x_y))

            # Collect file values for bulk insert
            for f_dto in k_data.files:
                file_values_list.append(
                    create_file_orm_dict(
                        cache_dir, vllm_hash, k_data.hash, rank_x_y, f_dto
                    )
                )

        return kernel_ids_to_clear, file_values_list

    def _upsert_legacy_kernel(
        self,
        session: SqlaSession,
        kernel_info: Tuple[Kernel, str, str, str],
    ) -> None:
        """Upsert a single legacy vLLM kernel.

        Args:
            session: Database session
            kernel_info: Tuple of (k_data, cache_dir, vllm_hash, rank_x_y)
        """
        k_data, cache_dir, vllm_hash, rank_x_y = kernel_info
        kernel_values = VllmLegacyKernelOrm.get_vllm_kernel_values(
            k_data, cache_dir, vllm_hash, rank_x_y
        )

        stmt = sqlite_insert(VllmLegacyKernelOrm).values(kernel_values)
        update_dict = {
            col.name: getattr(stmt.excluded, col.name)
            for col in VllmLegacyKernelOrm.__table__.columns
            if col.name not in ("cache_dir", "vllm_hash", "triton_cache_key", "rank_x_y")
        }
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=[
                    "cache_dir",
                    "vllm_hash",
                    "triton_cache_key",
                    "rank_x_y",
                ],
                set_=update_dict,
            )
        )

    def bulk_insert_kernels(
        self, kernels_data: List[Tuple[Kernel, str, str, str]], batch_size: int = 1000
    ) -> int:
        """
        Bulk insert multiple legacy vLLM kernels efficiently.

        Args:
            kernels_data: List of tuples containing (Kernel, cache_dir, vllm_hash, rank_x_y)
            batch_size: Number of kernels to insert per transaction

        Returns:
            Number of kernels inserted
        """
        if not kernels_data:
            log.info("No legacy vLLM kernels to insert")
            return 0

        session = self.get_session()
        inserted_count = 0

        try:
            for i in range(0, len(kernels_data), batch_size):
                batch = kernels_data[i : i + batch_size]

                # Prepare batch data
                kernel_ids_to_clear, file_values_list = self._prepare_legacy_batch_data(
                    batch
                )

                # Delete all files for batch kernels in one query
                if kernel_ids_to_clear:
                    session.query(VllmLegacyKernelFileOrm).filter(
                        tuple_(
                            VllmLegacyKernelFileOrm.cache_dir,
                            VllmLegacyKernelFileOrm.vllm_hash,
                            VllmLegacyKernelFileOrm.triton_cache_key,
                            VllmLegacyKernelFileOrm.rank_x_y,
                        ).in_(kernel_ids_to_clear)
                    ).delete(synchronize_session=False)

                # Insert/update kernels
                for kernel_info in batch:
                    self._upsert_legacy_kernel(session, kernel_info)
                    inserted_count += 1

                # Bulk insert all files at once
                if file_values_list:
                    session.bulk_insert_mappings(VllmLegacyKernelFileOrm, file_values_list)

                session.commit()
                log.info(
                    "Batch of %d legacy vLLM kernels committed (%d total so far)",
                    len(batch),
                    inserted_count,
                )

        except Exception as e:
            session.rollback()
            log.error("Legacy vLLM bulk insert failed: %s", e, exc_info=True)
            raise
        finally:
            session.close()

        return inserted_count

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
