# pylint: disable=c-extension-no-member
"""
Database service class for managing kernel metadata.

This module provides the `Database` class, which acts as a high-level API
for interacting with the kernel cache database. It uses ORM models (SqlAlchemy).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Set, Iterable, Tuple

from sqlalchemy import and_, exc, or_, func, tuple_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .db_config import engine, SessionLocal, create_engine_and_session
from .db_models import (
    KernelOrm,
    KernelFileOrm,
    VllmKernelOrm,
    VllmKernelFileOrm,
    HelionKernelOrm,
    HelionKernelFileOrm,
    SqlaSession,
)

from ..models.criteria import SearchCriteria
from ..models.kernel import Kernel, VllmKernelMetadata
from ..utils.mcm_constants import IR_EXTS
from ..utils.utils import build_common_search_filters
from . import database_utils
from .database_utils import create_file_orm_dict

# Batch item format constants for vLLM kernels
# (kernel, cache_dir, vllm_hash, rank_x_y)
VLLM_LEGACY_BATCH_ITEM_LENGTH = 4
# (kernel, cache_dir, vllm_hash, rank_x_y, artifact_compile_range, best_config, triton_subpath)
VLLM_NEW_BATCH_ITEM_LENGTH = 7

# Kernel ID tuple length constants for database operations
# (cache_dir, vllm_hash, triton_hash, rank_x_y)
VLLM_LEGACY_KERNEL_ID_LENGTH = 4
# (cache_dir, vllm_hash, triton_hash, rank_x_y, artifact_compile_range)
VLLM_NEW_KERNEL_ID_LENGTH = 5

# Best config field names
BEST_CONFIG_TRITON_HASH_KEY = "triton_cache_hash"
KERNEL_DICT_IS_BEST_KEY = "is_best"

log = logging.getLogger(__name__)


class Database:
    """
    Manages database interactions for kernel metadata.
    """

    def __init__(self) -> None:
        """Initializes DB engine, session factory, and ensures schema exists."""
        self.engine = engine
        self.SessionLocal = SessionLocal  # pylint: disable=invalid-name
        self._ensure_schema()
        log.info("Database service interface initialized successfully.")

    def _ensure_schema(self) -> None:
        """Ensures database schema (tables, indexes) exists."""
        database_utils.ensure_schema(self.engine)

    def get_session(self) -> SqlaSession:
        """Returns a new database session."""
        return self.SessionLocal()

    def insert_kernel(self, k_data: Kernel, cache_dir: str) -> None:
        """
        Upserts a kernel and its associated files into the database.

        Args:
            k_data: A `Kernel` DTO containing the metadata.
        """
        session = self.get_session()
        k_data.cache_dir = str(cache_dir)
        try:
            KernelOrm.upsert_from_dto(session, k_data)
            session.commit()
            log.info(
                "Kernel %s with cache_dir %s and its files upserted into DB.",
                k_data.hash,
                k_data.cache_dir,
            )
        except exc.IntegrityError as e:
            session.rollback()
            log.error(
                "Failed to upsert kernel %s with cache_dir %s due to a constraint violation: %s",
                k_data.hash,
                k_data.cache_dir,
                e,
                exc_info=True,
            )
            raise
        except exc.OperationalError as e:
            session.rollback()
            log.error(
                "Failed to upsert kernel %s with cache_dir %s due to a db operation issue: %s",
                k_data.hash,
                k_data.cache_dir,
                e,
                exc_info=True,
            )
            raise
        except Exception:  # pylint: disable=broad-except
            session.rollback()
            log.error(
                "DB Error: Failed to upsert kernel %s with cache_dir %s.",
                k_data.hash,
                k_data.cache_dir,
                exc_info=True,
            )
            raise
        finally:
            session.close()

    def _prepare_batch_data(self, batch: List[Tuple[Kernel, str]]) -> Tuple[List, List]:
        """Prepare kernel IDs and file values for batch processing."""
        kernel_ids_to_clear = []
        file_values_list = []

        for k_data, cache_dir in batch:
            k_data.cache_dir = str(cache_dir)
            kernel_ids_to_clear.append((k_data.hash, str(cache_dir)))

            # Collect file values for bulk insert
            for f_dto in k_data.files:
                file_values_list.append(
                    {
                        "kernel_hash": k_data.hash,
                        "kernel_cache_dir": str(cache_dir),
                        "type": f_dto.file_type,
                        "rel_path": f_dto.path.name,
                        "size": f_dto.size,
                    }
                )

        return kernel_ids_to_clear, file_values_list

    def _upsert_kernel(
        self, session: SqlaSession, k_data: Kernel, cache_dir: str
    ) -> None:
        """Upsert a single kernel."""
        kernel_values = KernelOrm.get_common_kernel_values(k_data)
        kernel_values.update(
            {
                "hash": k_data.hash,
                "cache_dir": str(cache_dir),
            }
        )

        stmt = sqlite_insert(KernelOrm).values(kernel_values)
        # Preserve runtime statistics during updates
        # These fields should not be overwritten during re-indexing
        preserved_fields = {"runtime_hits", "last_access_time"}
        update_dict = {
            col.name: getattr(stmt.excluded, col.name)
            for col in KernelOrm.__table__.columns
            if col.name not in ("hash", "cache_dir")
            and col.name not in preserved_fields
        }
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=["hash", "cache_dir"], set_=update_dict
            )
        )

    def bulk_insert_kernels(
        self, kernels_data: List[Tuple[Kernel, str]], batch_size: int = 1000
    ) -> int:
        """
        Bulk insert multiple kernels efficiently.

        Args:
            kernels_data: List of tuples containing (Kernel, cache_dir)
            batch_size: Number of kernels to insert per transaction

        Returns:
            Number of kernels inserted
        """
        if not kernels_data:
            log.info("No kernels to insert")
            return 0

        session = self.get_session()
        inserted_count = 0

        # pylint: disable=duplicate-code  # Similar pattern in vllm_legacy_database.py
        try:
            for i in range(0, len(kernels_data), batch_size):
                batch = kernels_data[i : i + batch_size]

                # Prepare batch data
                kernel_ids_to_clear, file_values_list = self._prepare_batch_data(batch)

                # Delete all files for batch kernels in one query
                if kernel_ids_to_clear:
                    session.query(KernelFileOrm).filter(
                        tuple_(
                            KernelFileOrm.kernel_hash, KernelFileOrm.kernel_cache_dir
                        ).in_(kernel_ids_to_clear)
                    ).delete(synchronize_session="evaluate")

                # Insert/update kernels
                for k_data, cache_dir in batch:
                    self._upsert_kernel(session, k_data, cache_dir)
                    inserted_count += 1

                # Bulk insert all files at once
                if file_values_list:
                    session.bulk_insert_mappings(KernelFileOrm, file_values_list)

                session.commit()
                log.info(
                    "Batch of %d kernels committed (%d total so far)",
                    len(batch),
                    inserted_count,
                )

        except Exception as e:
            session.rollback()
            log.error("Bulk insert failed: %s", e, exc_info=True)
            raise
        finally:
            session.close()
        return inserted_count

    def search(self, criteria: SearchCriteria) -> List[Dict[str, Any]]:
        """
        Searches for kernels matching criteria.

        Args:
            criteria: `SearchCriteria` object with filter values.

        Returns:
            A list of dictionaries, each representing a matching kernel.
        """
        session = self.get_session()
        try:
            query = session.query(KernelOrm)
            equality_filter_configs = [
                ("cache_dir", KernelOrm.cache_dir, str),
                ("name", KernelOrm.name, None),
                ("backend", KernelOrm.backend, None),
                ("arch", KernelOrm.arch, str),
            ]

            active_filters = build_common_search_filters(
                criteria, KernelOrm, equality_filter_configs
            )

            if active_filters:
                query = query.filter(and_(*active_filters))
            query = query.order_by(KernelOrm.modified_time.desc())
            results_orm = query.all()
            log.debug(
                "DB Search: Found %d results for criteria: %s.",
                len(results_orm),
                criteria,
            )
            return [kernel_orm.to_dict() for kernel_orm in results_orm]
        except Exception:  # pylint: disable=broad-except
            log.error("DB Search: Failed for criteria %s.", criteria, exc_info=True)
            return []
        finally:
            session.close()

    # pylint: disable=too-many-locals
    # we'll change this logic
    def find_duplicates(self) -> List[List[Dict[str, Any]]]:
        """
        Finds groups of duplicate kernels.
        1. Kernels are grouped by 'name' and 'total_size'.
        2. Within each name-group, kernels are duplicates if their 'kernel_metadata_json'
           objects meet the criteria defined in _are_kernel_metadata_jsons_duplicates
           (identical or differ only in an internal 'hash' field).
        Returns a list of lists, where each inner list contains dictionaries of duplicate kernels,
        sorted by 'modified_time' (oldest first).
        """
        return database_utils.find_duplicates_generic(
            self.SessionLocal, KernelOrm, "hash"
        )

    def estimate_space(self, hashes: Iterable[str], f_ext: Set[str] | None) -> int:
        """Sum the sizes of artefacts that would be deleted."""
        size = 0
        with self.get_session() as s:
            q = s.query(func.sum(KernelFileOrm.size)).filter(
                KernelFileOrm.kernel_hash.in_(hashes)
            )

            if f_ext:
                q = q.filter(
                    or_(*[KernelFileOrm.rel_path.like(f"%{ext}") for ext in IR_EXTS])
                )

            size = q.scalar() or 0
        return size

    def close(self) -> None:
        """Closes the database engine's connection pool."""
        if self.engine:
            self.engine.dispose()
            log.info("Database engine connection pool disposed.")


class VllmDatabase:
    """
    Manages database interactions for new vLLM kernel metadata.
    """

    def __init__(self) -> None:
        """Initializes DB engine, session factory, and ensures schema exists."""
        # pylint: disable=invalid-name
        self.engine, self.SessionLocal = create_engine_and_session(
            "vllm"
        )  # pylint: disable=invalid-name
        self._ensure_schema()
        log.info("New vLLM Database service interface initialized successfully.")

    def estimate_space(self, hashes: Iterable[str], f_ext: Set[str] | None) -> int:
        """Sum the sizes of artefacts that would be deleted."""
        size = 0
        with self.get_session() as s:
            q = s.query(func.sum(VllmKernelFileOrm.size)).filter(
                VllmKernelFileOrm.triton_cache_key.in_(hashes)
            )

            if f_ext:
                q = q.filter(
                    or_(
                        *[VllmKernelFileOrm.rel_path.like(f"%{ext}") for ext in IR_EXTS]
                    )
                )

            size = q.scalar() or 0
        return size

    def _ensure_schema(self) -> None:
        """Ensures database schema (tables, indexes) exists."""
        database_utils.ensure_schema(self.engine)

    def get_session(self) -> SqlaSession:
        """Returns a new database session."""
        return self.SessionLocal()

    def insert_kernel(
        self,
        k_data: Kernel,
        cache_dir: str,
        vllm_meta: VllmKernelMetadata,
    ) -> None:
        """
        Upserts a new vLLM kernel and its associated files into the database.

        Args:
            k_data: A `Kernel` DTO containing the metadata.
            cache_dir: Root path of the vLLM cache
            vllm_meta: vLLM-specific metadata (hash, rank, artifact_compile_range, etc.)
        """
        session = self.get_session()

        def operation():
            VllmKernelOrm.upsert_from_dto(
                session,
                k_data,
                cache_dir,
                vllm_meta,
            )

        extra_args = {"artifact_compile_range": vllm_meta.artifact_compile_range}
        if vllm_meta.best_config:
            extra_args["best_config"] = vllm_meta.best_config

        context = database_utils.KernelInsertContext(
            kernel_data=k_data,
            cache_dir=cache_dir,
            vllm_hash=vllm_meta.vllm_hash,
            rank_x_y=vllm_meta.rank_x_y,
            extra_args=extra_args,
            error_prefix="New vLLM ",
        )
        database_utils.handle_kernel_insert(session, operation, context)

    def _prepare_vllm_batch_data(self, batch: List[Tuple]) -> Tuple[List, List]:
        """Prepare vLLM kernel IDs and file values for batch processing.

        Handles both old format (4 values) and new format
        (6 values with artifact_compile_range and best_config).
        """
        kernel_ids_to_clear = []
        file_values_list = []

        for item in batch:
            # Handle both legacy and new vLLM batch formats
            if len(item) == VLLM_LEGACY_BATCH_ITEM_LENGTH:
                # Legacy format: (kernel, cache_dir, vllm_hash, rank_x_y)
                k_data, cache_dir, vllm_hash, rank_x_y = item
                artifact_compile_range = None
            elif len(item) == VLLM_NEW_BATCH_ITEM_LENGTH:
                # New format: (kernel, cache_dir, vllm_hash, rank_x_y,
                # artifact_compile_range, best_config, triton_subpath)
                (
                    k_data,
                    cache_dir,
                    vllm_hash,
                    rank_x_y,
                    artifact_compile_range,
                    _best_config,
                    _triton_subpath,
                ) = item
            else:
                raise ValueError(
                    f"Expected {VLLM_LEGACY_BATCH_ITEM_LENGTH} or "
                    f"{VLLM_NEW_BATCH_ITEM_LENGTH} values in batch item, "
                    f"got {len(item)}"
                )

            # For new structure, we need to include artifact_compile_range in the key
            if artifact_compile_range is not None:
                kernel_ids_to_clear.append(
                    (
                        cache_dir,
                        vllm_hash,
                        k_data.hash,
                        rank_x_y,
                        artifact_compile_range,
                    )
                )
            else:
                kernel_ids_to_clear.append(
                    (cache_dir, vllm_hash, k_data.hash, rank_x_y)
                )

            # Collect file values for bulk insert
            for f_dto in k_data.files:
                file_val = create_file_orm_dict(
                    cache_dir, vllm_hash, k_data.hash, rank_x_y, f_dto
                )
                if artifact_compile_range is not None:
                    file_val["artifact_compile_range"] = artifact_compile_range
                file_values_list.append(file_val)

        return kernel_ids_to_clear, file_values_list

    # pylint: disable=too-many-locals
    def _upsert_vllm_kernel(
        self,
        session: SqlaSession,
        kernel_info: Tuple,
    ) -> None:
        """Upsert a single vLLM kernel.

        Args:
            session: Database session
            kernel_info: Tuple of (k_data, cache_dir, vllm_hash, rank_x_y) or
                    (k_data, cache_dir, vllm_hash, rank_x_y, artifact_compile_range,
                     best_config, triton_subpath)
        """
        # Handle both legacy and new vLLM formats
        if len(kernel_info) == VLLM_LEGACY_BATCH_ITEM_LENGTH:
            k_data, cache_dir, vllm_hash, rank_x_y = kernel_info
            artifact_compile_range = ""
            best_config = None
            triton_subpath = None
        elif len(kernel_info) == VLLM_NEW_BATCH_ITEM_LENGTH:
            (
                k_data,
                cache_dir,
                vllm_hash,
                rank_x_y,
                artifact_compile_range,
                best_config,
                triton_subpath,
            ) = kernel_info
        else:
            raise ValueError(
                f"Expected {VLLM_LEGACY_BATCH_ITEM_LENGTH} or "
                f"{VLLM_NEW_BATCH_ITEM_LENGTH} values in kernel_info, "
                f"got {len(kernel_info)}"
            )

        # Create VllmKernelMetadata object
        vllm_meta = VllmKernelMetadata(
            vllm_hash=vllm_hash,
            rank_x_y=rank_x_y,
            artifact_compile_range=artifact_compile_range or "",
            best_config=best_config,
            triton_subpath=triton_subpath,
        )
        kernel_values = VllmKernelOrm.get_vllm_kernel_values(
            k_data, cache_dir, vllm_meta
        )

        stmt = sqlite_insert(VllmKernelOrm).values(kernel_values)
        # For new vLLM structure, artifact_compile_range is part of the primary key
        primary_key_fields = [
            "cache_dir",
            "vllm_hash",
            "triton_cache_key",
            "rank_x_y",
            "artifact_compile_range",  # Always included in primary key for VllmKernelOrm
        ]
        # Preserve runtime statistics during updates
        # These fields should not be overwritten during re-indexing
        preserved_fields = {"runtime_hits", "last_access_time"}
        update_dict = {
            col.name: getattr(stmt.excluded, col.name)
            for col in VllmKernelOrm.__table__.columns
            if col.name not in primary_key_fields and col.name not in preserved_fields
        }
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=primary_key_fields,
                set_=update_dict,
            )
        )

    def bulk_insert_kernels(
        self, kernels_data: List[Tuple], batch_size: int = 1000
    ) -> int:
        """
        Bulk insert multiple vLLM kernels efficiently.

        Args:
            kernels_data: List of tuples containing either:
                - (Kernel, cache_dir, vllm_hash, rank_x_y) for legacy format or
                - (Kernel, cache_dir, vllm_hash, rank_x_y, artifact_compile_range,
                  best_config) for new format
            batch_size: Number of kernels to insert per transaction

        Returns:
            Number of kernels inserted
        """
        if not kernels_data:
            log.info("No vLLM kernels to insert")
            return 0

        session = self.get_session()
        inserted_count = 0

        try:
            for i in range(0, len(kernels_data), batch_size):
                batch = kernels_data[i : i + batch_size]

                # Prepare batch data
                kernel_ids_to_clear, file_values_list = self._prepare_vllm_batch_data(
                    batch
                )

                # Delete all files for batch kernels in one query
                if kernel_ids_to_clear:
                    # Check if we have artifact_compile_range in the data
                    has_artifact_compile_range = (
                        len(kernel_ids_to_clear[0]) == VLLM_NEW_KERNEL_ID_LENGTH
                    )
                    if kernel_ids_to_clear and has_artifact_compile_range:
                        # New structure with artifact_compile_range
                        session.query(VllmKernelFileOrm).filter(
                            tuple_(
                                VllmKernelFileOrm.cache_dir,
                                VllmKernelFileOrm.vllm_hash,
                                VllmKernelFileOrm.triton_cache_key,
                                VllmKernelFileOrm.rank_x_y,
                                VllmKernelFileOrm.artifact_compile_range,
                            ).in_(kernel_ids_to_clear)
                        ).delete(synchronize_session=False)
                    else:
                        # Legacy structure without artifact_compile_range
                        session.query(VllmKernelFileOrm).filter(
                            tuple_(
                                VllmKernelFileOrm.cache_dir,
                                VllmKernelFileOrm.vllm_hash,
                                VllmKernelFileOrm.triton_cache_key,
                                VllmKernelFileOrm.rank_x_y,
                            ).in_(kernel_ids_to_clear)
                        ).delete(synchronize_session=False)

                # Insert/update kernels
                for kernel_info in batch:
                    self._upsert_vllm_kernel(session, kernel_info)
                    inserted_count += 1

                # Bulk insert all files at once
                if file_values_list:
                    session.bulk_insert_mappings(VllmKernelFileOrm, file_values_list)

                session.commit()
                log.info(
                    "Batch of %d vLLM kernels committed (%d total so far)",
                    len(batch),
                    inserted_count,
                )

        except Exception as e:
            session.rollback()
            log.error("Bulk insert failed: %s", e, exc_info=True)
            raise
        finally:
            session.close()
        return inserted_count

    def search(self, criteria: SearchCriteria) -> List[Dict[str, Any]]:
        """
        Searches for vLLM kernels matching criteria.

        Args:
            criteria: `SearchCriteria` object with filter values.

        Returns:
            A list of dictionaries, each representing a matching kernel.
        """
        session = self.get_session()
        try:
            query = session.query(VllmKernelOrm)
            equality_filter_configs = [
                ("cache_dir", VllmKernelOrm.cache_dir, str),
                ("name", VllmKernelOrm.name, None),
                ("backend", VllmKernelOrm.backend, None),
                ("arch", VllmKernelOrm.arch, str),
            ]

            active_filters = build_common_search_filters(
                criteria, VllmKernelOrm, equality_filter_configs
            )

            if active_filters:
                query = query.filter(and_(*active_filters))
            query = query.order_by(VllmKernelOrm.modified_time.desc())
            results_orm = query.all()
            log.debug(
                "vLLM DB Search: Found %d results for criteria: %s.",
                len(results_orm),
                criteria,
            )

            results = []

            # Cache for parsed best_config data to avoid repeated parsing
            best_config_cache = {}

            for kernel_orm in results_orm:
                kernel_dict = kernel_orm.to_dict()
                # Check if this kernel is the best one by comparing
                # triton_cache_key with triton_cache_hash in best_config
                is_best = False
                if kernel_orm.best_config:
                    # Use cached parsed result if available
                    if kernel_orm.best_config not in best_config_cache:
                        try:
                            best_config_data = json.loads(kernel_orm.best_config)
                            triton_cache_hash = best_config_data.get(
                                BEST_CONFIG_TRITON_HASH_KEY
                            )
                            best_config_cache[kernel_orm.best_config] = (
                                triton_cache_hash
                            )
                        except (json.JSONDecodeError, TypeError):
                            best_config_cache[kernel_orm.best_config] = None

                    triton_cache_hash = best_config_cache[kernel_orm.best_config]
                    is_best = kernel_orm.triton_cache_key == triton_cache_hash

                # Filter based on only_best criteria:
                # - None: show all kernels (no filtering)
                # - True: show only best kernels
                # - False: show only non-best kernels
                if criteria.only_best is True and not is_best:
                    continue
                if criteria.only_best is False and is_best:
                    continue

                kernel_dict[KERNEL_DICT_IS_BEST_KEY] = is_best
                results.append(kernel_dict)

            log.debug(
                "vLLM DB Search: Returning %d results after filtering.",
                len(results),
            )
            return results
        except Exception:  # pylint: disable=broad-except
            log.error(
                "vLLM DB Search: Failed for criteria %s.", criteria, exc_info=True
            )
            return []
        finally:
            session.close()

    def find_duplicates(self) -> List[List[Dict[str, Any]]]:
        """
        Finds groups of duplicate vLLM kernels.
        1. Kernels are grouped by 'name' and 'total_size'.
        2. Within each name-group, kernels are duplicates if their 'kernel_metadata_json'
           objects meet the criteria defined in _are_kernel_metadata_jsons_duplicates
           (identical or differ only in an internal 'hash' field).
        Returns a list of lists, where each inner list contains dictionaries of duplicate kernels,
        sorted by 'modified_time' (oldest first).
        """
        # Use the static generic method from Database class
        return database_utils.find_duplicates_generic(
            self.SessionLocal, VllmKernelOrm, "triton_cache_key", ["vllm_hash"]
        )

    def close(self) -> None:
        """Closes the database engine's connection pool."""
        if self.engine:
            self.engine.dispose()
            log.info("vLLM Database engine connection pool disposed.")


HELION_BATCH_ITEM_LENGTH = 5  # (kernel, cache_dir, helion_hash, best_config, is_best)


class HelionDatabase:
    """Manages database interactions for Helion kernel metadata."""

    def __init__(self) -> None:
        """Initializes DB engine, session factory, and ensures schema exists."""
        self.engine, self.SessionLocal = create_engine_and_session(  # pylint: disable=invalid-name
            "helion"
        )
        self._ensure_schema()
        log.info("Helion Database service interface initialized successfully.")

    def _ensure_schema(self) -> None:
        """Ensures database schema (tables, indexes) exists."""
        database_utils.ensure_schema(self.engine)

    def get_session(self) -> SqlaSession:
        """Returns a new database session."""
        return self.SessionLocal()

    def insert_kernel(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        k_data: Kernel,
        cache_dir: str,
        helion_hash: str | None,
        best_config: str | None,
        is_best: bool,
    ) -> None:
        """Upserts a Helion kernel and its associated files into the database."""
        session = self.get_session()
        try:
            HelionKernelOrm.upsert_from_dto(
                session, k_data, cache_dir, helion_hash, best_config, is_best
            )
            session.commit()
            log.info(
                "Helion kernel %s with cache_dir %s upserted into DB.",
                k_data.hash,
                cache_dir,
            )
        except exc.IntegrityError as e:
            session.rollback()
            log.error(
                "Failed to upsert Helion kernel %s: %s",
                k_data.hash, e, exc_info=True,
            )
            raise
        except exc.OperationalError as e:
            session.rollback()
            log.error(
                "Failed to upsert Helion kernel %s: %s",
                k_data.hash, e, exc_info=True,
            )
            raise
        except Exception:
            session.rollback()
            log.error(
                "DB Error: Failed to upsert Helion kernel %s.",
                k_data.hash, exc_info=True,
            )
            raise
        finally:
            session.close()

    def _upsert_helion_kernel(
        self, session: SqlaSession, kernel_info: tuple
    ) -> None:
        """Upsert a single Helion kernel."""
        k_data, cache_dir, helion_hash, best_config, is_best = kernel_info
        kernel_values = HelionKernelOrm.get_helion_kernel_values(
            k_data, cache_dir, helion_hash, best_config, is_best
        )

        stmt = sqlite_insert(HelionKernelOrm).values(kernel_values)
        preserved_fields = {"runtime_hits", "last_access_time"}
        update_dict = {
            col.name: getattr(stmt.excluded, col.name)
            for col in HelionKernelOrm.__table__.columns
            if col.name not in ("triton_cache_key", "cache_dir")
            and col.name not in preserved_fields
        }
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=["triton_cache_key", "cache_dir"],
                set_=update_dict,
            )
        )

    def bulk_insert_kernels(
        self, kernels_data: List[Tuple], batch_size: int = 1000
    ) -> int:
        """Bulk insert multiple Helion kernels efficiently."""
        if not kernels_data:
            log.info("No Helion kernels to insert")
            return 0

        session = self.get_session()
        inserted_count = 0

        try:
            for i in range(0, len(kernels_data), batch_size):
                batch = kernels_data[i : i + batch_size]

                kernel_ids_to_clear = []
                file_values_list = []
                for item in batch:
                    k_data, cache_dir = item[0], item[1]
                    kernel_ids_to_clear.append((k_data.hash, str(cache_dir)))
                    for f_dto in k_data.files:
                        file_values_list.append(
                            {
                                "triton_cache_key": k_data.hash,
                                "cache_dir": str(cache_dir),
                                "type": f_dto.file_type,
                                "rel_path": f_dto.path.name,
                                "size": f_dto.size,
                            }
                        )

                if kernel_ids_to_clear:
                    session.query(HelionKernelFileOrm).filter(
                        tuple_(
                            HelionKernelFileOrm.triton_cache_key,
                            HelionKernelFileOrm.cache_dir,
                        ).in_(kernel_ids_to_clear)
                    ).delete(synchronize_session="evaluate")

                for kernel_info in batch:
                    self._upsert_helion_kernel(session, kernel_info)
                    inserted_count += 1

                if file_values_list:
                    session.bulk_insert_mappings(
                        HelionKernelFileOrm, file_values_list
                    )

                session.commit()
                log.info(
                    "Batch of %d Helion kernels committed (%d total so far)",
                    len(batch),
                    inserted_count,
                )

        except Exception as e:
            session.rollback()
            log.error("Helion bulk insert failed: %s", e, exc_info=True)
            raise
        finally:
            session.close()
        return inserted_count

    def search(self, criteria: SearchCriteria) -> List[Dict[str, Any]]:
        """Searches for Helion kernels matching criteria."""
        session = self.get_session()
        try:
            query = session.query(HelionKernelOrm)
            equality_filter_configs = [
                ("cache_dir", HelionKernelOrm.cache_dir, str),
                ("name", HelionKernelOrm.name, None),
                ("backend", HelionKernelOrm.backend, None),
                ("arch", HelionKernelOrm.arch, str),
            ]

            active_filters = build_common_search_filters(
                criteria, HelionKernelOrm, equality_filter_configs
            )

            if active_filters:
                query = query.filter(and_(*active_filters))
            query = query.order_by(HelionKernelOrm.modified_time.desc())
            results_orm = query.all()

            results = []
            for kernel_orm in results_orm:
                kernel_dict = kernel_orm.to_dict()
                is_best = kernel_orm.is_best or False

                if criteria.only_best is True and not is_best:
                    continue
                if criteria.only_best is False and is_best:
                    continue

                kernel_dict[KERNEL_DICT_IS_BEST_KEY] = is_best
                results.append(kernel_dict)

            return results
        except Exception:  # pylint: disable=broad-except
            log.error(
                "Helion DB Search: Failed for criteria %s.",
                criteria, exc_info=True,
            )
            return []
        finally:
            session.close()

    def find_duplicates(self) -> List[List[Dict[str, Any]]]:
        """Finds groups of duplicate Helion kernels."""
        return database_utils.find_duplicates_generic(
            self.SessionLocal, HelionKernelOrm, "triton_cache_key"
        )

    def estimate_space(self, hashes: Iterable[str], f_ext: Set[str] | None) -> int:
        """Sum the sizes of artefacts that would be deleted."""
        size = 0
        with self.get_session() as s:
            q = s.query(func.sum(HelionKernelFileOrm.size)).filter(
                HelionKernelFileOrm.triton_cache_key.in_(hashes)
            )
            if f_ext:
                q = q.filter(
                    or_(
                        *[HelionKernelFileOrm.rel_path.like(f"%{ext}")
                          for ext in IR_EXTS]
                    )
                )
            size = q.scalar() or 0
        return size

    def close(self) -> None:
        """Closes the database engine's connection pool."""
        if self.engine:
            self.engine.dispose()
            log.info("Helion Database engine connection pool disposed.")
