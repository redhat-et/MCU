# pylint: disable=c-extension-no-member
"""
Database service class for managing Triton kernel metadata.

This module provides the `Database` class, which acts as a high-level API
for interacting with the kernel cache database. It uses ORM models (SqlAlchemy).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Iterable
import collections
from sqlalchemy import and_, exc, or_, func

from .db_config import engine, SessionLocal, DB_PATH
from .db_models import Base, KernelOrm, KernelFileOrm, SqlaSession

from ..models.criteria import SearchCriteria
from ..models.kernel import Kernel
from ..utils.tcm_constants import IR_EXTS

log = logging.getLogger(__name__)


class Database:
    """
    Manages database interactions for Triton kernel metadata.
    """

    def __init__(self) -> None:
        """Initializes DB engine, session factory, and ensures schema exists."""
        self.engine = engine
        self.SessionLocal = SessionLocal  # pylint: disable=invalid-name
        self._ensure_schema()
        log.info("Database service interface initialized successfully.")

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

    def insert_kernel(self, k_data: Kernel) -> None:
        """
        Upserts a kernel and its associated files into the database.

        Args:
            k_data: A `Kernel` DTO containing the metadata.
        """
        session = self.get_session()
        try:
            KernelOrm.upsert_from_dto(session, k_data)
            session.commit()
            log.info("Kernel %s and its files upserted into DB.", k_data.hash)
        except exc.IntegrityError as e:
            session.rollback()
            log.error(
                "Failed to upsert kernel %s due to a constraint violation: %s",
                k_data.hash,
                e,
                exc_info=True,
            )
            raise
        except exc.OperationalError as e:
            session.rollback()
            log.error(
                "Failed to upsert kernel %s due to a db operation issue: %s",
                k_data.hash,
                e,
                exc_info=True,
            )
            raise
        except Exception:  # pylint: disable=broad-except
            session.rollback()
            log.error(
                "DB Error: Failed to upsert kernel %s.", k_data.hash, exc_info=True
            )
            raise
        finally:
            session.close()

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
            active_filters = []

            equality_filter_configs = [
                ("name", KernelOrm.name, None),
                ("backend", KernelOrm.backend, None),
                ("arch", KernelOrm.arch, str),
            ]

            for crit_attr, orm_column, transformer in equality_filter_configs:
                value = getattr(criteria, crit_attr, None)
                if value is not None:
                    if transformer:
                        value = transformer(value)
                    active_filters.append(orm_column == value)

            if criteria.older_than_timestamp is not None:
                active_filters.append(
                    KernelOrm.modified_time < criteria.older_than_timestamp
                )

            if criteria.younger_than_timestamp is not None:
                active_filters.append(
                    KernelOrm.modified_time > criteria.younger_than_timestamp
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

    @staticmethod
    def _are_kernel_metadata_jsons_duplicates(metadata1: Any, metadata2: Any) -> bool:
        """
        Compares two kernel metadata dictionaries field by field.
        Kernels are considered duplicates if their metadata JSON objects:
        1. Are identical (0 differences).
        2. Differ *only* in a field named 'hash' within the JSON content
        (1 difference, specific to 'hash').
        They are NOT duplicates if they differ in 2 or more fields, or if they differ in 1 field
        that is NOT named 'hash'.
        """
        differences_count = 0
        hash_field_differed = False

        all_keys: Set[str] = set(metadata1.keys()) | set(metadata2.keys())

        for key in all_keys:
            value1 = metadata1.get(key)
            value2 = metadata2.get(key)

            if value1 != value2:
                differences_count += 1
                if key == "hash":
                    hash_field_differed = True

            if differences_count >= 2:
                return False

        if differences_count == 0:
            return True
        if differences_count == 1:
            return hash_field_differed

        return False

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
        session = self.get_session()
        try:
            all_kernels_data_from_orm = (
                session.query(
                    KernelOrm.hash,
                    KernelOrm.name,
                    KernelOrm.kernel_metadata_json,
                    KernelOrm.modified_time,
                    KernelOrm.backend,
                    KernelOrm.arch,
                    KernelOrm.triton_version,
                    KernelOrm.total_size,
                )
                .order_by(KernelOrm.modified_time.asc())
                .all()
            )

            if not all_kernels_data_from_orm:
                return []

            kernel_list_of_dicts: List[Dict[str, Any]] = [
                {
                    "hash": k.hash,
                    "name": k.name,
                    "metadata": k.kernel_metadata_json,
                    "modified_time": k.modified_time,
                    "backend": k.backend,
                    "arch": k.arch,
                    "triton_version": k.triton_version,
                    "total_size": k.total_size,
                }
                for k in all_kernels_data_from_orm
            ]

            kernels_grouped_by_name_and_size = collections.defaultdict(list)
            for kernel_dict in kernel_list_of_dicts:
                hash_val = kernel_dict.get("hash", "")
                name_val = kernel_dict.get("name", "")
                size_val = kernel_dict.get("total_size")
                log.debug("hash %s name %s total_size %s", hash_val, name_val, size_val)
                grouping_key = (name_val, size_val)
                kernels_grouped_by_name_and_size[grouping_key].append(kernel_dict)

            final_duplicate_groups: List[List[Dict[str, Any]]] = []
            for (
                _,
                kernels_with_this_name_size,
            ) in kernels_grouped_by_name_and_size.items():
                if len(kernels_with_this_name_size) < 2:
                    continue

                processed_in_name_group = [False] * len(kernels_with_this_name_size)
                # pylint: disable=consider-using-enumerate
                # we'll change this logic
                for i in range(len(kernels_with_this_name_size)):
                    if processed_in_name_group[i]:
                        continue

                    current_duplicate_set = [kernels_with_this_name_size[i]]
                    processed_in_name_group[i] = True

                    for j in range(i + 1, len(kernels_with_this_name_size)):
                        if processed_in_name_group[j]:
                            continue

                        if Database._are_kernel_metadata_jsons_duplicates(
                            kernels_with_this_name_size[i].get("metadata"),
                            kernels_with_this_name_size[j].get("metadata"),
                        ):
                            current_duplicate_set.append(kernels_with_this_name_size[j])
                            processed_in_name_group[j] = True

                    if len(current_duplicate_set) > 1:
                        final_duplicate_groups.append(current_duplicate_set)

            log.debug(
                "Found %s sets of duplicate kernels "
                "(grouped by name, JSON metadata identical or "
                "differs only in internal 'hash' field).",
                len(final_duplicate_groups),
            )
            return final_duplicate_groups

        except Exception as e:  # pylint: disable=broad-except
            log.error(
                "DB Find Duplicates (Name-grouped, specific 'hash' diff JSON method): Failed: %s",
                e,
                exc_info=True,
            )
            return []
        finally:
            session.close()

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
