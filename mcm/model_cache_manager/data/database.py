# pylint: disable=c-extension-no-member
"""
Database service class for managing kernel metadata.

This module provides the `Database` class, which acts as a high-level API
for interacting with the kernel cache database. It uses ORM models (SqlAlchemy).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Iterable
import collections
from sqlalchemy import and_, exc, or_, func

from .db_config import engine, SessionLocal, DB_PATH, create_engine_and_session
from .db_models import (
    Base,
    KernelOrm,
    KernelFileOrm,
    VllmKernelOrm,
    VllmKernelFileOrm,
    SqlaSession,
)

from ..models.criteria import SearchCriteria
from ..models.kernel import Kernel
from ..utils.mcm_constants import IR_EXTS
from ..utils.utils import build_common_search_filters

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
        try:
            Base.metadata.create_all(bind=self.engine)
            log.info("Database schema verified/created at %s.", DB_PATH)
        except Exception as e:  # pylint: disable=broad-except
            log.error("Fatal error creating database schema: %s", e, exc_info=True)
            raise

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
        TODO : we will probably change it since we can get more metadata from inductor
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
        return Database.find_duplicates_generic(self.SessionLocal, KernelOrm, "hash")

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

    @staticmethod
    def find_duplicates_generic(
        session_factory,
        orm_class,
        hash_field: str,
        additional_fields: List[str] | None = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Generic method to find duplicate kernels for any ORM class.

        Args:
            session_factory: SQLAlchemy session factory function
            orm_class: The ORM class (KernelOrm or VllmKernelOrm)
            hash_field: Primary hash field name ("hash" for triton, "triton_cache_key" for vllm)
            additional_fields: Additional fields to include in the result (e.g., ["vllm_hash"])
        """
        session = session_factory()
        try:
            kernel_data = Database._query_all_kernels(
                session, orm_class, hash_field, additional_fields
            )
            if not kernel_data:
                return []

            kernel_dicts = Database._build_kernel_dictionaries(
                kernel_data, hash_field, additional_fields
            )
            grouped_kernels = Database._group_kernels_by_name_and_size(
                kernel_dicts, hash_field
            )
            duplicate_groups = Database._find_duplicate_groups_in_groups(
                grouped_kernels
            )

            log.debug(
                "Found %s sets of duplicate kernels using %s "
                "(grouped by name, JSON metadata identical or "
                "differs only in internal 'hash' field).",
                len(duplicate_groups),
                orm_class.__name__,
            )
            return duplicate_groups

        except Exception as e:  # pylint: disable=broad-except
            log.error(
                "DB Find Duplicates (%s): Failed: %s",
                orm_class.__name__,
                e,
                exc_info=True,
            )
            return []
        finally:
            session.close()

    @staticmethod
    def _query_all_kernels(
        session, orm_class, hash_field: str, additional_fields: List[str] | None = None
    ):
        """Query all kernels with required fields."""
        base_fields = [
            getattr(orm_class, hash_field),
            orm_class.name,
            orm_class.kernel_metadata_json,
            orm_class.modified_time,
            orm_class.backend,
            orm_class.arch,
            orm_class.triton_version,
            orm_class.total_size,
        ]

        if additional_fields:
            for field_name in additional_fields:
                base_fields.append(getattr(orm_class, field_name))

        return session.query(*base_fields).order_by(orm_class.modified_time.asc()).all()

    @staticmethod
    def _build_kernel_dictionaries(
        kernel_data, hash_field: str, additional_fields: List[str] | None = None
    ) -> List[Dict[str, Any]]:
        """Build kernel dictionaries from ORM query results."""
        kernel_list_of_dicts: List[Dict[str, Any]] = []

        for k in kernel_data:
            kernel_dict = {
                hash_field: getattr(k, hash_field),
                "name": k.name,
                "metadata": k.kernel_metadata_json,
                "modified_time": k.modified_time,
                "backend": k.backend,
                "arch": k.arch,
                "triton_version": k.triton_version,
                "total_size": k.total_size,
            }

            if additional_fields:
                for field_name in additional_fields:
                    kernel_dict[field_name] = getattr(k, field_name)

            kernel_list_of_dicts.append(kernel_dict)

        return kernel_list_of_dicts

    @staticmethod
    def _group_kernels_by_name_and_size(
        kernel_dicts: List[Dict[str, Any]], hash_field: str
    ) -> Dict:
        """Group kernels by name and total size."""
        grouped_kernels = collections.defaultdict(list)

        for kernel_dict in kernel_dicts:
            hash_val = kernel_dict.get(hash_field, "")
            name_val = kernel_dict.get("name", "")
            size_val = kernel_dict.get("total_size")
            log.debug(
                "%s %s name %s total_size %s", hash_field, hash_val, name_val, size_val
            )
            grouping_key = (name_val, size_val)
            grouped_kernels[grouping_key].append(kernel_dict)

        return grouped_kernels

    @staticmethod
    def _find_duplicate_groups_in_groups(
        grouped_kernels: Dict,
    ) -> List[List[Dict[str, Any]]]:
        """Find duplicate groups within name/size groups."""
        final_duplicate_groups: List[List[Dict[str, Any]]] = []

        for kernels_with_same_name_size in grouped_kernels.values():
            if len(kernels_with_same_name_size) < 2:
                continue

            duplicate_sets = Database._find_duplicates_in_single_group(
                kernels_with_same_name_size
            )
            final_duplicate_groups.extend(duplicate_sets)

        return final_duplicate_groups

    @staticmethod
    def _find_duplicates_in_single_group(
        kernels: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Find duplicate sets within a single name/size group."""
        processed = [False] * len(kernels)
        duplicate_sets = []

        for i, _ in enumerate(kernels):
            if processed[i]:
                continue

            current_duplicate_set = [kernels[i]]
            processed[i] = True

            for j in range(i + 1, len(kernels)):
                if processed[j]:
                    continue

                if Database._are_kernel_metadata_jsons_duplicates(
                    kernels[i].get("metadata"),
                    kernels[j].get("metadata"),
                ):
                    current_duplicate_set.append(kernels[j])
                    processed[j] = True

            if len(current_duplicate_set) > 1:
                duplicate_sets.append(current_duplicate_set)

        return duplicate_sets


class VllmDatabase:
    """
    Manages database interactions for vLLM kernel metadata.
    """

    def __init__(self) -> None:
        """Initializes DB engine, session factory, and ensures schema exists."""
        # pylint: disable=invalid-name
        self.engine, self.SessionLocal = create_engine_and_session(
            "vllm"
        )  # pylint: disable=invalid-name
        self._ensure_schema()
        log.info("vLLM Database service interface initialized successfully.")

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
        self, k_data: Kernel, vllm_cache_root: str, vllm_hash: str
    ) -> None:
        """
        Upserts a vLLM kernel and its associated files into the database.

        Args:
            k_data: A `Kernel` DTO containing the metadata.
            vllm_cache_root: Root path of the vLLM cache
            vllm_hash: Hash identifier for the vLLM cache group
        """
        session = self.get_session()
        try:
            VllmKernelOrm.upsert_from_dto(session, k_data, vllm_cache_root, vllm_hash)
            session.commit()
            log.info(
                "vLLM Kernel %s with vllm_cache_root %s vllm_hash %s upserted into DB.",
                k_data.hash,
                vllm_cache_root,
                vllm_hash,
            )
        except exc.IntegrityError as e:
            session.rollback()
            log.error(
                "Failed to upsert vLLM kernel %s with vllm_cache_root %s "
                "vllm_hash %s due to a constraint violation: %s",
                k_data.hash,
                vllm_cache_root,
                vllm_hash,
                e,
                exc_info=True,
            )
            raise
        except exc.OperationalError as e:
            session.rollback()
            log.error(
                "Failed to upsert vLLM kernel %s with vllm_cache_root %s "
                "vllm_hash %s due to a db operation issue: %s",
                k_data.hash,
                vllm_cache_root,
                vllm_hash,
                e,
                exc_info=True,
            )
            raise
        except Exception:  # pylint: disable=broad-except
            session.rollback()
            log.error(
                "DB Error: Failed to upsert vLLM kernel %s with vllm_cache_root %s vllm_hash %s.",
                k_data.hash,
                vllm_cache_root,
                vllm_hash,
                exc_info=True,
            )
            raise
        finally:
            session.close()

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
                ("cache_dir", VllmKernelOrm.vllm_cache_root, str),
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
            return [kernel_orm.to_dict() for kernel_orm in results_orm]
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
        return Database.find_duplicates_generic(
            self.SessionLocal, VllmKernelOrm, "triton_cache_key", ["vllm_hash"]
        )

    def close(self) -> None:
        """Closes the database engine's connection pool."""
        if self.engine:
            self.engine.dispose()
            log.info("vLLM Database engine connection pool disposed.")
