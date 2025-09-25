"""
Database utilities module for shared functionality.

This module contains generic database operations that can be used by both
Database and VllmLegacyDatabase classes to avoid circular imports.
"""
import collections
import logging
from typing import Any, Callable, Dict, List, Optional, Set
from sqlalchemy import exc
from sqlalchemy.orm import Session

from .db_config import DB_PATH
from .db_models import Base
from ..models.kernel import Kernel

log = logging.getLogger(__name__)


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
        kernel_data = _query_all_kernels(
            session, orm_class, hash_field, additional_fields
        )
        if not kernel_data:
            return []

        kernel_dicts = _build_kernel_dictionaries(
            kernel_data, hash_field, additional_fields
        )
        grouped_kernels = _group_kernels_by_name_and_size(
            kernel_dicts, hash_field
        )
        duplicate_groups = _find_duplicate_groups_in_groups(
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


def _find_duplicate_groups_in_groups(
    grouped_kernels: Dict,
) -> List[List[Dict[str, Any]]]:
    """Find duplicate groups within name/size groups."""
    final_duplicate_groups: List[List[Dict[str, Any]]] = []

    for kernels_with_same_name_size in grouped_kernels.values():
        if len(kernels_with_same_name_size) < 2:
            continue

        duplicate_sets = _find_duplicates_in_single_group(
            kernels_with_same_name_size
        )
        final_duplicate_groups.extend(duplicate_sets)

    return final_duplicate_groups


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

            if _are_kernel_metadata_jsons_duplicates(
                kernels[i].get("metadata"),
                kernels[j].get("metadata"),
            ):
                current_duplicate_set.append(kernels[j])
                processed[j] = True

        if len(current_duplicate_set) > 1:
            duplicate_sets.append(current_duplicate_set)

    return duplicate_sets


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
        val1 = metadata1.get(key)
        val2 = metadata2.get(key)

        if val1 != val2:
            differences_count += 1
            if key == "hash":
                hash_field_differed = True

    # Return True if exactly 0 differences OR exactly 1 difference in the 'hash' field
    return differences_count == 0 or (differences_count == 1 and hash_field_differed)


def ensure_schema(engine) -> None:
    """
    Ensures database schema (tables, indexes) exists.

    Args:
        engine: SQLAlchemy engine instance
    """
    try:
        Base.metadata.create_all(bind=engine)
        log.info("Database schema verified/created at %s.", DB_PATH)
    except Exception as e:  # pylint: disable=broad-except
        log.error("Fatal error creating database schema: %s", e, exc_info=True)
        raise


def handle_kernel_insert(
    session: Session,
    operation: Callable,
    kernel_data: Kernel,
    cache_dir: str,
    vllm_hash: str,
    rank_x_y: str,
    extra_args: Optional[dict] = None,
    error_prefix: str = "",
) -> None:
    """
    Handle database insert/upsert operations with standardized error handling.

    Args:
        session: Database session
        operation: The database operation to perform
        kernel_data: Kernel DTO containing metadata
        cache_dir: Root path of the cache
        vllm_hash: Hash identifier for the vLLM cache group
        rank_x_y: Rank identifier
        extra_args: Additional arguments for logging
        error_prefix: Prefix for error messages to distinguish legacy vs new
    """
    error_prefix = error_prefix or ""
    extra_args = extra_args or {}

    try:
        operation()
        session.commit()

        # Build log message
        log_msg = (
            f"{error_prefix}Kernel %s with cache_dir %s vllm_hash %s and "
            f"rank_x_y %s"
        )
        log_args = [kernel_data.hash, cache_dir, vllm_hash, rank_x_y]

        # Add extra arguments if provided
        for key, value in extra_args.items():
            log_msg += f" and {key} %s"
            log_args.append(value)

        log_msg += " upserted into DB."
        log.info(log_msg, *log_args)

    except exc.IntegrityError as e:
        session.rollback()
        _log_kernel_error(
            error_prefix,
            "constraint violation",
            kernel_data.hash,
            cache_dir,
            vllm_hash,
            rank_x_y,
            extra_args,
            e,
        )
        raise
    except exc.OperationalError as e:
        session.rollback()
        _log_kernel_error(
            error_prefix,
            "db operation issue",
            kernel_data.hash,
            cache_dir,
            vllm_hash,
            rank_x_y,
            extra_args,
            e,
        )
        raise
    except Exception:  # pylint: disable=broad-except
        session.rollback()
        _log_kernel_error(
            error_prefix,
            None,
            kernel_data.hash,
            cache_dir,
            vllm_hash,
            rank_x_y,
            extra_args,
            None,
        )
        raise
    finally:
        session.close()


def _log_kernel_error(
    error_prefix: str,
    error_type: Optional[str],
    kernel_hash: str,
    cache_dir: str,
    vllm_hash: str,
    rank_x_y: str,
    extra_args: dict,
    exception: Optional[Exception],
) -> None:
    """
    Helper to log kernel-related errors consistently.

    Args:
        error_prefix: Prefix for error messages
        error_type: Type of error (e.g., "constraint violation")
        kernel_hash: Hash of the kernel
        cache_dir: Cache directory path
        vllm_hash: vLLM hash identifier
        rank_x_y: Rank identifier
        extra_args: Additional arguments for logging
        exception: The exception that occurred
    """
    if error_type and exception:
        log_msg = (
            f"Failed to upsert {error_prefix}kernel %s with cache_dir %s "
            f"vllm_hash %s and rank_x_y %s"
        )
        log_args = [kernel_hash, cache_dir, vllm_hash, rank_x_y]

        # Add extra arguments
        for key, value in extra_args.items():
            log_msg += f" and {key} %s"
            log_args.append(value)

        log_msg += f" due to a {error_type}: %s"
        log_args.append(exception)

        log.error(log_msg, *log_args, exc_info=True)
    else:
        # Generic error
        log_msg = (
            f"DB Error: Failed to upsert {error_prefix}kernel %s with cache_dir %s "
            f"vllm_hash %s and rank_x_y %s"
        )
        log_args = [kernel_hash, cache_dir, vllm_hash, rank_x_y]

        # Add extra arguments
        for key, value in extra_args.items():
            log_msg += f" and {key} %s"
            log_args.append(value)

        log.error(log_msg, *log_args, exc_info=True)


def create_file_orm_dict(
    cache_dir: str,
    vllm_hash: str,
    triton_cache_key: str,
    rank_x_y: str,
    file_dto: Any
) -> Dict[str, Any]:
    """
    Create a file ORM dictionary from file DTO data.

    Shared between VllmDatabase and VllmLegacyDatabase to avoid duplication.
    """
    return {
        "cache_dir": cache_dir,
        "vllm_hash": vllm_hash,
        "triton_cache_key": triton_cache_key,
        "rank_x_y": rank_x_y,
        "type": file_dto.file_type,
        "rel_path": file_dto.path.name,
        "size": file_dto.size,
    }


def bulk_insert_kernels_generic(
    db_instance,
    kernels_data: List,
    batch_size: int = 1000
) -> int:
    """
    Generic bulk insert implementation for kernels.

    Shared between VllmDatabase and VllmLegacyDatabase to avoid duplication.

    Args:
        db_instance: Database instance with get_session() and _prepare_batch() methods
        kernels_data: List of kernel data tuples
        batch_size: Number of kernels per batch

    Returns:
        Number of kernels inserted
    """
    if not kernels_data:
        return 0

    session = db_instance.get_session()
    inserted_count = 0

    try:
        for i in range(0, len(kernels_data), batch_size):
            batch = kernels_data[i : i + batch_size]

            # Prepare batch data using instance-specific method
            kernel_batch_data, file_batch_data = db_instance._prepare_batch(batch)

            if kernel_batch_data:
                # Bulk insert kernels
                session.bulk_insert_mappings(
                    db_instance.get_kernel_orm_class(), kernel_batch_data
                )
                inserted_count += len(kernel_batch_data)

            if file_batch_data:
                # Bulk insert files
                session.bulk_insert_mappings(
                    db_instance.get_file_orm_class(), file_batch_data
                )

        session.commit()
        return inserted_count

    except Exception as e:
        session.rollback()
        log.error(
            "Failed to bulk insert kernels: %s",
            e,
            exc_info=True
        )
        raise
    finally:
        session.close()