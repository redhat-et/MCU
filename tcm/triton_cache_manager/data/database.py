# pylint: disable=c-extension-no-member
"""
Database service class for managing Triton kernel metadata.

This module provides the `Database` class, which acts as a high-level API
for interacting with the kernel cache database. It uses ORM models (SqlAlchemy).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List
from sqlalchemy import and_

from .db_config import engine, SessionLocal, DB_PATH
from .db_models import Base, KernelOrm, SqlaSession

from ..models.criteria import SearchCriteria
from ..models.kernel import Kernel

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

    def close(self) -> None:
        """Closes the database engine's connection pool."""
        if self.engine:
            self.engine.dispose()
            log.info("Database engine connection pool disposed.")
