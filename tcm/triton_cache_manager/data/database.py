"""
Database module for storing and retrieving Triton kernel metadata.

This module provides a SQLite-based database interface for kernel metadata.
"""

from __future__ import annotations
import sqlite3
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict
from ..utils.paths import get_db_path
from ..models.kernel import Kernel
from ..models.criteria import SearchCriteria

log = logging.getLogger(__name__)


def _json(x):
    return json.dumps(x)


def _bool(x):
    return int(bool(x))


class Database:
    """
    SQLite database for storing and querying Triton kernel metadata.

    This class provides methods to initialize the database schema,
    insert kernel metadata, and search for kernels based on criteria.
    """

    def __init__(self, path: Path | None = None):
        """
        Initialize the database connection.

        Args:
            path: Path to the database file. If None, uses the default location.
        """
        self.path = path or get_db_path()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema()

    def _ensure_schema(self):
        """
        Ensure the database schema exists and is properly initialized.

        Creates the necessary tables and indexes if they do not exist.
        """

        cur = self.conn.cursor()
        cur.execute("PRAGMA user_version")
        if cur.fetchone()[0] == 0:
            self.conn.executescript(self._schema_sql())
            self.conn.execute("PRAGMA user_version = 1")
            self.conn.commit()

    @staticmethod
    def _schema_sql() -> str:
        """
        Generate SQL for database schema creation.

        Returns:
            SQL script to create the database schema.
        """

        return """
        CREATE TABLE kernels(
            hash TEXT PRIMARY KEY,
            backend TEXT,
            arch TEXT,
            name TEXT,
            warp_size INTEGER,
            num_warps INTEGER,
            num_stages INTEGER,
            num_ctas INTEGER,
            maxnreg INTEGER,
            cluster_dims JSON,
            ptx_version TEXT,
            enable_fp_fusion BOOLEAN,
            launch_cooperative_grid BOOLEAN,
            supported_fp8_dtypes JSON,
            deprecated_fp8_dtypes JSON,
            default_dot_input_precision TEXT,
            allowed_dot_input_precisions JSON,
            max_num_imprecise_acc_default INTEGER,
            extern_libs JSON,
            debug BOOLEAN,
            backend_name TEXT,
            sanitize_overflow BOOLEAN,
            triton_version TEXT,
            shared INTEGER,
            tmem_size INTEGER,
            global_scratch_size INTEGER,
            global_scratch_align INTEGER,
            waves_per_eu INTEGER,
            kpack INTEGER,
            matrix_instr_nonkdim INTEGER,
            created INTEGER,
            total_size INTEGER,
            metadata JSON,
            modified_time REAL
        );
        CREATE TABLE files(
            hash TEXT,
            type TEXT,
            rel_path TEXT,
            size INTEGER,
            FOREIGN KEY(hash) REFERENCES kernels(hash) ON DELETE CASCADE
        );
        CREATE INDEX idx_name ON kernels(name);
        """

    def insert_kernel(self, k: Kernel) -> None:
        """
        Upsert a kernel and refresh its file list.

        Args:
            k: Kernel object containing metadata to be stored.
        """
        c = self.conn.cursor()

        total_size = sum(f.size for f in k.files)

        row: Dict[str, Any] = {
            "hash": k.hash,
            "backend": k.backend,
            "arch": k.arch,
            "name": k.name,
            "warp_size": k.warp_size,
            "num_warps": k.num_warps,
            "num_stages": k.num_stages,
            "num_ctas": k.num_ctas,
            "maxnreg": k.maxnreg,
            "cluster_dims": _json(k.cluster_dims),
            "ptx_version": k.ptx_version,
            "enable_fp_fusion": _bool(k.enable_fp_fusion),
            "launch_cooperative_grid": _bool(k.launch_cooperative_grid),
            "supported_fp8_dtypes": _json(k.supported_fp8_dtypes),
            "deprecated_fp8_dtypes": _json(k.deprecated_fp8_dtypes),
            "default_dot_input_precision": k.default_dot_input_precision,
            "allowed_dot_input_precisions": _json(k.allowed_dot_input_precisions),
            "max_num_imprecise_acc_default": k.max_num_imprecise_acc_default,
            "extern_libs": _json(k.extern_libs),
            "debug": _bool(k.debug),
            "backend_name": k.backend_name,
            "sanitize_overflow": _bool(k.sanitize_overflow),
            "triton_version": k.triton_version,
            "shared": k.shared,
            "tmem_size": k.tmem_size,
            "global_scratch_size": k.global_scratch_size,
            "global_scratch_align": k.global_scratch_align,
            "waves_per_eu": k.waves_per_eu,
            "kpack": k.kpack,
            "matrix_instr_nonkdim": k.matrix_instr_nonkdim,
            "created": int(time.time()),
            "total_size": total_size,
            "metadata": _json(k.metadata),
            "modified_time": k.modified_time,
        }

        filtered = {col: val for col, val in row.items() if val is not None}
        cols = ", ".join(filtered.keys())
        placeholders = ", ".join("?" for _ in filtered)
        updates = ", ".join(
            f"{col}=excluded.{col}" for col in filtered if col != "hash"
        )
        values = tuple(filtered.values())

        sql = (
            f"INSERT INTO kernels ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(hash) DO UPDATE SET {updates}"
        )
        c.execute(sql, values)

        c.execute("DELETE FROM files WHERE hash = ?", (k.hash,))
        c.executemany(
            "INSERT INTO files(hash, type, rel_path, size) VALUES (?, ?, ?, ?)",
            [(k.hash, f.file_type, f.path.name, f.size) for f in k.files],
        )

        self.conn.commit()

    def search(self, criteria: SearchCriteria):
        """
        Search for kernels matching specified criteria.

        Args:
            criteria: A SearchCriteria object containing filter values.

        Returns:
            List of dictionaries containing kernel metadata matching the criteria.
        """
        where_clauses = ["1=1"]
        params = []

        simple_equality_filters = ["name", "backend", "arch"]

        for field_name in simple_equality_filters:
            value = getattr(criteria, field_name, None)
            if value is not None:
                where_clauses.append(f"{field_name}=?")
                params.append(value)

        if criteria.older_than_timestamp is not None:
            where_clauses.append("modified_time < ?")
            params.append(criteria.older_than_timestamp)
        if criteria.younger_than_timestamp is not None:
            where_clauses.append("modified_time > ?")
            params.append(criteria.younger_than_timestamp)

        sql = f"SELECT * FROM kernels WHERE {' AND '.join(where_clauses)}"\
                +" ORDER BY modified_time DESC"

        log.debug("Executing SQL: %s with params: %s", sql, params)

        try:
            return [dict(r) for r in self.conn.execute(sql, tuple(params))]
        except sqlite3.Error as e:
            log.error(
                "Database search failed. SQL: %s, Params: %s, Error: %s", sql, params, e
            )
            return []

    def close(self):
        """Close the database connection."""
        self.conn.close()
