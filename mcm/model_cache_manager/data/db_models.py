# pylint: disable=c-extension-no-member,too-many-instance-attributes,too-many-locals
"""
SQLAlchemy ORM model definitions for the Triton kernel cache.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    Integer,
    String,
    ForeignKeyConstraint,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    Session as SqlaSession,
)

from ..models.kernel import Kernel
from ..utils.strategy_constants import VLLM_EXTENDED_PRIMARY_FIELDS


log = logging.getLogger(__name__)


class Base(DeclarativeBase):  # pylint: disable=too-few-public-methods
    """Base class for SQLAlchemy ORM models."""


class BaseKernelMixin:  # pylint: disable=too-few-public-methods
    """
    Mixin class containing common fields for kernel ORM models.
    """

    backend: Mapped[Optional[str]] = mapped_column(String, index=True)
    arch: Mapped[Optional[str]] = mapped_column(String, index=True)
    name: Mapped[Optional[str]] = mapped_column(String, index=True)
    warp_size: Mapped[Optional[int]] = mapped_column(Integer)
    num_warps: Mapped[Optional[int]] = mapped_column(Integer)
    num_stages: Mapped[Optional[int]] = mapped_column(Integer)
    num_ctas: Mapped[Optional[int]] = mapped_column(Integer)
    maxnreg: Mapped[Optional[int]] = mapped_column(Integer)
    cluster_dims: Mapped[Optional[List[int]]] = mapped_column(JSON)
    ptx_version: Mapped[Optional[str]] = mapped_column(String)
    enable_fp_fusion: Mapped[Optional[bool]] = mapped_column(Boolean)
    launch_cooperative_grid: Mapped[Optional[bool]] = mapped_column(Boolean)
    supported_fp8_dtypes: Mapped[Optional[List[str]]] = mapped_column(JSON)
    deprecated_fp8_dtypes: Mapped[Optional[List[str]]] = mapped_column(JSON)
    default_dot_input_precision: Mapped[Optional[str]] = mapped_column(String)
    allowed_dot_input_precisions: Mapped[Optional[List[str]]] = mapped_column(JSON)
    max_num_imprecise_acc_default: Mapped[Optional[int]] = mapped_column(Integer)
    extern_libs: Mapped[Optional[List[List[str]]]] = mapped_column(JSON)
    debug: Mapped[Optional[bool]] = mapped_column(Boolean)
    backend_name: Mapped[Optional[str]] = mapped_column(String)
    sanitize_overflow: Mapped[Optional[bool]] = mapped_column(Boolean)
    triton_version: Mapped[Optional[str]] = mapped_column(String)
    shared: Mapped[Optional[int]] = mapped_column(Integer)
    tmem_size: Mapped[Optional[int]] = mapped_column(Integer)
    global_scratch_size: Mapped[Optional[int]] = mapped_column(Integer)
    global_scratch_align: Mapped[Optional[int]] = mapped_column(Integer)
    waves_per_eu: Mapped[Optional[int]] = mapped_column(Integer)
    kpack: Mapped[Optional[int]] = mapped_column(Integer)
    matrix_instr_nonkdim: Mapped[Optional[int]] = mapped_column(Integer)
    created: Mapped[int] = mapped_column(Integer, default=lambda: int(time.time()))
    total_size: Mapped[Optional[int]] = mapped_column(Integer)
    kernel_metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    modified_time: Mapped[Optional[float]] = mapped_column(Float, index=True)
    runtime_hits: Mapped[Optional[int]] = mapped_column(
        Integer, default=0, nullable=False
    )
    last_access_time: Mapped[Optional[float]] = mapped_column(
        Float, index=True, nullable=False, default=time.time()
    )

    @classmethod
    def get_common_kernel_values(cls, k_data: Kernel) -> Dict[str, Any]:
        """
        Extracts and returns a dictionary of common kernel field values from
        a Kernel data transfer object (DTO).

        This method is used to map the attributes of a Kernel instance to a
        dictionary suitable for database insertion
        or update operations. The returned dictionary contains key-value
        pairs corresponding to the fields defined in the ORM model.

        Args:
            k_data (Kernel): The Kernel DTO instance containing kernel metadata and configuration.

        Returns:
            Dict[str, Any]: A dictionary mapping field names to their corresponding
            values extracted from the
            Kernel DTO. This includes fields such as backend, arch,
            name, warp_size, num_warps, and others.
            The 'total_size' field is computed as the sum of the sizes of all files in k_data.files.
        """
        return {
            "backend": k_data.backend,
            "arch": str(k_data.arch),
            "name": k_data.name,
            "warp_size": k_data.warp_size,
            "num_warps": k_data.num_warps,
            "num_stages": k_data.num_stages,
            "num_ctas": k_data.num_ctas,
            "maxnreg": k_data.maxnreg,
            "cluster_dims": k_data.cluster_dims,
            "ptx_version": k_data.ptx_version,
            "enable_fp_fusion": k_data.enable_fp_fusion,
            "launch_cooperative_grid": k_data.launch_cooperative_grid,
            "supported_fp8_dtypes": k_data.supported_fp8_dtypes,
            "deprecated_fp8_dtypes": k_data.deprecated_fp8_dtypes,
            "default_dot_input_precision": k_data.default_dot_input_precision,
            "allowed_dot_input_precisions": k_data.allowed_dot_input_precisions,
            "max_num_imprecise_acc_default": k_data.max_num_imprecise_acc_default,
            "extern_libs": k_data.extern_libs,
            "debug": k_data.debug,
            "backend_name": k_data.backend_name,
            "sanitize_overflow": k_data.sanitize_overflow,
            "triton_version": k_data.triton_version,
            "shared": k_data.shared,
            "tmem_size": k_data.tmem_size,
            "global_scratch_size": k_data.global_scratch_size,
            "global_scratch_align": k_data.global_scratch_align,
            "waves_per_eu": k_data.waves_per_eu,
            "kpack": k_data.kpack,
            "matrix_instr_nonkdim": k_data.matrix_instr_nonkdim,
            "created": int(time.time()),
            "total_size": sum(f.size for f in k_data.files if f.size is not None),
            "kernel_metadata_json": k_data.metadata,
            "modified_time": k_data.modified_time,
        }


class KernelOrm(Base, BaseKernelMixin):
    """
    SQLAlchemy ORM model for a Triton kernel.
    """

    __tablename__ = "kernels"

    hash: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    cache_dir: Mapped[str] = mapped_column(String, primary_key=True, index=True)

    files: Mapped[List["KernelFileOrm"]] = relationship(
        "KernelFileOrm",
        back_populates="kernel",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the ORM object to a dictionary.
        Maps 'kernel_metadata_json' back to 'metadata' for compatibility.
        """
        d = {c.key: getattr(self, c.key) for c in self.__table__.columns}
        if "kernel_metadata_json" in d:
            d["metadata"] = d.pop("kernel_metadata_json")
        return d

    @classmethod
    def upsert_from_dto(cls, session: SqlaSession, k_data: Kernel) -> None:
        """
        Creates or updates a kernel record from a Kernel DTO, including files.
        """
        kernel_values = cls.get_common_kernel_values(k_data)
        kernel_values.update(
            {
                "hash": k_data.hash,
                "cache_dir": k_data.cache_dir,
            }
        )

        stmt = sqlite_insert(cls).values(kernel_values)
        update_dict = {
            col.name: getattr(stmt.excluded, col.name)
            for col in cls.__table__.columns
            if col.name not in ("hash", "cache_dir")
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["hash", "cache_dir"], set_=update_dict
        )
        session.execute(stmt)
        log.debug(
            "Upserted kernel_hash %s kernel_cache_dir %s", k_data.hash, k_data.cache_dir
        )

        session.query(KernelFileOrm).filter(
            KernelFileOrm.kernel_hash == k_data.hash,
            KernelFileOrm.kernel_cache_dir == k_data.cache_dir,
        ).delete(synchronize_session="fetch")
        log.debug(
            "Deleted existing files for kernel_hash %s and cache_dir %s",
            k_data.hash,
            k_data.cache_dir,
        )

        for f_dto in k_data.files:
            kernel_file_orm = KernelFileOrm(
                kernel_hash=k_data.hash,
                kernel_cache_dir=k_data.cache_dir,
                type=f_dto.file_type,
                rel_path=f_dto.path.name,
                size=f_dto.size,
            )
            session.add(kernel_file_orm)
        log.debug(
            "Added %d files for kernel_hash %s and cache_dir %s ",
            len(k_data.files),
            k_data.hash,
            k_data.cache_dir,
        )


class VllmLegacyKernelOrm(Base, BaseKernelMixin):
    """
    SQLAlchemy ORM model for a legacy vLLM Triton kernel.
    Uses a composite primary key of (cache_dir, vllm_hash, triton_cache_key, rank_x_y).
    """

    __tablename__ = "vllm_legacy_kernels"

    cache_dir: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    vllm_hash: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    triton_cache_key: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    rank_x_y: Mapped[str] = mapped_column(String, primary_key=True, index=True)

    files: Mapped[List["VllmLegacyKernelFileOrm"]] = relationship(
        "VllmLegacyKernelFileOrm",
        back_populates="kernel",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the ORM object to a dictionary.
        Maps 'kernel_metadata_json' back to 'metadata' for compatibility.
        """
        d = {c.key: getattr(self, c.key) for c in self.__table__.columns}
        if "kernel_metadata_json" in d:
            d["metadata"] = d.pop("kernel_metadata_json")
        return d

    @staticmethod
    def get_vllm_kernel_values(
        k_data: Kernel, cache_dir: str, vllm_hash: str, rank_x_y: str
    ) -> Dict[str, Any]:
        """
        Get legacy vLLM-specific kernel values.

        Args:
            k_data: Kernel data
            cache_dir: Cache directory path
            vllm_hash: vLLM hash
            rank_x_y: Rank identifier

        Returns:
            Dictionary with legacy vLLM kernel values
        """
        kernel_values = VllmLegacyKernelOrm.get_common_kernel_values(k_data)
        kernel_values.update(
            {
                "cache_dir": cache_dir,
                "vllm_hash": vllm_hash,
                "triton_cache_key": k_data.hash,
                "rank_x_y": rank_x_y,
            }
        )
        return kernel_values

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    @classmethod
    def upsert_from_dto(
        cls,
        session: SqlaSession,
        k_data: Kernel,
        cache_dir: str,
        vllm_hash: str,
        rank_x_y: str,
    ) -> None:
        """
        Creates or updates a legacy vLLM kernel record from a Kernel DTO, including files.
        """
        kernel_values = cls.get_vllm_kernel_values(
            k_data, cache_dir, vllm_hash, rank_x_y
        )

        stmt = sqlite_insert(cls).values(kernel_values)
        update_dict = {
            col.name: getattr(stmt.excluded, col.name)
            for col in cls.__table__.columns
            if col.name
            not in ("cache_dir", "rank_x_y", "vllm_hash", "triton_cache_key")
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                "cache_dir",
                "rank_x_y",
                "vllm_hash",
                "triton_cache_key",
            ],
            set_=update_dict,
        )
        session.execute(stmt)
        log.debug(
            "Upserted legacy vLLM kernel cache_dir %s vllm_hash %s "
            "triton_cache_key %s with rank_x_y %s",
            cache_dir,
            vllm_hash,
            k_data.hash,
            rank_x_y,
        )

        session.query(VllmLegacyKernelFileOrm).filter(
            VllmLegacyKernelFileOrm.cache_dir == cache_dir,
            VllmLegacyKernelFileOrm.vllm_hash == vllm_hash,
            VllmLegacyKernelFileOrm.triton_cache_key == k_data.hash,
            VllmLegacyKernelFileOrm.rank_x_y == rank_x_y,
        ).delete(synchronize_session="fetch")
        log.debug(
            "Deleted existing files for cache_dir %s vllm_hash %s "
            "triton_cache_key %s and rank_x_y %s",
            cache_dir,
            vllm_hash,
            k_data.hash,
            rank_x_y,
        )

        for f_dto in k_data.files:
            kernel_file_orm = VllmLegacyKernelFileOrm(
                cache_dir=cache_dir,
                vllm_hash=vllm_hash,
                triton_cache_key=k_data.hash,
                rank_x_y=rank_x_y,
                type=f_dto.file_type,
                rel_path=f_dto.path.name,
                size=f_dto.size,
            )
            session.add(kernel_file_orm)
        log.debug(
            "Added %d files for cache_dir %s vllm_hash %s "
            "triton_cache_key %s and rank_x_y %s",
            len(k_data.files),
            cache_dir,
            vllm_hash,
            k_data.hash,
            rank_x_y,
        )


class VllmKernelOrm(Base, BaseKernelMixin):
    """
    SQLAlchemy ORM model for a new vLLM Triton kernel.
    Uses a composite primary key of (cache_dir, vllm_hash, triton_cache_key,
    rank_x_y, artifact_shape).
    """

    __tablename__ = "vllm_kernels"

    cache_dir: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    vllm_hash: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    triton_cache_key: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    rank_x_y: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    artifact_shape: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    best_config: Mapped[Optional[str]] = mapped_column(
        String
    )  # JSON string for best config

    files: Mapped[List["VllmKernelFileOrm"]] = relationship(
        "VllmKernelFileOrm",
        back_populates="kernel",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the ORM object to a dictionary.
        Maps 'kernel_metadata_json' back to 'metadata' for compatibility.
        """
        d = {c.key: getattr(self, c.key) for c in self.__table__.columns}
        if "kernel_metadata_json" in d:
            d["metadata"] = d.pop("kernel_metadata_json")
        return d

    @staticmethod
    def get_vllm_kernel_values(
        k_data: Kernel,
        cache_dir: str,
        *,
        vllm_hash: str,
        rank_x_y: str,
        artifact_shape: str,
        best_config: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get new vLLM-specific kernel values.

        Args:
            k_data: Kernel data
            cache_dir: Cache directory path
            vllm_hash: vLLM hash
            rank_x_y: Rank identifier
            artifact_shape: Artifact shape directory name
            best_config: Optional JSON string for best config

        Returns:
            Dictionary with new vLLM kernel values
        """
        kernel_values = VllmKernelOrm.get_common_kernel_values(k_data)
        kernel_values.update(
            {
                "cache_dir": cache_dir,
                "vllm_hash": vllm_hash,
                "triton_cache_key": k_data.hash,
                "rank_x_y": rank_x_y,
                "artifact_shape": artifact_shape,
                "best_config": best_config,
            }
        )
        return kernel_values

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    @classmethod
    def upsert_from_dto(
        cls,
        session: SqlaSession,
        k_data: Kernel,
        cache_dir: str,
        vllm_hash: str,
        rank_x_y: str,
        artifact_shape: str,
        best_config: Optional[str] = None,
    ) -> None:
        """
        Creates or updates a new vLLM kernel record from a Kernel DTO, including files.
        """
        kernel_values = cls.get_vllm_kernel_values(
            k_data, cache_dir,
            vllm_hash=vllm_hash, rank_x_y=rank_x_y,
            artifact_shape=artifact_shape, best_config=best_config
        )

        stmt = sqlite_insert(cls).values(kernel_values)
        update_dict = {
            col.name: getattr(stmt.excluded, col.name)
            for col in cls.__table__.columns
            if col.name
            not in (
                "cache_dir",
                "rank_x_y",
                "vllm_hash",
                "triton_cache_key",
                "artifact_shape",
            )
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                "cache_dir",
                "rank_x_y",
                "vllm_hash",
                "triton_cache_key",
                "artifact_shape",
            ],
            set_=update_dict,
        )
        session.execute(stmt)
        log.debug(
            "Upserted new vLLM kernel cache_dir %s vllm_hash %s "
            "triton_cache_key %s with rank_x_y %s",
            cache_dir,
            vllm_hash,
            k_data.hash,
            rank_x_y,
        )

        session.query(VllmKernelFileOrm).filter(
            VllmKernelFileOrm.cache_dir == cache_dir,
            VllmKernelFileOrm.vllm_hash == vllm_hash,
            VllmKernelFileOrm.triton_cache_key == k_data.hash,
            VllmKernelFileOrm.rank_x_y == rank_x_y,
            VllmKernelFileOrm.artifact_shape == artifact_shape,
        ).delete(synchronize_session="fetch")
        log.debug(
            "Deleted existing files for cache_dir %s vllm_hash %s "
            "triton_cache_key %s and rank_x_y %s",
            cache_dir,
            vllm_hash,
            k_data.hash,
            rank_x_y,
        )

        for f_dto in k_data.files:
            kernel_file_orm = VllmKernelFileOrm(
                cache_dir=cache_dir,
                vllm_hash=vllm_hash,
                triton_cache_key=k_data.hash,
                rank_x_y=rank_x_y,
                artifact_shape=artifact_shape,
                type=f_dto.file_type,
                rel_path=f_dto.path.name,
                size=f_dto.size,
            )
            session.add(kernel_file_orm)
        log.debug(
            "Added %d files for cache_dir %s vllm_hash %s "
            "triton_cache_key %s and rank_x_y %s",
            len(k_data.files),
            cache_dir,
            vllm_hash,
            k_data.hash,
            rank_x_y,
        )


class KernelFileOrm(Base):  # pylint: disable=too-few-public-methods
    """SQLAlchemy ORM model for a file associated with a Triton kernel."""

    __tablename__ = "files"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True
    )
    kernel_hash: Mapped[str] = mapped_column(String)
    kernel_cache_dir: Mapped[str] = mapped_column(String)

    __table_args__ = (
        ForeignKeyConstraint(
            ["kernel_hash", "kernel_cache_dir"],
            ["kernels.hash", "kernels.cache_dir"],
            ondelete="CASCADE",
        ),
    )

    type: Mapped[Optional[str]] = mapped_column(String)
    rel_path: Mapped[Optional[str]] = mapped_column(String)
    size: Mapped[Optional[int]] = mapped_column(Integer)

    kernel: Mapped["KernelOrm"] = relationship("KernelOrm", back_populates="files")


class VllmLegacyKernelFileOrm(Base):  # pylint: disable=too-few-public-methods
    """SQLAlchemy ORM model for a file associated with a legacy vLLM Triton kernel."""

    __tablename__ = "vllm_legacy_files"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True
    )
    cache_dir: Mapped[str] = mapped_column(String)
    vllm_hash: Mapped[str] = mapped_column(String)
    triton_cache_key: Mapped[str] = mapped_column(String)
    rank_x_y: Mapped[str] = mapped_column(String)

    __table_args__ = (
        ForeignKeyConstraint(
            ["cache_dir", "vllm_hash", "triton_cache_key", "rank_x_y"],
            [
                "vllm_legacy_kernels.cache_dir",
                "vllm_legacy_kernels.vllm_hash",
                "vllm_legacy_kernels.triton_cache_key",
                "vllm_legacy_kernels.rank_x_y",
            ],
            ondelete="CASCADE",
        ),
    )

    type: Mapped[Optional[str]] = mapped_column(String)
    rel_path: Mapped[Optional[str]] = mapped_column(String)
    size: Mapped[Optional[int]] = mapped_column(Integer)

    kernel: Mapped["VllmLegacyKernelOrm"] = relationship(
        "VllmLegacyKernelOrm", back_populates="files"
    )


class VllmKernelFileOrm(Base):  # pylint: disable=too-few-public-methods
    """SQLAlchemy ORM model for a file associated with a new vLLM Triton kernel."""

    __tablename__ = "vllm_files"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True
    )
    cache_dir: Mapped[str] = mapped_column(String)
    vllm_hash: Mapped[str] = mapped_column(String)
    triton_cache_key: Mapped[str] = mapped_column(String)
    rank_x_y: Mapped[str] = mapped_column(String)
    artifact_shape: Mapped[str] = mapped_column(String)

    __table_args__ = (
        ForeignKeyConstraint(
            VLLM_EXTENDED_PRIMARY_FIELDS,
            [
                "vllm_kernels.cache_dir",
                "vllm_kernels.vllm_hash",
                "vllm_kernels.triton_cache_key",
                "vllm_kernels.rank_x_y",
                "vllm_kernels.artifact_shape",
            ],
            ondelete="CASCADE",
        ),
    )

    type: Mapped[Optional[str]] = mapped_column(String)
    rel_path: Mapped[Optional[str]] = mapped_column(String)
    size: Mapped[Optional[int]] = mapped_column(Integer)

    kernel: Mapped["VllmKernelOrm"] = relationship(
        "VllmKernelOrm", back_populates="files"
    )
