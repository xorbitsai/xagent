from dataclasses import dataclass
from enum import Enum
from typing import Any

from filelock import FileLock, Timeout

from .lancedb_maintenance import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LOCK_TIMEOUT,
    MaintenanceOutcome,
    MaintenanceStatus,
    lancedb_lock_path,
)
from .vector_compatibility import (
    EmbeddingIdentity,
    VectorCompatibility,
    _inspect_lancedb_vector_state,
    _lancedb_table_exists,
    canonical_embedding_identity,
    prepare_lancedb_memory_table,
)

REPAIR_REQUIRED_DETAIL = "Persistent memory requires offline repair and restart."
ADMISSION_FAILED_DETAIL = "Persistent memory admission failed safely."
QUIESCENCE_REQUIRED_DETAIL = "Persistent memory requires writer quiescence."


class StorageAdmissionState(str, Enum):
    DORMANT = "dormant"
    ADMITTED = "admitted"
    ABSENT = "absent"
    QUIESCENCE_REQUIRED = "quiescence_required"
    RETRYABLE_UNAVAILABLE = "retryable_unavailable"
    MAINTENANCE_INCOMPLETE = "maintenance_incomplete"
    BLOCKED_REPAIR = "blocked_repair"


class MemoryStorageMode(str, Enum):
    VECTOR = "vector"
    TEXT_ONLY = "text_only"


@dataclass(frozen=True)
class MemoryStorageCapabilities:
    mode: MemoryStorageMode
    readable: bool
    writable: bool
    vector_search: bool


@dataclass(frozen=True)
class DormantLanceDBMemoryHandle:
    connection: Any
    table_name: str


@dataclass(frozen=True)
class AdmittedLanceDBMemoryStore:
    handle: DormantLanceDBMemoryHandle
    capabilities: MemoryStorageCapabilities
    vector_compatibility: VectorCompatibility


@dataclass(frozen=True)
class StorageAdmissionOutcome:
    state: StorageAdmissionState
    dormant: DormantLanceDBMemoryHandle
    admitted: AdmittedLanceDBMemoryStore | None = None
    maintenance: MaintenanceOutcome | None = None
    detail: str | None = None


def _blocked(
    dormant: DormantLanceDBMemoryHandle,
    detail: str,
    maintenance: MaintenanceOutcome | None = None,
) -> StorageAdmissionOutcome:
    return StorageAdmissionOutcome(
        StorageAdmissionState.BLOCKED_REPAIR,
        dormant,
        maintenance=maintenance,
        detail=detail,
    )


def _unavailable(
    state: StorageAdmissionState,
    dormant: DormantLanceDBMemoryHandle,
    detail: str,
    maintenance: MaintenanceOutcome | None = None,
) -> StorageAdmissionOutcome:
    return StorageAdmissionOutcome(
        state, dormant, maintenance=maintenance, detail=detail
    )


def admit_lancedb_memory_storage(
    dormant: DormantLanceDBMemoryHandle,
    expected_identity: EmbeddingIdentity | dict[str, Any],
    *,
    writers_quiesced: bool,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
) -> StorageAdmissionOutcome:
    if writers_quiesced is not True:
        return _unavailable(
            StorageAdmissionState.QUIESCENCE_REQUIRED,
            dormant,
            QUIESCENCE_REQUIRED_DETAIL,
        )
    try:
        identity = canonical_embedding_identity(expected_identity)
        with FileLock(
            lancedb_lock_path(dormant.connection, dormant.table_name, "admission"),
            timeout=lock_timeout,
        ):
            if not _lancedb_table_exists(dormant.connection, dormant.table_name):
                return StorageAdmissionOutcome(StorageAdmissionState.ABSENT, dormant)
            maintenance = prepare_lancedb_memory_table(
                dormant.connection,
                dormant.table_name,
                identity,
                batch_size=batch_size,
                lock_timeout=lock_timeout,
            )
            if maintenance.status is MaintenanceStatus.ABSENT:
                return StorageAdmissionOutcome(
                    StorageAdmissionState.ABSENT, dormant, maintenance=maintenance
                )
            if maintenance.status in {
                MaintenanceStatus.INVALID_LEGACY_DATA,
                MaintenanceStatus.INCOMPATIBLE_SCHEMA,
            }:
                return _blocked(dormant, REPAIR_REQUIRED_DETAIL, maintenance)
            if maintenance.status is MaintenanceStatus.INCOMPLETE:
                return _unavailable(
                    StorageAdmissionState.MAINTENANCE_INCOMPLETE,
                    dormant,
                    ADMISSION_FAILED_DETAIL,
                    maintenance,
                )
            if maintenance.status is not MaintenanceStatus.COMPLETE:
                return _unavailable(
                    StorageAdmissionState.RETRYABLE_UNAVAILABLE,
                    dormant,
                    ADMISSION_FAILED_DETAIL,
                    maintenance,
                )
            # The persisted vector space is read back from the committed table,
            # never inferred from a snapshot taken before maintenance ran. A
            # concurrent recreation may have installed another identity's
            # vectors, and only this read under the still-held admission lock
            # reports the space the caller would actually be admitted over.
            _has_vector, compatibility = _inspect_lancedb_vector_state(
                dormant.connection, dormant.table_name, identity
            )
    except Timeout:
        return _unavailable(
            StorageAdmissionState.RETRYABLE_UNAVAILABLE,
            dormant,
            ADMISSION_FAILED_DETAIL,
        )
    except Exception:
        return _unavailable(
            StorageAdmissionState.RETRYABLE_UNAVAILABLE,
            dormant,
            ADMISSION_FAILED_DETAIL,
        )

    mode = (
        MemoryStorageMode.TEXT_ONLY
        if compatibility is VectorCompatibility.MISMATCHING
        else MemoryStorageMode.VECTOR
    )
    capabilities = MemoryStorageCapabilities(
        mode=mode,
        readable=True,
        writable=True,
        vector_search=compatibility is not VectorCompatibility.MISMATCHING,
    )
    admitted = AdmittedLanceDBMemoryStore(dormant, capabilities, compatibility)
    return StorageAdmissionOutcome(
        StorageAdmissionState.ADMITTED,
        dormant,
        admitted=admitted,
        maintenance=maintenance,
    )
