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
    canonical_embedding_identity,
    inspect_lancedb_vector_compatibility,
    prepare_lancedb_memory_table,
)

REPAIR_REQUIRED_DETAIL = "Persistent memory requires offline repair and restart."
ADMISSION_FAILED_DETAIL = "Persistent memory admission failed safely."
QUIESCENCE_REQUIRED_DETAIL = "Persistent memory requires writer quiescence."


class StorageAdmissionState(str, Enum):
    DORMANT = "dormant"
    ADMITTED = "admitted"
    BLOCKED_REPAIR = "blocked_repair"


@dataclass(frozen=True)
class MemoryStorageCapabilities:
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


def admit_lancedb_memory_storage(
    dormant: DormantLanceDBMemoryHandle,
    expected_identity: EmbeddingIdentity | dict[str, Any],
    *,
    writers_quiesced: bool,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
) -> StorageAdmissionOutcome:
    if writers_quiesced is not True:
        return _blocked(dormant, QUIESCENCE_REQUIRED_DETAIL)
    try:
        identity = canonical_embedding_identity(expected_identity)
        with FileLock(
            lancedb_lock_path(dormant.connection, dormant.table_name, "admission"),
            timeout=lock_timeout,
        ):
            maintenance = prepare_lancedb_memory_table(
                dormant.connection,
                dormant.table_name,
                identity,
                batch_size=batch_size,
                lock_timeout=lock_timeout,
            )
            if maintenance.status is not MaintenanceStatus.COMPLETE:
                return _blocked(dormant, REPAIR_REQUIRED_DETAIL, maintenance)
            compatibility = inspect_lancedb_vector_compatibility(
                dormant.connection, dormant.table_name, identity
            )
    except Timeout:
        return _blocked(dormant, ADMISSION_FAILED_DETAIL)
    except Exception:
        return _blocked(dormant, ADMISSION_FAILED_DETAIL)

    capabilities = MemoryStorageCapabilities(
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
