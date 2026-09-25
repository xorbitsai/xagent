"""Startup admission and publication for the persistent-memory lifecycle.

This is the layer that turns the redesigned lifecycle on for the web runtimes.
Four owner contracts shape everything below:

* The runtime consumes only the explicit global embedding authority. An
  administrator's personal default embedding model is never a source of truth
  for persistent memory, so nothing here reads the model hub or
  ``UserDefaultModel``.
* Invalid legacy data becomes :attr:`MemoryLifecycleState.BLOCKED_REPAIR`.
  Memory reads and writes fail closed while unrelated functions keep starting.
* There is no normal online reload or migration. A change to the authority, or
  to the stored vector space, requires quiescence, offline repair where the
  data is invalid, and an all-worker restart.
* LanceDB is only ever touched through the layer B admission primitives, which
  is what keeps the runtime inside the supported-version matrix recorded in
  ``docs/deployment.md``.

Admission runs once, while writers are quiesced, and a store is published only
after it succeeds -- see :class:`AdmissionResult`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..core.memory import (
    DormantLanceDBMemoryHandle,
    MemoryStorageCapabilities,
    MemoryStorageMode,
    StorageAdmissionOutcome,
    StorageAdmissionState,
    admit_lancedb_memory_storage,
)
from ..core.memory.lancedb import LanceDBMemoryStore
from ..core.memory.lancedb_maintenance import (
    DEFAULT_LOCK_TIMEOUT,
    validate_lock_timeout,
)
from ..core.memory.vector_compatibility import (
    canonical_embedding_identity,
    create_or_recreate_vector_capable_table,
    embedding_identity_fingerprint,
)
from ..core.model.embedding import create_embedding_adapter
from ..core.model.embedding.base import BaseEmbedding
from ..core.model.model import EmbeddingModelConfig
from ..core.storage.manager import get_storage_root
from ..providers.vector_store.lancedb import get_connection
from .services.global_memory_embedding_authority import (
    AuthorityCredentialUnavailable,
    GlobalMemoryEmbeddingAuthoritySnapshot,
)
from .user_isolated_memory import UserIsolatedMemoryStore

logger = logging.getLogger(__name__)

MEMORY_TABLE_NAME = "memories"
AUTHORITY_CONFIG_ID = "global-memory-embedding-authority"
DEFAULT_SIMILARITY_THRESHOLD = 1.5

EmbeddingFactory = Callable[[EmbeddingModelConfig], BaseEmbedding]


class MemoryLifecycleState(str, Enum):
    """Public-safe lifecycle states.

    Every internal admission, credential and drift condition is folded onto one
    of these before it reaches an API response, a task runtime or a log line
    that an unprivileged caller can observe.
    """

    READY = "ready"
    NOT_CONFIGURED = "not_configured"
    CREDENTIAL_UNAVAILABLE = "credential_unavailable"
    RETRYABLE_UNAVAILABLE = "retryable_unavailable"
    RESTART_REQUIRED = "restart_required"
    BLOCKED_REPAIR = "blocked_repair"


#: The states in which no store may be served. Derived rather than listed, so
#: a state added later is fenced -- and detailed -- by default instead of by
#: someone remembering to add it here.
FENCED_STATES = frozenset(MemoryLifecycleState) - {
    MemoryLifecycleState.READY,
    MemoryLifecycleState.NOT_CONFIGURED,
}

#: The one caller-safe detail every fenced state answers with. One string for
#: all of them is the contract, not an oversight: the API returns this detail
#: verbatim, so a per-state wording would let an unprivileged caller tell a
#: credential fault from invalid legacy data from a transient backend failure.
#: The state-specific text lives in :data:`OPERATOR_GUIDANCE` alone.
FENCED_DETAIL = "Persistent memory is unavailable pending administrator action."

#: Stable, caller-safe wording. These strings are part of the API contract and
#: deliberately carry no path, model name, endpoint or credential material.
#: The fenced entries are generated, which is what makes "one detail for every
#: fenced state" structural rather than a convention a new state can break.
PUBLIC_DETAILS: dict[MemoryLifecycleState, str] = {
    MemoryLifecycleState.READY: "Persistent memory is available.",
    MemoryLifecycleState.NOT_CONFIGURED: "Persistent memory is not configured.",
    **{state: FENCED_DETAIL for state in FENCED_STATES},
}

#: Detailed guidance for the people who can act on it. This never reaches an
#: API response; it is written to the operator log and mirrored in
#: ``docs/deployment.md``.
OPERATOR_GUIDANCE: dict[MemoryLifecycleState, str] = {
    MemoryLifecycleState.NOT_CONFIGURED: (
        "No global memory embedding authority is configured. Persistent memory "
        "is disabled and an ephemeral in-process store is used instead. "
        "Configure the authority, then restart every worker."
    ),
    MemoryLifecycleState.CREDENTIAL_UNAVAILABLE: (
        "The stored global memory embedding credential could not be decrypted "
        "or failed its verifier. Re-set the authority through the admin API, "
        "then restart every worker. Memory reads and writes stay closed until "
        "then; unrelated functions are unaffected."
    ),
    MemoryLifecycleState.RETRYABLE_UNAVAILABLE: (
        "Storage admission could not complete: the admission lock was held or "
        "the backend failed transiently. Confirm that no other process is "
        "mid-maintenance on the memory directory, then restart this worker so "
        "it re-runs admission."
    ),
    MemoryLifecycleState.RESTART_REQUIRED: (
        "The global memory embedding authority no longer describes the vector "
        "space the stored memories were written under, or admission left "
        "maintenance incomplete. Quiesce all writers, re-embed offline if the "
        "existing vectors must be preserved, then restart every worker "
        "together. Do not restart one worker into a mixed fleet."
    ),
    MemoryLifecycleState.BLOCKED_REPAIR: (
        "Persistent memory storage holds invalid legacy data or an "
        "incompatible schema and has been fenced off. Stop every worker, take "
        "an offline backup of the memory LanceDB directory, repair or remove "
        "the offending rows offline, then restart every worker together. "
        "Memory reads and writes fail closed until that is done; unrelated "
        "functions continue to start and serve."
    ),
}

#: States an operator must clear. A worker never re-attempts admission out of
#: them, because doing so would be the online reload the owner contract forbids.
TERMINAL_STATES = frozenset(
    {
        MemoryLifecycleState.NOT_CONFIGURED,
        MemoryLifecycleState.CREDENTIAL_UNAVAILABLE,
        MemoryLifecycleState.RESTART_REQUIRED,
        MemoryLifecycleState.BLOCKED_REPAIR,
    }
)

_ADMISSION_STATES: dict[StorageAdmissionState, MemoryLifecycleState] = {
    StorageAdmissionState.ADMITTED: MemoryLifecycleState.READY,
    StorageAdmissionState.BLOCKED_REPAIR: MemoryLifecycleState.BLOCKED_REPAIR,
    StorageAdmissionState.QUIESCENCE_REQUIRED: MemoryLifecycleState.RESTART_REQUIRED,
    StorageAdmissionState.MAINTENANCE_INCOMPLETE: MemoryLifecycleState.RESTART_REQUIRED,
    StorageAdmissionState.RETRYABLE_UNAVAILABLE: (
        MemoryLifecycleState.RETRYABLE_UNAVAILABLE
    ),
    StorageAdmissionState.DORMANT: MemoryLifecycleState.RETRYABLE_UNAVAILABLE,
    StorageAdmissionState.ABSENT: MemoryLifecycleState.RETRYABLE_UNAVAILABLE,
}


@dataclass(frozen=True)
class MemoryLifecycleStatus:
    """What the runtime may say about persistent memory, to anyone."""

    state: MemoryLifecycleState
    mode: Optional[MemoryStorageMode] = None
    vector_search: bool = False

    @property
    def detail(self) -> str:
        return PUBLIC_DETAILS[self.state]

    @property
    def ready(self) -> bool:
        return self.state is MemoryLifecycleState.READY

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


@dataclass(frozen=True)
class AdmissionResult:
    """Outcome of one admission attempt.

    ``store`` is ``None`` unless admission succeeded, which is what makes
    "publish no store before admission completes" structural rather than a
    convention: there is simply nothing to publish in any other state.
    """

    status: MemoryLifecycleStatus
    store: Optional[UserIsolatedMemoryStore] = None
    vector_space_fingerprint: Optional[str] = None


class MemoryUnavailableError(RuntimeError):
    """Raised instead of handing back a store that must not be used."""

    def __init__(self, status: MemoryLifecycleStatus) -> None:
        super().__init__(status.detail)
        self.status = status


def log_operator_guidance(status: MemoryLifecycleStatus) -> None:
    """Write the detailed quiescence/repair guidance to the operator log."""
    guidance = OPERATOR_GUIDANCE.get(status.state)
    if guidance is None:
        return
    if status.state in {
        MemoryLifecycleState.BLOCKED_REPAIR,
        MemoryLifecycleState.RESTART_REQUIRED,
        MemoryLifecycleState.CREDENTIAL_UNAVAILABLE,
    }:
        logger.error("Persistent memory %s. %s", status.state.value, guidance)
    else:
        logger.warning("Persistent memory %s. %s", status.state.value, guidance)


def default_similarity_threshold() -> float:
    configured = os.getenv("MEMORY_SIMILARITY_THRESHOLD")
    if not configured:
        return DEFAULT_SIMILARITY_THRESHOLD
    try:
        return float(configured)
    except ValueError:
        logger.warning(
            "Ignoring unparsable MEMORY_SIMILARITY_THRESHOLD; using the default"
        )
        return DEFAULT_SIMILARITY_THRESHOLD


def memory_store_dir() -> str:
    """Resolve the memory LanceDB directory, legacy location first.

    Mirrors the location rules recorded in ``CLAUDE.md``: an existing
    project-local ``memory_store/`` wins, otherwise ``<storage root>/
    memory_store``.
    """
    legacy_dir = Path(__file__).resolve().parents[3] / "memory_store"
    try:
        if legacy_dir.is_dir() and any(legacy_dir.iterdir()):
            logger.info("Using legacy memory store location")
            return str(legacy_dir)
    except OSError:
        pass
    new_dir = get_storage_root() / "memory_store"
    new_dir.mkdir(parents=True, exist_ok=True)
    return str(new_dir)


def authority_embedding_config(
    snapshot: GlobalMemoryEmbeddingAuthoritySnapshot,
) -> EmbeddingModelConfig:
    """Build the runtime embedding config from the global authority alone."""
    return EmbeddingModelConfig(
        id=AUTHORITY_CONFIG_ID,
        model_name=snapshot.model_name,
        model_provider=snapshot.provider,
        base_url=snapshot.endpoint,
        api_key=snapshot.api_key.get_secret_value(),
        dimension=snapshot.dimension,
        instruct=snapshot.instruct,
        max_retries=snapshot.max_retries,
    )


def _capability_status(
    capabilities: MemoryStorageCapabilities,
) -> MemoryLifecycleStatus:
    return MemoryLifecycleStatus(
        state=MemoryLifecycleState.READY,
        mode=capabilities.mode,
        vector_search=capabilities.vector_search,
    )


def _admit_once(
    connection: Any,
    identity: Any,
    *,
    writers_quiesced: bool,
    lock_timeout: float,
) -> StorageAdmissionOutcome:
    dormant = DormantLanceDBMemoryHandle(connection, MEMORY_TABLE_NAME)
    return admit_lancedb_memory_storage(
        dormant,
        identity,
        writers_quiesced=writers_quiesced,
        lock_timeout=lock_timeout,
    )


def admit_authority_storage(
    snapshot: GlobalMemoryEmbeddingAuthoritySnapshot,
    *,
    db_dir: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    writers_quiesced: bool = True,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    embedding_factory: EmbeddingFactory = create_embedding_adapter,
) -> AdmissionResult:
    """Admit the LanceDB memory storage for ``snapshot`` and build its store.

    The caller owns quiescence: this runs at startup, before any store is
    published, which is the only moment at which ``writers_quiesced`` is
    truthfully ``True`` for this process.
    """
    # A caller that asks for an unbounded wait has a bug; surface it here
    # rather than letting the admission primitive raise mid-flow, where the
    # broad handler below would report it as a transient backend failure.
    validate_lock_timeout(lock_timeout)
    try:
        config = authority_embedding_config(snapshot)
        identity = canonical_embedding_identity(config)
    except ValueError:
        # A stored authority that no longer canonicalizes needs an administrator
        # to re-set it; retrying in-process would never change the answer.
        #
        # Logged without the exception payload on purpose: a pydantic
        # validation error echoes the offending input, and one of the fields
        # being validated here is the credential.
        logger.error("Global memory embedding authority is not usable")
        return AdmissionResult(
            MemoryLifecycleStatus(MemoryLifecycleState.RESTART_REQUIRED)
        )

    # Taken from the canonical identity, never from the raw snapshot. The same
    # ``identity`` builds the embedding adapter and the stored LanceDB
    # vector-space metadata below, and the manager's drift check compares this
    # value against a freshly canonicalized authority, so all five key on one
    # vector space. Fingerprinting the raw snapshot instead let two spellings
    # of the same space disagree and turned a cosmetic edit into a demand for
    # an all-worker restart.
    fingerprint = embedding_identity_fingerprint(identity)

    directory = db_dir if db_dir is not None else memory_store_dir()
    threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else default_similarity_threshold()
    )

    try:
        connection = get_connection(directory)
        outcome = _admit_once(
            connection,
            identity,
            writers_quiesced=writers_quiesced,
            lock_timeout=lock_timeout,
        )
        if outcome.state is StorageAdmissionState.ABSENT:
            # First boot against this directory: create the vector-capable
            # table under the same admission lock, then admit the table we
            # just created. Table creation is a lifecycle step, which is why
            # it happens here and never on a request path.
            create_or_recreate_vector_capable_table(
                connection, MEMORY_TABLE_NAME, identity
            )
            outcome = _admit_once(
                connection,
                identity,
                writers_quiesced=writers_quiesced,
                lock_timeout=lock_timeout,
            )
    except Exception:
        logger.exception("Persistent memory admission failed")
        return AdmissionResult(
            MemoryLifecycleStatus(MemoryLifecycleState.RETRYABLE_UNAVAILABLE)
        )

    state = _ADMISSION_STATES[outcome.state]
    if state is not MemoryLifecycleState.READY or outcome.admitted is None:
        logger.warning("Persistent memory admission returned %s", outcome.state.value)
        return AdmissionResult(MemoryLifecycleStatus(state))

    capabilities = outcome.admitted.capabilities
    # Only a VECTOR-mode admission may carry the authority's embedding adapter.
    #
    # In TEXT_ONLY mode the stored vectors were written under a different
    # identity than the authority now describes. Handing the adapter to the
    # store there would make the *first ordinary write* rewrite the table to
    # the authority's width and re-embed every historical row under the new
    # model -- irreversibly, and without any operator action. That is exactly
    # the online re-embedding the lifecycle forbids, and it would happen in the
    # state the runtime reports as ``text_only`` with vector search off.
    #
    # Passing ``None`` is what layer B's TEXT_ONLY contract actually means:
    # ``LanceDBMemoryStore`` keeps the table writable, stores new notes without
    # vectors, leaves existing vectors untouched, and answers searches from the
    # lexical fallback. Restoring vector search is an offline re-embed plus an
    # all-worker restart, never a side effect of a user write.
    try:
        embedding_model = (
            embedding_factory(config)
            if capabilities.mode is MemoryStorageMode.VECTOR
            else None
        )
        store = LanceDBMemoryStore(
            db_dir=directory,
            collection_name=MEMORY_TABLE_NAME,
            embedding_model=embedding_model,
            similarity_threshold=threshold,
        )
    except Exception:
        # Admission certified the table, but this worker cannot build the
        # adapter for it. Nothing is published, so no caller can observe a
        # half-built store.
        logger.exception("Persistent memory store construction failed")
        return AdmissionResult(
            MemoryLifecycleStatus(MemoryLifecycleState.RETRYABLE_UNAVAILABLE)
        )

    return AdmissionResult(
        _capability_status(capabilities),
        store=UserIsolatedMemoryStore(store),
        vector_space_fingerprint=fingerprint,
    )


def credential_failure_result() -> AdmissionResult:
    """The result for an authority whose credential cannot be used."""
    return AdmissionResult(
        MemoryLifecycleStatus(MemoryLifecycleState.CREDENTIAL_UNAVAILABLE)
    )


__all__ = [
    "AUTHORITY_CONFIG_ID",
    "FENCED_DETAIL",
    "FENCED_STATES",
    "MEMORY_TABLE_NAME",
    "OPERATOR_GUIDANCE",
    "PUBLIC_DETAILS",
    "TERMINAL_STATES",
    "AdmissionResult",
    "AuthorityCredentialUnavailable",
    "MemoryLifecycleState",
    "MemoryLifecycleStatus",
    "MemoryUnavailableError",
    "admit_authority_storage",
    "authority_embedding_config",
    "credential_failure_result",
    "default_similarity_threshold",
    "log_operator_guidance",
    "memory_store_dir",
]
