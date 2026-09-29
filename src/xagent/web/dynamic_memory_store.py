"""Memory store manager for the web application.

The manager owns one thing: it admits the persistent memory storage once, and
publishes a store only if that admission succeeded. It deliberately does *not*
reload configuration online -- a changed authority, a changed vector space or
invalid legacy data all require quiescence, offline repair where applicable,
and an all-worker restart. See :mod:`xagent.web.memory_lifecycle` for the
contracts and for the operator guidance this module logs.

Admission is startup-only. :meth:`DynamicMemoryStoreManager.admit` is the sole
entry point; request and status paths never admit, because admission's repair
path rewrites the table and cannot fence the rest of the fleet. Every entry
point that needs persistent memory must therefore call it during its own
startup -- the FastAPI app does, and so does ``xagent.web.worker``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional, Union

from ..core.memory.in_memory import InMemoryMemoryStore
from ..core.memory.lancedb import LanceDBMemoryStore
from .memory_lifecycle import (
    AdmissionResult,
    AuthorityCredentialUnavailable,
    MemoryLifecycleState,
    MemoryLifecycleStatus,
    MemoryUnavailableError,
    admit_authority_storage,
    credential_failure_result,
    default_similarity_threshold,
    log_operator_guidance,
)
from .models.database import get_db
from .revocable_memory_store import RevocableMemoryStore, unwrap_memory_store
from .services.db_runtime import is_database_pool_timeout
from .services.global_memory_embedding_authority import (
    GlobalMemoryEmbeddingAuthorityService,
    GlobalMemoryEmbeddingAuthoritySnapshot,
)
from .user_isolated_memory import UserIsolatedMemoryStore

logger = logging.getLogger(__name__)

# Type alias for our memory store types that includes user isolation and the
# revocation wrapper the manager publishes.
MemoryStoreType = Union[
    InMemoryMemoryStore,
    LanceDBMemoryStore,
    UserIsolatedMemoryStore,
    RevocableMemoryStore,
]


class AuthorityUnreadable(RuntimeError):
    """The authority row could not be read for a transient reason.

    Kept distinct from :class:`AuthorityCredentialUnavailable` so a database
    hiccup is never mistaken for an identity change or a credential failure.
    """


def _read_authority_snapshot() -> Optional[GlobalMemoryEmbeddingAuthoritySnapshot]:
    """Read the global authority, or ``None`` when none is configured.

    Raises :class:`AuthorityCredentialUnavailable` when the stored credential
    cannot be used, and :class:`AuthorityUnreadable` for a transient database
    failure. A connection-pool timeout is deliberately left to propagate: pool
    exhaustion is a deployment fault that must stay loud, not be reported as a
    quiet memory outage.
    """
    try:
        db = next(get_db())
    except Exception as error:  # pragma: no cover - session factory failure
        if is_database_pool_timeout(error):
            raise
        raise AuthorityUnreadable("memory authority session unavailable") from error
    try:
        return GlobalMemoryEmbeddingAuthorityService(db).load_snapshot()
    except AuthorityCredentialUnavailable:
        raise
    except Exception as error:
        if is_database_pool_timeout(error):
            raise
        raise AuthorityUnreadable("memory authority read failed") from error
    finally:
        db.close()


@dataclass(frozen=True)
class _Publication:
    """A store that admission has certified, bound to the space it was built for.

    Holding the store and the vector-space fingerprint together is what makes
    "never return a stale adapter" structural: the only way to reach the store
    is through a revalidation that compares this fingerprint and drops the
    whole publication when it no longer matches.
    """

    store: MemoryStoreType
    status: MemoryLifecycleStatus
    vector_space_fingerprint: Optional[str]


class DynamicMemoryStoreManager:
    """Admits persistent memory storage once and publishes the result."""

    def __init__(self, similarity_threshold: Optional[float] = None):
        """
        Args:
            similarity_threshold: Optional similarity threshold for vector search.
        """
        self._similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else default_similarity_threshold()
        )
        self._lock = threading.RLock()
        self._publication: Optional[_Publication] = None
        # Moves whenever a publication is established or revoked. Every store
        # handed out captures the value it was published under, which is what
        # lets an already-distributed reference be revoked -- see
        # :mod:`xagent.web.revocable_memory_store`.
        self._generation = 0
        # Nothing is published before admission runs, so the honest starting
        # point is "not available yet, an attempt is still owed".
        self._status = MemoryLifecycleStatus(MemoryLifecycleState.RETRYABLE_UNAVAILABLE)

    @property
    def _settled(self) -> bool:
        """True once admission reached an outcome no retry can change."""
        return self._status.ready or self._status.terminal

    # -- the slice RevocableMemoryStore depends on -------------------------

    def publication_generation(self) -> int:
        """The generation a reference must still match to be usable."""
        return self._generation

    def published_status(self) -> MemoryLifecycleStatus:
        """The status a revoked reference reports (``restart_required`` on drift)."""
        return self._status

    def _publish_locked(self, publication: Optional[_Publication]) -> None:
        """Install or revoke the publication and move the generation.

        The status is set by the caller *before* this runs, so a lock-free
        reader that sees the new generation also sees the status explaining it.
        """
        self._publication = publication
        self._generation += 1

    def _revocable(self, store: MemoryStoreType) -> MemoryStoreType:
        """Bind ``store`` to the generation it is about to be published under."""
        return RevocableMemoryStore(store, self, self._generation + 1)

    def admit(self, *, writers_quiesced: bool = True) -> MemoryLifecycleStatus:
        """Run storage admission once, publishing only if it succeeds.

        This is the *only* entry point that admits. Call it from an entry
        point's startup sequence, before any request is served or any task is
        claimed: that is the moment at which this process truly has no writers.
        Quiescing the rest of the fleet is the operator's job, and the contract
        that makes it safe is an all-worker restart, not a rolling one.

        Admission never happens lazily on a request or status path. The repair
        path inside admission reads the table, stages it, and rewrites it with
        ``mode="overwrite"``; ``writers_quiesced=True`` is only a claim about
        *this* process and fences no other worker. Running that from a request
        would let one worker's repair silently destroy a commit another worker
        had already made -- the version check that would catch it only runs
        after the overwrite. A worker whose startup admission failed therefore
        stays fenced until an operator restarts it, which is exactly what
        ``docs/deployment.md`` already prescribes for ``retryable_unavailable``.
        """
        with self._lock:
            if self._settled:
                return self._status
            self._admit_locked(writers_quiesced=writers_quiesced)
            return self._status

    def _admit_locked(self, *, writers_quiesced: bool) -> None:
        previous_publication = self._publication
        try:
            snapshot = _read_authority_snapshot()
        except AuthorityCredentialUnavailable:
            result = credential_failure_result()
        except AuthorityUnreadable:
            logger.warning(
                "Persistent memory authority could not be read; admission deferred"
            )
            result = AdmissionResult(
                MemoryLifecycleStatus(MemoryLifecycleState.RETRYABLE_UNAVAILABLE)
            )
        else:
            if snapshot is None:
                self._publish_in_memory_locked()
                return
            result = admit_authority_storage(
                snapshot,
                similarity_threshold=self._similarity_threshold,
                writers_quiesced=writers_quiesced,
            )

        if result.store is None:
            # A failed admission publishes nothing and takes nothing away: a
            # manager that is already serving an admitted store keeps serving
            # it, and its status keeps describing that publication rather than
            # the attempt that just failed. The generation deliberately does
            # not move -- references handed out under that publication stay
            # valid, because nothing about it changed.
            self._publication = previous_publication
            self._status = (
                previous_publication.status
                if previous_publication is not None
                else result.status
            )
            log_operator_guidance(result.status)
            return

        self._status = result.status
        self._publish_locked(
            _Publication(
                store=self._revocable(result.store),
                status=result.status,
                vector_space_fingerprint=result.vector_space_fingerprint,
            )
        )
        logger.info(
            "Persistent memory admitted in %s mode (vector search: %s)",
            result.status.mode.value if result.status.mode else "unknown",
            result.status.vector_search,
        )

    def _publish_in_memory_locked(self) -> None:
        """No authority configured: serve an ephemeral, non-persistent store."""
        status = MemoryLifecycleStatus(MemoryLifecycleState.NOT_CONFIGURED)
        self._status = status
        self._publish_locked(
            _Publication(
                store=self._revocable(UserIsolatedMemoryStore(InMemoryMemoryStore())),
                status=status,
                vector_space_fingerprint=None,
            )
        )
        log_operator_guidance(status)

    def _revalidated_publication_locked(self) -> Optional[_Publication]:
        """Return the publication only while it still matches the authority."""
        publication = self._publication
        if publication is None or publication.vector_space_fingerprint is None:
            # Nothing published, or the ephemeral store, which no authority
            # change can invalidate. Configuring one takes effect on restart.
            return publication
        try:
            snapshot = _read_authority_snapshot()
        except AuthorityCredentialUnavailable:
            # A rotated or broken credential does not change what the stored
            # vectors mean, so it is not drift. The next restart re-admits.
            return publication
        except AuthorityUnreadable:
            # A transient database failure is not an identity change.
            return publication

        fingerprint = (
            snapshot.vector_space_fingerprint() if snapshot is not None else None
        )
        if fingerprint == publication.vector_space_fingerprint:
            return publication

        # Meaningful drift: the authority now describes a different vector
        # space than the one these vectors were written under. Drop the
        # publication before returning, so there is no path by which a caller
        # can still be handed the adapter built for the previous space --
        # and move the generation, which fails closed every reference already
        # handed out, including one a cached agent is using mid-execution.
        self._status = MemoryLifecycleStatus(MemoryLifecycleState.RESTART_REQUIRED)
        self._publish_locked(None)
        log_operator_guidance(self._status)
        return None

    def acquire(self) -> tuple[Optional[MemoryStoreType], MemoryLifecycleStatus]:
        """Return the live store and status. Never admits.

        Read-only with respect to memory storage: the only thing this may do
        beyond returning what startup published is *revoke* it, by re-reading
        the authority row and comparing vector-space fingerprints. That read
        touches the database, never the memory LanceDB directory. A worker that
        never ran :meth:`admit`, or whose admission failed, returns ``None``
        here for the rest of its life.
        """
        with self._lock:
            publication = self._revalidated_publication_locked()
            if publication is None:
                return None, self._status
            return publication.store, publication.status

    def get_memory_store(self) -> MemoryStoreType:
        """Return the published store, or fail closed.

        Raises:
            MemoryUnavailableError: when no store may be served.
        """
        store, status = self.acquire()
        if store is None:
            raise MemoryUnavailableError(status)
        return store

    def status(self) -> MemoryLifecycleStatus:
        """Current public-safe lifecycle status. Never admits."""
        return self.acquire()[1]

    def force_reinitialize(self) -> None:
        """Retained no-op: rebuilding a live store is an online reload.

        Persistent memory changes take effect through quiescence and an
        all-worker restart, never by swapping the store underneath callers.
        """
        logger.warning(
            "Ignoring memory store reinitialization request: persistent memory "
            "changes require quiescence and an all-worker restart"
        )

    def check_embedding_model_change(self) -> bool:
        """Re-check the authority for vector-space drift.

        Never rebuilds: a meaningful change revokes the publication and leaves
        the runtime asking for a restart.

        Returns:
            True if the publication was revoked by this check.
        """
        with self._lock:
            before = self._publication
            self._revalidated_publication_locked()
            return before is not self._publication

    def get_store_info(self) -> dict:
        """Public-safe description of the current memory store.

        Degrades instead of raising. This is the one report an operator has
        for seeing *why* memory is fenced off, and ``docs/deployment.md``
        promises it answers in every state, so a database fault on the drift
        read must not turn the lifecycle report itself into an error. When the
        authority cannot be read, the last published status and state are
        reported as they stand.

        Deliberately confined to this boundary. ``_read_authority_snapshot``
        still re-raises a connection-pool timeout, and the drift check on the
        memory path still propagates it: pool exhaustion is a deployment fault
        that has to stay loud for the callers that would otherwise read and
        write through a store this manager could not revalidate. A status
        report is the one caller for which answering beats failing.
        """
        with self._lock:
            try:
                store, status = self.acquire()
            except Exception:
                logger.warning(
                    "Persistent memory authority could not be read for the "
                    "store-info report; reporting the last published state",
                    exc_info=True,
                )
                publication = self._publication
                store = publication.store if publication is not None else None
                status = self._status
            threshold = self._similarity_threshold
        # Report what the storage adapter actually is, through both the
        # revocation and the user-isolation wrapper.
        base_store = unwrap_memory_store(store)
        return {
            "store_type": type(base_store).__name__ if base_store is not None else None,
            "is_lancedb": isinstance(base_store, LanceDBMemoryStore),
            "state": status.state.value,
            "detail": status.detail,
            "mode": status.mode.value if status.mode is not None else None,
            "supports_vector_search": status.vector_search,
            "similarity_threshold": threshold,
        }


# Global instance
_dynamic_manager: Optional[DynamicMemoryStoreManager] = None
_manager_lock = threading.Lock()


def get_memory_store_manager(
    similarity_threshold: Optional[float] = None,
) -> DynamicMemoryStoreManager:
    """Get or create the global memory store manager."""
    global _dynamic_manager

    if _dynamic_manager is None:
        with _manager_lock:
            if _dynamic_manager is None:
                _dynamic_manager = DynamicMemoryStoreManager(similarity_threshold)

    return _dynamic_manager


def admit_memory_storage() -> MemoryLifecycleStatus:
    """Run startup admission for the process-wide manager."""
    return get_memory_store_manager().admit()


def admit_memory_storage_at_startup() -> Optional[MemoryLifecycleStatus]:
    """Admit persistent memory during an entry point's startup, never raising.

    Shared by every entry point that serves memory -- the FastAPI app and the
    shared task worker -- so they cannot drift apart on the two things that
    matter: memory may never abort a boot, and a non-ready outcome must reach
    the operator log. Returns ``None`` when admission itself raised.
    """
    try:
        status = admit_memory_storage()
        store_info = get_memory_store_manager().get_store_info()
    except Exception:
        # Memory must never be able to abort a boot. Unrelated functions start;
        # memory itself stays closed until an operator acts.
        logger.exception("Persistent memory admission raised; memory stays unavailable")
        return None

    if status.ready:
        logger.info(
            "Persistent memory admitted in %s mode (vector search: %s)",
            status.mode.value if status.mode else "unknown",
            status.vector_search,
        )
    else:
        logger.warning(
            "Persistent memory is not serving (%s); unrelated functions "
            "continue to start",
            status.state.value,
        )
    logger.info(
        "Memory store similarity threshold: %s", store_info["similarity_threshold"]
    )
    return status


def memory_store_status() -> MemoryLifecycleStatus:
    """Current public-safe lifecycle status of persistent memory."""
    return get_memory_store_manager().status()


def get_memory_store() -> MemoryStoreType:
    """Get the current memory store, failing closed when none may be served."""
    return get_memory_store_manager().get_memory_store()


def force_reinitialize_memory_store() -> None:
    """Retained no-op; see :meth:`DynamicMemoryStoreManager.force_reinitialize`."""
    get_memory_store_manager().force_reinitialize()
