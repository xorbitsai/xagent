"""Switch one collection to a new embedding model inside a freeze window.

The window is: freeze the collection, wait for the writers the freeze let in
(running ingest jobs and live write leases), copy what the target model still
lacks (chunks ingested since the precompute), prune what the source no longer
has, check the target's indexes and the per-row acceptance, wait once more for
writers, and flip the binding. A refusal restores the row that was there
before; a crash leaves the collection frozen, and a rerun resumes the window
only when it owns it or its heartbeat stopped.

    python -m xagent.migrations.embedding.switch --model <model-id> \
        --collection <name> [--wait-seconds N]
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    IngestionConfig,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.core.tools.core.RAG_tools.management.embedding_switch import (
    switch_collection_embedding_model,
)
from xagent.core.tools.core.RAG_tools.management.migration_state import (
    FROZEN,
    IDLE,
    CollectionMigrationState,
    claim_freeze,
    heartbeat_freeze,
    read_migration_state,
    read_raw_state,
    restore_state,
    write_migration_state,
)
from xagent.core.tools.core.RAG_tools.management.write_leases import (
    live_leases,
)
from xagent.core.tools.core.RAG_tools.storage.factory import get_vector_index_store
from xagent.core.tools.core.RAG_tools.utils.string_utils import escape_lancedb_string
from xagent.migrations.embedding.acceptance import AcceptanceResult, check_collection
from xagent.migrations.embedding.embed import (
    PacedEmbedding,
    RateLimiter,
    embed_collection,
    resolve_target_model,
)
from xagent.migrations.embedding.scope import (
    METADATA_TABLE,
    MigrationCliError,
    MissingCollectionError,
    add_scope_args,
    collection_filter,
    collection_info,
    connection,
    iter_rows,
    preflight_source_target,
    resolve_collection_names,
    source_eligibility,
    source_table,
    table_names,
)

logger = logging.getLogger(__name__)

DEFAULT_WAIT_SECONDS = 15 * 60
DEFAULT_POLL_SECONDS = 2.0
# Writers admitted in the race before the freeze can still land; re-check right
# before the flip so they delay the switch instead of racing it.
FINAL_LEASE_WAIT_SECONDS = 30.0
# A window renews its row every few seconds; a row that has not moved for this
# long has no live owner and may be resumed. The hour figure is only an alarm.
HEARTBEAT_SECONDS = 15.0
STALE_FREEZE_SECONDS = 5 * HEARTBEAT_SECONDS
STALE_ALARM_SECONDS = 60 * 60

RESULT_SWITCHED = "switched"
RESULT_ALREADY = "already-switched"
RESULT_REFUSED = "refused"
RESULT_FAILED = "failed"
RESULT_INTERRUPTED = "interrupted-restored"
RESULT_NOT_STARTED = "not-started"
RESULT_RUNNING_AT_EXIT = "running-at-exit"

MODE_SWITCH = "switch"
MODE_ROLLBACK = "rollback"


class WindowRefused(Exception):
    """The collection stays on the old model; ``reason`` goes in the report."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class SwitchReport:
    name: str
    target_model_id: str
    mode: str = MODE_SWITCH
    result: str = RESULT_REFUSED
    reason: str = ""
    source_model_id: Optional[str] = None
    window_seconds: float = 0.0
    documents_copied: int = 0
    rows_copied: int = 0
    rows_pruned: int = 0
    excluded_rows: Dict[str, int] = field(default_factory=dict)
    acceptance: Dict[str, Any] = field(default_factory=dict)


class InflightJobWatcher:
    """Running KB ingest jobs for a collection, read from the web database."""

    def __init__(self) -> None:
        self._session_factory: Any = None
        self._init_lock = threading.Lock()

    def _session(self) -> Any:
        if self._session_factory is None:
            with self._init_lock:
                if self._session_factory is None:
                    from xagent.web.models.database import (
                        configure_db,
                        get_session_local,
                    )

                    configure_db(read_only=True)
                    self._session_factory = get_session_local()
        return self._session_factory()

    def running(self, collection: str) -> List[str]:
        from xagent.web.models.background_job import (
            BackgroundJob,
            BackgroundJobStatus,
            BackgroundJobType,
        )

        db = self._session()
        try:
            rows = (
                db.query(BackgroundJob)
                .filter(
                    BackgroundJob.status == BackgroundJobStatus.RUNNING.value,
                    BackgroundJob.job_type.in_(
                        [
                            BackgroundJobType.KB_INGEST_DOCUMENT.value,
                            BackgroundJobType.KB_INGEST_WEB.value,
                        ]
                    ),
                )
                .all()
            )
            return [
                str(row.id)
                for row in rows
                if str(dict(row.payload or {}).get("collection") or "") == collection
            ]
        finally:
            db.close()


def _run_owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def switch_collection(
    name: str,
    target_model_id: str,
    *,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    job_watcher: Optional[InflightJobWatcher] = None,
    acceptance: Optional[Callable[..., AcceptanceResult]] = None,
    mode: str = MODE_SWITCH,
    cfg: Optional[IngestionConfig] = None,
    rate_limit: float = 0.0,
    rate_limiter: Optional[RateLimiter] = None,
    owner: Optional[str] = None,
) -> SwitchReport:
    """Run the switch window for one collection.

    A ``KeyboardInterrupt`` inside the window first restores the row (before the
    flip) or records the switch (after it), then re-raises as
    :class:`WindowInterrupted` carrying the report.
    """
    started = time.monotonic()
    owner = owner or _run_owner()
    acceptance = acceptance or check_collection
    report = SwitchReport(name=name, target_model_id=target_model_id, mode=mode)

    def finish(result: str, reason: str = "") -> SwitchReport:
        report.result = result
        report.reason = reason
        report.window_seconds = round(time.monotonic() - started, 3)
        return report

    try:
        info = _info(name)
    except Exception as exc:  # noqa: BLE001 - nothing frozen yet: refuse
        return finish(RESULT_REFUSED, f"metadata-unreadable: {exc}")
    if info is None:
        return finish(RESULT_REFUSED, "unknown-collection")
    eligibility = source_eligibility(name)
    if eligibility.reason:
        # The same rule precompute uses: a collection whose rows are not in its
        # binding's writer table is never switched either.
        return finish(RESULT_REFUSED, eligibility.reason)
    source = eligibility.model_id
    report.source_model_id = source

    try:
        embedding_config, adapter = resolve_target_model(target_model_id)
    except MigrationCliError as exc:
        return finish(RESULT_REFUSED, str(exc))
    except Exception as exc:  # noqa: BLE001 - provider/hub failure, not a window
        return finish(RESULT_REFUSED, f"target-not-configured: {exc}")
    target = embedding_config.id.strip()

    previous_row = read_raw_state(name)
    previous = read_migration_state(name)
    if previous.state == FROZEN:
        refusal = _frozen_refusal(previous, target, mode)
        if refusal:
            logger.error(
                "%s: %s. A window frozen for another target or mode is released "
                "by finishing it (run that command with the same --model), or, "
                "if its run is gone, by clearing the row after its heartbeat "
                "stops: collection_migration_state row owner=%s state=%s.",
                name,
                refusal,
                previous.owner or "unknown",
                previous.state,
            )
            return finish(RESULT_REFUSED, refusal)
    if mode == MODE_ROLLBACK:
        refusal = _rollback_refusal(previous, source, target)
        if refusal:
            return finish(RESULT_REFUSED, refusal)

    if source == target:
        if previous.state != FROZEN:
            return finish(RESULT_ALREADY)
        # A window that flipped and then died before recording it.
        if not _may_resume(previous, owner):
            return finish(RESULT_REFUSED, f"window-in-progress: {previous.owner}")
        _record_switch(name, previous.source_model_id, target, previous.mode or mode)
        return finish(RESULT_SWITCHED, "recovered-record")

    try:
        # The precompute's own rule. Distinct ids that share a writer table are
        # not "already switched": the copy would be its own source.
        preflight_source_target(source, target)
    except MigrationCliError as exc:
        return finish(RESULT_REFUSED, str(exc))

    claim = claim_freeze(
        name,
        source_model_id=source,
        target_model_id=target,
        mode=mode,
        owner=owner,
        stale_after_seconds=STALE_FREEZE_SECONDS,
    )
    if claim == "busy":
        return finish(RESULT_REFUSED, f"window-in-progress: {previous.owner}")
    if claim == "resumed":
        logger.warning(
            "%s: resuming a frozen window (owner %s, last heartbeat %s)",
            name,
            previous.owner or "unknown",
            previous.updated_at,
        )
        if _age_seconds(previous.updated_at) > STALE_ALARM_SECONDS:
            logger.error(
                "%s: this window has been frozen for over %.0f minutes; check "
                "with the operator before continuing",
                name,
                STALE_ALARM_SECONDS / 60,
            )
    resumed = claim == "resumed"

    heartbeat = _Heartbeat(name, owner)
    try:
        with heartbeat:
            _run_window(
                report,
                name,
                source=source,
                target=target,
                embedding_config=embedding_config,
                adapter=PacedEmbedding(
                    adapter, rate_limiter or RateLimiter(rate_limit)
                ),
                cfg=cfg or IngestionConfig(embedding_model_id=target),
                wait_seconds=wait_seconds,
                poll_seconds=poll_seconds,
                watcher=job_watcher or InflightJobWatcher(),
                acceptance=acceptance,
                heartbeat=heartbeat,
            )
    except WindowRefused as exc:
        _restore(name, owner, previous_row, source, resumed=resumed)
        return finish(RESULT_REFUSED, exc.reason)
    except KeyboardInterrupt:
        settled = _settle_after_failure(
            name,
            source,
            target,
            mode,
            owner,
            previous_row,
            resumed,
            report,
            interrupted=True,
        )
        if settled == RESULT_SWITCHED:
            finish(RESULT_SWITCHED, report.reason or "interrupted-after-flip")
        elif settled == RESULT_FAILED:
            finish(RESULT_FAILED, report.reason)
        else:
            finish(RESULT_INTERRUPTED)
        raise WindowInterrupted(report) from None
    except Exception as exc:  # noqa: BLE001 - keep the collection usable
        settled = _settle_after_failure(
            name,
            source,
            target,
            mode,
            owner,
            previous_row,
            resumed,
            report,
            interrupted=False,
        )
        if settled == RESULT_SWITCHED:
            return finish(RESULT_SWITCHED, report.reason or "interrupted-after-flip")
        logger.exception("%s: switch failed", name)
        return finish(
            RESULT_FAILED,
            report.reason or f"{type(exc).__name__}: {exc}",
        )

    _record_switch(name, source, target, mode, report=report)
    return finish(RESULT_SWITCHED, report.reason)


class WindowInterrupted(KeyboardInterrupt):
    """Ctrl-C inside a window, after its row was restored or its switch recorded."""

    def __init__(self, report: SwitchReport) -> None:
        super().__init__(report.name)
        self.report = report


class _Heartbeat:
    """Renews the frozen row while the window runs; notices a lost ownership."""

    def __init__(self, name: str, owner: str) -> None:
        self.name = name
        self.owner = owner
        self.lost = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"kb-switch-heartbeat-{name}", daemon=True
        )

    def __enter__(self) -> "_Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=HEARTBEAT_SECONDS)

    def beat(self) -> bool:
        try:
            if not heartbeat_freeze(self.name, self.owner):
                self.lost = True
        except Exception:  # noqa: BLE001 - an unrenewable row is a lost one
            logger.exception("%s: heartbeat failed", self.name)
            self.lost = True
        return not self.lost

    def _run(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            if not self.beat():
                return


def _bound_model(info: Any) -> Optional[str]:
    """The collection's own binding, the model eligibility copied from."""
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        _normalize_model_id,
    )

    return _normalize_model_id(getattr(info, "embedding_model_id", None))


def _info(name: str) -> Any:
    """The metadata row, ``None`` only when it is absent; a failed read raises."""
    try:
        return collection_info(name)
    except MissingCollectionError:
        return None


def _settle_after_failure(
    name: str,
    source: str,
    target: str,
    mode: str,
    owner: str,
    previous_row: Optional[Dict[str, Any]],
    resumed: bool,
    report: SwitchReport,
    *,
    interrupted: bool,
) -> str:
    """After an exception: record the flip, restore, or leave it frozen.

    An unreadable binding is never "not flipped": restoring the old row over a
    flipped collection would hide every chunk. Left frozen, the next run
    recovers it from the row this run still owns.
    """
    bound = _bound_to(name)
    if bound == target:
        _record_switch(name, source, target, mode, report=report)
        return RESULT_SWITCHED
    if bound is not None:
        _restore(name, owner, previous_row, source, resumed=resumed)
        return RESULT_INTERRUPTED if interrupted else RESULT_FAILED
    logger.error(
        "%s: binding unreadable after the failure; leaving the collection "
        "frozen for the next run (owner %s)",
        name,
        owner,
    )
    report.reason = "binding-unreadable: collection left frozen"
    return RESULT_FAILED


def _bound_to(name: str, attempts: int = 3) -> Optional[str]:
    """The collection's binding after the flip, or ``None`` when unreadable."""
    for attempt in range(attempts):
        try:
            info = _info(name)
            return None if info is None else _bound_model(info)
        except Exception:  # noqa: BLE001 - retry: a failed read is not "unflipped"
            logger.warning(
                "%s: could not read the binding (attempt %d/%d)",
                name,
                attempt + 1,
                attempts,
            )
            time.sleep(0.1 * (attempt + 1))
    return None


def _may_resume(state: CollectionMigrationState, owner: str) -> bool:
    if state.owner == owner:
        return True
    return _age_seconds(state.updated_at) > STALE_FREEZE_SECONDS


def _frozen_refusal(
    state: CollectionMigrationState, target: str, mode: str
) -> Optional[str]:
    """A frozen row may only be resumed by a window of the same target and mode."""
    frozen_target = (state.target_model_id or "").strip()
    if frozen_target and frozen_target != target:
        return f"frozen-for-another-target: {frozen_target}"
    if (state.mode or MODE_SWITCH) != mode:
        return f"frozen-for-{state.mode or MODE_SWITCH}"
    return None


def _rollback_refusal(
    previous: CollectionMigrationState, source: str, target: str
) -> Optional[str]:
    """Rollback only undoes a recorded switch of this collection back to ``target``.

    A frozen rollback window carries the same shape: source = the switched-to
    model, target = the model it switched from.
    """
    if previous.state == FROZEN:
        if previous.mode == MODE_ROLLBACK and previous.target_model_id == target:
            return None
        return "no-recorded-switch"
    if previous.mode != MODE_SWITCH or previous.target_model_id != source:
        return "no-recorded-switch"
    if previous.source_model_id != target:
        return f"recorded-source-is-{previous.source_model_id}"
    return None


def _run_window(
    report: SwitchReport,
    name: str,
    *,
    source: str,
    target: str,
    embedding_config: EmbeddingModelConfig,
    adapter: BaseEmbedding,
    cfg: IngestionConfig,
    wait_seconds: float,
    poll_seconds: float,
    watcher: InflightJobWatcher,
    acceptance: Callable[..., AcceptanceResult],
    heartbeat: _Heartbeat,
) -> None:
    exists, identity = collection_identity(name)
    if identity is None:
        logger.warning(
            "%s: no stored created_at; a rename during the window is only "
            "detected by the name disappearing",
            name,
        )
    stragglers = _wait_until_idle(
        name, timeout=wait_seconds, poll=poll_seconds, watcher=watcher
    )
    if stragglers:
        raise WindowRefused("in-flight-writers: " + ", ".join(stragglers[:10]))

    # A rename admitted before the freeze moves the row; compare a stored
    # identity, not the name resolving at all.
    still, current = collection_identity(name)
    if not exists or not still or current != identity:
        raise WindowRefused("renamed")

    conn = connection()
    item = embed_collection(
        name,
        model_id=target,
        embedding_config=embedding_config,
        embedding_adapter=adapter,
        cfg=cfg,
        dry_run=False,
        conn=conn,
    )
    report.documents_copied = item.documents_embedded
    report.rows_copied = item.chunks_embedded
    report.excluded_rows = dict(item.skipped)
    if item.collection_skip or item.aborted or item.failures:
        raise WindowRefused(
            "embed-failed: "
            + (item.collection_skip or f"{len(item.failures)} documents failed")
        )

    report.rows_pruned = prune_target_only(name, source, target, conn=conn)

    problem = target_index_problem(target, conn=conn)
    if problem:
        raise WindowRefused(f"target-index-missing: {problem}")

    verdict = acceptance(
        name, source_model_id=source, target_model_id=target, conn=conn
    )
    report.acceptance = dict(verdict.checks)
    if not verdict.ok:
        raise WindowRefused("acceptance-failed: " + "; ".join(verdict.failures))

    stragglers = _wait_until_idle(
        name, timeout=FINAL_LEASE_WAIT_SECONDS, poll=poll_seconds, watcher=watcher
    )
    if stragglers:
        raise WindowRefused("writers-restarted: " + ", ".join(stragglers[:10]))

    # Stop renewing before the flip and check we still own the row: a window
    # whose heartbeat lapsed may already have been resumed by another run.
    heartbeat.stop()
    if heartbeat.lost or not heartbeat.beat():
        raise WindowRefused("ownership-lost")
    switch_collection_embedding_model(name, target)


def collection_identity(name: str) -> Tuple[bool, Optional[Any]]:
    """Whether the metadata row exists, and its stored ``created_at``.

    The stored value, not ``collection_info``: a legacy row without the column
    reads back as ``now()`` there, which would look like a rename every time.
    """
    conn = connection()
    if METADATA_TABLE not in table_names(conn):
        return False, None
    rows = list(
        iter_rows(
            conn,
            METADATA_TABLE,
            f"name = '{escape_lancedb_string(name)}'",
            ["name", "created_at"],
        )
    )
    if not rows:
        return False, None
    return True, rows[0].get("created_at")


def prune_target_only(
    name: str, source: str, target: str, *, conn: Optional[Any] = None
) -> int:
    """Delete target rows whose ``(doc_id, chunk_id)`` the source no longer has.

    Both sides are read in full with the strict reader before anything is
    deleted; a failed read raises and deletes nothing. Rows acceptance excludes
    (ownerless, textless) are excluded here too: they are not part of the
    comparison, so deleting them would lose data no one measured.
    """
    conn = conn if conn is not None else connection()
    names = table_names(conn)
    source_table_name = source_table(source)
    target_table_name = source_table(target)
    if source_table_name not in names or target_table_name not in names:
        return 0
    where = collection_filter(name)
    keep = {
        (str(row.get("doc_id") or ""), str(row.get("chunk_id") or ""))
        for row in iter_rows(conn, source_table_name, where, ["doc_id", "chunk_id"])
    }
    extra: Dict[str, List[str]] = {}
    for row in iter_rows(
        conn, target_table_name, where, ["doc_id", "chunk_id", "user_id", "text"]
    ):
        doc_id = str(row.get("doc_id") or "")
        chunk_id = str(row.get("chunk_id") or "")
        if not doc_id or not chunk_id:
            continue
        if row.get("user_id") is None or not row.get("text"):
            # Excluded from acceptance on both sides: never pruned.
            continue
        if (doc_id, chunk_id) not in keep:
            extra.setdefault(doc_id, []).append(chunk_id)
    if not extra:
        return 0
    store = get_vector_index_store()
    deleted = 0
    for doc_id, chunk_ids in sorted(extra.items()):
        deleted += store.delete_embedding_records(
            name, doc_id, chunk_ids=chunk_ids, model_tag=target, is_admin=True
        )
    logger.warning("%s: pruned %d target rows the source no longer has", name, deleted)
    return deleted


def target_index_problem(target: str, *, conn: Optional[Any] = None) -> Optional[str]:
    """Why search would degrade on the target table after the flip, if it would."""
    conn = conn if conn is not None else connection()
    table_name = source_table(target)
    if table_name not in table_names(conn):
        return None
    table = conn.open_table(table_name)
    try:
        indexes = table.list_indices()
        rows = table.count_rows()
    finally:
        _safe_close_table(table)
    if rows and not any(
        idx.index_type == "FTS" and "text" in idx.columns for idx in indexes
    ):
        return "no full-text index on text"
    if rows >= IndexPolicy().enable_threshold_rows and not any(
        "vector" in idx.columns for idx in indexes
    ):
        return "no vector index"
    return None


def _wait_until_idle(
    name: str, *, timeout: float, poll: float, watcher: InflightJobWatcher
) -> List[str]:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        jobs = watcher.running(name)
        leases = live_leases(name)
        if not jobs and not leases:
            return []
        if time.monotonic() >= deadline:
            return [f"job:{job}" for job in jobs] + [
                f"lease:{lease.operation}@{lease.owner}" for lease in leases
            ]
        time.sleep(poll)


def _age_seconds(stamp: Any) -> float:
    if not isinstance(stamp, datetime):
        return 0.0
    current = datetime.now(timezone.utc)
    value = stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    return max(0.0, (current - value).total_seconds())


def _record_switch(
    name: str,
    source: Optional[str],
    target: str,
    mode: str,
    *,
    report: Optional[SwitchReport] = None,
) -> None:
    """The completed-switch record the rollback window checks; never masks the flip."""
    try:
        write_migration_state(
            name,
            CollectionMigrationState(
                state=IDLE, source_model_id=source, target_model_id=target, mode=mode
            ),
        )
        logger.info("%s: switched %s -> %s (%s)", name, source, target, mode)
    except Exception as exc:  # noqa: BLE001 - report, never mask the switch
        logger.exception("%s: switched but could not record it", name)
        if report is not None:
            report.reason = f"left-frozen: {type(exc).__name__}: {exc}"


def _restore(
    name: str,
    owner: str,
    previous_row: Optional[Dict[str, Any]],
    source: str,
    *,
    resumed: bool,
) -> None:
    """Put back the row the window replaced, only while this run still owns it."""
    try:
        if read_migration_state(name).owner != owner:
            logger.error("%s: no longer the window's owner; row left as is", name)
            return
        if not resumed:
            restore_state(name, previous_row)
            return
        # A resumed window never saw the pre-freeze row. Unfreeze without a
        # target so it never looks like a completed switch.
        write_migration_state(
            name, CollectionMigrationState(state=IDLE, source_model_id=source)
        )
    except Exception:  # noqa: BLE001 - the freeze write is what matters
        logger.exception("%s: could not unfreeze; the collection stays frozen", name)


def describe(report: SwitchReport) -> None:
    """One line per collection, then excluded rows by reason and acceptance checks."""
    logger.info(
        "%s: %s%s in %.2fs (copied %d rows in %d documents, pruned %d rows)",
        report.name,
        report.result,
        f" ({report.reason})" if report.reason else "",
        report.window_seconds,
        report.rows_copied,
        report.documents_copied,
        report.rows_pruned,
    )
    excluded = {k: v for k, v in sorted(report.excluded_rows.items()) if v}
    if excluded:
        # Not a refusal: these rows keep only their old-model vectors.
        logger.warning(
            "%s: switched with excluded rows (%s); they stay on the old model only",
            report.name,
            ", ".join(f"{k}={v}" for k, v in excluded.items()),
        )
    for key, value in sorted(report.acceptance.items()):
        logger.info("%s: acceptance %s=%s", report.name, key, value)


def switch_collections(
    target_model_id: str,
    *,
    collections: Optional[Sequence[str]] = None,
    team: Optional[int] = None,
    concurrency: int = 1,
    mode: str = MODE_SWITCH,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    cfg: Optional[IngestionConfig] = None,
    rate_limit: float = 0.0,
) -> List[SwitchReport]:
    """Run the window over every collection in scope, ``concurrency`` at a time.

    One run id owns every window, and one rate limiter paces them all. Ctrl-C
    stops new windows; the one that was interrupted reports what it did with
    its row, and the rest are reported ``not-started``.
    """
    names = resolve_collection_names(collections=collections, team=team)
    if not names:
        logger.warning("No collections in scope; check --collection/--team")
        return []
    watcher = InflightJobWatcher()
    owner = _run_owner()
    limiter = RateLimiter(rate_limit)
    stop = threading.Event()
    reports: Dict[str, SwitchReport] = {}

    def _one(name: str) -> SwitchReport:
        if stop.is_set():
            return _not_started(name, target_model_id, mode)
        try:
            return switch_collection(
                name,
                target_model_id,
                wait_seconds=wait_seconds,
                job_watcher=watcher,
                mode=mode,
                cfg=cfg,
                rate_limiter=limiter,
                owner=owner,
            )
        except WindowInterrupted as exc:
            stop.set()
            return exc.report
        except Exception as exc:  # noqa: BLE001 - one collection must not stop the rest
            logger.exception("%s: window crashed", name)
            return SwitchReport(
                name=name,
                target_model_id=target_model_id,
                mode=mode,
                result=RESULT_FAILED,
                reason=f"{type(exc).__name__}: {exc}",
            )

    workers = max(1, min(concurrency, len(names)))
    if workers == 1:
        for name in names:
            try:
                reports[name] = _one(name)
            except KeyboardInterrupt:
                # Outside any window now: start nothing more.
                stop.set()
                break
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, name): name for name in names}
            try:
                for future, name in futures.items():
                    reports[name] = future.result()
            except KeyboardInterrupt:
                logger.warning("Interrupted: finishing the windows already running")
                stop.set()
                for future in futures:
                    future.cancel()
                interrupted_again = False
                for future, name in futures.items():
                    if future.cancelled() or name in reports:
                        continue
                    try:
                        reports[name] = future.result()
                    except KeyboardInterrupt:
                        # A second Ctrl-C: this window is still running, not
                        # failed, and its own exit will settle it.
                        interrupted_again = True
                        reports[name] = _running_at_exit(name, target_model_id, mode)
                    except BaseException as exc:  # noqa: BLE001 - the crashed window
                        reports[name] = SwitchReport(
                            name=name,
                            target_model_id=target_model_id,
                            mode=mode,
                            result=RESULT_FAILED,
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                if interrupted_again:
                    for name in names:
                        reports.setdefault(
                            name, _running_at_exit(name, target_model_id, mode)
                        )
                    for name in names:
                        describe(reports[name])
                    raise
    return [
        reports.get(name) or _not_started(name, target_model_id, mode) for name in names
    ]


def _running_at_exit(name: str, target_model_id: str, mode: str) -> SwitchReport:
    return SwitchReport(
        name=name,
        target_model_id=target_model_id,
        mode=mode,
        result=RESULT_RUNNING_AT_EXIT,
        reason="second interrupt; the window settles its own row",
    )


def _not_started(name: str, target_model_id: str, mode: str) -> SwitchReport:
    return SwitchReport(
        name=name,
        target_model_id=target_model_id,
        mode=mode,
        result=RESULT_NOT_STARTED,
        reason="interrupted",
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Target embedding model id")
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=DEFAULT_WAIT_SECONDS,
        metavar="N",
        help=f"Maximum wait for in-flight writers (default {DEFAULT_WAIT_SECONDS})",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.0,
        metavar="N",
        help="Average maximum of texts embedded per second; 0 = unlimited",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        metavar="N",
        help=f"Chunks per embedding request (default {DEFAULT_EMBEDDING_BATCH_SIZE})",
    )
    parser.add_argument(
        "--use-async",
        action="store_true",
        help="Encode chunks concurrently; needed by models without batch support",
    )


def _run_config(args: argparse.Namespace) -> IngestionConfig:
    return IngestionConfig(
        embedding_model_id=args.model,
        embedding_batch_size=args.batch_size,
        embedding_use_async=args.use_async,
    )


def run_cli(mode: str, argv: Optional[List[str]], *, description: str) -> int:
    """The multi-collection entry point both commands use."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_args(parser)
    add_scope_args(parser)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        metavar="N",
        help="Collections processed at once (default 1)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    for flag, value in (
        ("--concurrency", args.concurrency),
        ("--batch-size", args.batch_size),
    ):
        if value <= 0:
            logger.error("%s must be positive", flag)
            return 2
    if not math.isfinite(args.rate_limit) or args.rate_limit < 0:
        logger.error("--rate-limit must be a finite, non-negative number")
        return 2

    try:
        reports = switch_collections(
            args.model,
            collections=args.collection,
            team=args.team,
            concurrency=args.concurrency,
            mode=mode,
            wait_seconds=args.wait_seconds,
            cfg=_run_config(args),
            rate_limit=args.rate_limit,
        )
    except MigrationCliError as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        logger.error("Migration command failed: %s", exc, exc_info=True)
        return 2

    if not reports:
        return 1
    for report in reports:
        describe(report)
    return (
        0 if all(r.result in (RESULT_SWITCHED, RESULT_ALREADY) for r in reports) else 1
    )


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(
        MODE_SWITCH,
        argv,
        description=(
            "Switch collections to a new embedding model.\n\n"
            "Precompute their vectors first (migrations.embedding.embed). Each\n"
            "collection is frozen for the duration of its window; a re-run\n"
            "resumes a window that was interrupted."
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
