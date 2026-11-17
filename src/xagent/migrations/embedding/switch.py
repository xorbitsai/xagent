"""Switch collections to a target embedding model inside a migration window.

    python -m xagent.migrations.embedding.switch --model TARGET \
        [--collection NAME ...] [--rate-limit N] [--max-fill N] \
        [--drain-timeout S] [--take-over]
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import logging
import math
import shlex
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from xagent.core.tools.core.RAG_tools.core.exceptions import (
    CollectionMigratingError,
    CollectionNotFrozenError,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.core.tools.core.RAG_tools.management.collection_manager import (
    _normalize_model_id,
)
from xagent.core.tools.core.RAG_tools.management.embedding_switch import (
    _held_state,
    switch_collection_embedding_model,
)
from xagent.core.tools.core.RAG_tools.management.migration_state import (
    FROZEN,
    MigrationWindow,
    claim_migration_window,
    read_migration_state,
    release_migration_window,
    take_over_migration_window,
)
from xagent.core.tools.core.RAG_tools.management.write_leases import (
    wait_for_no_leases,
)
from xagent.core.tools.core.RAG_tools.utils.string_utils import escape_lancedb_string
from xagent.migrations.embedding.embed import (
    FAILED_INDEX_STATUSES,
    SKIP_NAME_NOT_STRIPPED,
    WRITE_COLUMNS,
    CollectionReport,
    Target,
    TargetTableMissing,
    index_target,
    record,
    row_skip_reason,
    source_metadata,
    start_run,
    write_rows,
)
from xagent.migrations.embedding.scope import (
    SKIP_NO_METADATA,
    SKIP_OTHER_ENGINE,
    SKIP_TARGET_IS_SOURCE,
    SKIP_UNBOUND,
    MigrationCliError,
    MissingCollectionError,
    _owner,
    _usable_vector,
    chunk_key,
    collection_filter,
    collection_info,
    core_chunk_pairs,
    holds_rows,
    iter_rows,
    precompute_eligibility,
    source_table,
    table_names,
    written_by,
)

logger = logging.getLogger(__name__)

HOLDER = "embedding-switch"
SWITCHED = "switched"
ALREADY_SWITCHED = "already-switched"
REFUSED_FROZEN = "frozen"
FROZEN_ADVICE = (
    "frozen by a run that holds or left the window; rerun with --collection {name} "
    "--take-over if that run is gone"
)
FLIP_ATTEMPTS = 5
COPIED = ("text", "metadata", "chunk_hash", "parse_hash", "user_id")
NOTHING_TO_MIGRATE = (SKIP_UNBOUND, SKIP_NO_METADATA, SKIP_OTHER_ENGINE)

Key = Tuple[str, str]
Entry = Tuple[bytes, bool]
Rows = Dict[Key, List[Entry]]


class Refused(Exception):
    """No embeddings row written or deleted; a claimed window goes to ``release``."""


@dataclass
class SwitchReport(CollectionReport):
    outcome: Optional[str] = None
    refused: Optional[str] = None
    stale: int = 0
    left_frozen: Optional[str] = None
    window_lost: Optional[str] = None
    holder: str = HOLDER
    tried_window: bool = False
    max_fill: int = 0

    @property
    def ok(self) -> bool:
        done = self.outcome is not None or self.skip_reason in NOTHING_TO_MIGRATE
        failed = self.error or self.refused or self.left_frozen or self.window_lost
        return done and not failed


def switch(
    model_id: str,
    *,
    collections: Optional[Sequence[str]] = None,
    rate_limit: float = 0.0,
    max_fill: int = 200,
    drain_timeout: float = 300.0,
    take_over: bool = False,
) -> List[SwitchReport]:
    """Switch each collection in turn; only a missing target table stops the run."""
    if take_over and not collections:
        raise MigrationCliError("--take-over applies only with --collection")
    target, conn, names = start_run(
        model_id, collections=collections, dry_run=False, rate_limit=rate_limit
    )
    reports: List[SwitchReport] = []
    holder = f"{HOLDER}:{uuid.uuid4().hex[:12]}"
    for name in names:
        report = SwitchReport(name=name, holder=holder, max_fill=max_fill)
        reports.append(report)
        try:
            switch_collection(report, target, conn, max_fill, drain_timeout, take_over)
        except Refused as exc:
            report.refused = str(exc)
        except TargetTableMissing as exc:
            add_error(report, str(exc))
            report.aborted = True
        except MissingCollectionError:
            report.skip_reason = SKIP_NO_METADATA
        except Exception as exc:  # noqa: BLE001 - kept on the collection, exit 1
            add_error(report, f"{type(exc).__name__}: {exc}")
        if report.tried_window:
            check_window(report)
        (logger.info if report.ok else logger.error)("%s", describe(report))
        if report.aborted:
            logger.error("Run aborted after %s", name)
            break
    logger.info("Switch done: %s", summary(reports))
    return reports


def switch_collection(
    report: SwitchReport,
    target: Target,
    conn: Any,
    max_fill: int,
    drain_timeout: float,
    take_over: bool,
) -> None:
    name = report.name
    if name != name.strip():
        report.skip_reason = SKIP_NAME_NOT_STRIPPED
        return
    checked = precompute_eligibility(name, conn=conn, target_model_id=target.model_id)
    if checked.reason == SKIP_TARGET_IS_SOURCE:
        finish_already_switched(report, take_over)
        return
    if checked.reason:
        report.skip_reason = checked.reason
        return
    report.source_model_id = checked.model_id
    state = read_migration_state(name)
    if state.state == FROZEN and not take_over:
        raise Refused(REFUSED_FROZEN)
    if state.state == FROZEN:
        window = take_over_window(report, state)
    else:
        plan(report, target, conn, max_fill)
        window = open_window(report)
    try:
        fill_and_flip(report, window, target, conn, max_fill, drain_timeout)
    except MissingCollectionError as exc:
        raise RuntimeError(str(exc)) from exc
    finally:
        release(report, window)


def finish_already_switched(report: SwitchReport, take_over_frozen: bool) -> None:
    """Bound to the target and frozen (killed after its flip): only ``--take-over``."""
    state = read_migration_state(report.name)
    if state.state == FROZEN:
        if not take_over_frozen:
            raise Refused(REFUSED_FROZEN)
        release(report, take_over_window(report, state))
    report.outcome = ALREADY_SWITCHED


def open_window(report: SwitchReport) -> MigrationWindow:
    report.tried_window = True
    try:
        return claim_migration_window(report.name, holder=report.holder)
    except CollectionMigratingError:
        raise Refused(REFUSED_FROZEN) from None


def take_over_window(report: SwitchReport, state: Any) -> MigrationWindow:
    report.tried_window = True
    try:
        return take_over_migration_window(report.name, state, holder=report.holder)
    except CollectionMigratingError:
        raise Refused(REFUSED_FROZEN) from None


def release(report: SwitchReport, window: MigrationWindow) -> None:
    try:
        release_migration_window(window)
    except CollectionNotFrozenError as exc:
        report.window_lost = str(exc)
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        add_error(report, f"unfreeze {type(exc).__name__}: {exc}")


def check_window(report: SwitchReport) -> None:
    """The one place LEFT FROZEN is decided: the row as read after every step."""
    try:
        row = read_migration_state(report.name)
        held = row.state == FROZEN and str(row.owner).startswith(f"{report.holder}:")
        report.left_frozen = "still frozen by this run" if held else None
    except Exception as exc:  # noqa: BLE001 - unknown counts as held
        report.left_frozen = f"state unreadable: {type(exc).__name__}"
    if report.left_frozen:
        report.window_lost = None


def add_error(report: SwitchReport, message: str) -> None:
    report.error = "; ".join(filter(None, (message, report.error)))


def fill_and_flip(
    report: SwitchReport,
    window: MigrationWindow,
    target: Target,
    conn: Any,
    max_fill: int,
    drain_timeout: float,
) -> None:
    name = report.name
    # A writer admitted before the freeze finishes under its lease: read after it.
    if wait_for_no_leases(name, timeout_seconds=drain_timeout):
        raise Refused("writers-still-active")
    again = precompute_eligibility(name, conn=conn, target_model_id=target.model_id)
    if again.model_id != report.source_model_id:
        raise Refused("scope-changed")
    stale, pending = plan(report, target, conn, max_fill)
    still_held = functools.partial(_held_state, window)
    delete_rows(report, stale, target.model_id, conn, still_held)
    for doc_id in sorted(pending):
        write_rows(report, pending[doc_id], target, conn, before_write=still_held)
    table = source_table(target.model_id)
    if table in table_names(conn) and holds_rows(conn, table, name):
        still_held()
        status = index_target(name, target.model_id)
        if status in FAILED_INDEX_STATUSES:
            report.index_error = f"index build on {table} ended {status}"
            raise RuntimeError(report.index_error)
    if difference := compare(report, target, conn):
        raise RuntimeError(f"target differs from source at {difference}")
    flip(report, window, target.model_id, conn)
    report.outcome = SWITCHED


def plan(
    report: SwitchReport, target: Target, conn: Any, max_fill: int
) -> Tuple[Dict[Key, int], Dict[str, List[Dict[str, Any]]]]:
    """Target rows to delete per key and source rows to embed, or why it is refused."""
    assert report.source_model_id is not None
    held = read_rows(conn, source_table(target.model_id), report.name, target)
    core = core_chunk_pairs(report.name, conn=conn) or set()
    core_docs = {doc_id for doc_id, _ in core}
    source: Rows = {}
    pending: Dict[str, List[Dict[str, Any]]] = {}
    table = source_table(report.source_model_id)
    for row in scan(conn, table, report.name, WRITE_COLUMNS):
        key, entry = chunk_key(row), (fingerprint(row), True)
        source.setdefault(key, []).append(entry)
        reason = row_skip_reason(row, core, core_docs)
        if reason:
            report.skipped[reason] = report.skipped.get(reason, 0) + 1
        elif not matches(held.get(key, []), [entry]):
            pending.setdefault(key[0], []).append(row)
    if any(len(rows) > 1 for rows in source.values()):
        raise Refused("duplicate-source-rows")
    stale = {
        k: len(rows) for k, rows in held.items() if not matches(rows, source.get(k))
    }
    for row in (row for rows in pending.values() for row in rows):
        try:
            record(row, target.model_id)
        except ValueError:
            report.skipped["unwritable"] = report.skipped.get("unwritable", 0) + 1
    if report.skipped:
        raise Refused("unusable-source-rows")
    report.stale = sum(stale.values())
    report.pending = sum(len(rows) for rows in pending.values())
    if report.stale + report.pending > max_fill:
        raise Refused("over-max-fill")
    return stale, pending


def read_both(report: SwitchReport, target: Target, conn: Any) -> Tuple[Rows, Rows]:
    assert report.source_model_id is not None
    source = read_rows(conn, source_table(report.source_model_id), report.name, None)
    held = read_rows(conn, source_table(target.model_id), report.name, target)
    return source, held


def read_rows(conn: Any, table: str, collection: str, target: Optional[Target]) -> Rows:
    """Per key, a digest per row and, for ``target``, a usable flag; no vector kept."""
    found: Rows = {}
    columns = ["doc_id", "chunk_id", *COPIED]
    if target is not None:
        columns += ["vector", "model"]
    for row in scan(conn, table, collection, columns):
        usable = target is None or (
            written_by(row, target.model_id)
            and _usable_vector(row.get("vector"), target.dimension)
        )
        found.setdefault(chunk_key(row), []).append((fingerprint(row), usable))
    return found


def scan(conn: Any, table: str, collection: str, columns: Sequence[str]) -> Any:
    if table in table_names(conn):
        yield from iter_rows(conn, table, collection_filter(collection), list(columns))


def fingerprint(row: Dict[str, Any]) -> bytes:
    """The copied fields as ``write_rows`` normalises them; unparsable metadata raw."""
    try:
        metadata: Any = source_metadata(row.get("metadata"))
    except (ValueError, TypeError):
        metadata = ["unparsed", str(row.get("metadata"))]
    fields = [row.get("text"), metadata, row.get("chunk_hash"), row.get("parse_hash")]
    fields.append(_owner(row.get("user_id")))
    encoded = json.dumps(fields, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).digest()


def matches(held: List[Entry], source: Optional[List[Entry]]) -> bool:
    if len(held) != 1 or source is None or len(source) != 1:
        return False
    digest, usable = held[0]
    return usable and digest == source[0][0]


def delete_rows(
    report: SwitchReport,
    stale: Dict[Key, int],
    model_id: str,
    conn: Any,
    check: Callable[[], Any],
) -> None:
    if not stale:
        return
    opened = conn.open_table(source_table(model_id))
    try:
        for doc_id, group in itertools.groupby(sorted(stale), key=lambda key: key[0]):
            keys = list(group)
            ids = ", ".join(f"'{escape_lancedb_string(c)}'" for _, c in keys)
            check()
            opened.delete(
                f"{collection_filter(report.name)} AND doc_id = "
                f"'{escape_lancedb_string(doc_id)}' AND chunk_id IN ({ids})"
            )
            report.removed += sum(stale[key] for key in keys)
    finally:
        _safe_close_table(opened)


def compare(report: SwitchReport, target: Target, conn: Any) -> Optional[str]:
    """The first key at which the target is not one equal row per source row."""
    source, held = read_both(report, target, conn)
    for key in sorted(set(source) | set(held)):
        if not matches(held.get(key, []), source.get(key)):
            return "/".join(key)
    return None


def flip(
    report: SwitchReport, window: MigrationWindow, model_id: str, conn: Any
) -> None:
    """Searches rewrite the collection row: a missed write retries while on source."""
    for attempt in range(1, FLIP_ATTEMPTS + 1):
        try:
            switch_collection_embedding_model(window, model_id)
            return
        except ValueError:
            bound = _normalize_model_id(
                collection_info(report.name, conn).embedding_model_id
            )
            if attempt == FLIP_ATTEMPTS or bound != report.source_model_id:
                raise
            logger.warning("%s: binding write missed; retrying", report.name)
            time.sleep(0.05 * attempt)


def describe(report: SwitchReport) -> str:
    if report.skip_reason:
        return f"{report.name}: skipped ({report.skip_reason})"
    parts = []
    if report.source_model_id:
        parts.append(f"source model {report.source_model_id}")
        if report.pending or report.stale:
            parts.append(f"{report.pending} rows to embed, {report.stale} to delete")
        parts += [f"wrote {report.written}", f"deleted {report.removed}"]
    parts += [f"unusable {k}={v}" for k, v in sorted(report.skipped.items())]
    if report.outcome:
        parts.append(report.outcome)
    if report.refused:
        parts.append(f"refused ({refusal(report)})")
    if report.error:
        parts.append(f"failed: {report.error}")
    if report.window_lost:
        parts.append(f"window lost: {report.window_lost}")
    if report.left_frozen:
        resume = f"--collection {shlex.quote(report.name)} --take-over"
        parts.append(f"LEFT FROZEN ({report.left_frozen}); resume with {resume}")
    parts.append(binding_of(report.name))
    return f"{report.name}: " + "; ".join(parts)


def refusal(report: SwitchReport) -> str:
    if report.refused == REFUSED_FROZEN:
        return FROZEN_ADVICE.format(name=shlex.quote(report.name))
    if report.refused == "over-max-fill":
        advice = "run the warm-up, or raise" if report.pending else "raise"
        return (
            f"over-max-fill: {report.pending} to embed + {report.stale} to delete > "
            f"{report.max_fill}; {advice} --max-fill"
        )
    return str(report.refused)


def binding_of(name: str) -> str:
    try:
        model = collection_info(name).embedding_model_id
    except MissingCollectionError:
        return "metadata row gone"
    except Exception as exc:  # noqa: BLE001 - the report line is still written
        return f"binding unreadable ({type(exc).__name__})"
    return f"bound to {model}" if _normalize_model_id(model) else "unbound"


def summary(reports: List[SwitchReport]) -> str:
    groups: Dict[str, List[str]] = {}
    for report in reports:
        if report.left_frozen:
            key = "left frozen"
        elif report.outcome == SWITCHED and (report.window_lost or report.error):
            key = "switched, " + ("window lost" if report.window_lost else "failed")
        elif report.error or report.window_lost:
            key = "failed"
        elif report.refused:
            key = f"refused ({report.refused})"
        else:
            key = report.outcome or f"skipped ({report.skip_reason})"
        groups.setdefault(key, []).append(report.name)
    return "; ".join(
        f"{key}: {len(names)} ({', '.join(names)})"
        for key, names in sorted(groups.items())
    )


HELP = """\
Bind knowledge-base collections to a target embedding model, one at a time.

For each LanceDB collection in scope that is bound to another model: count what
it needs, unfrozen (more than --max-fill refuses it); freeze it and wait for
writers admitted before the freeze (--drain-timeout); delete target rows with
no source row or that differ from it (text, metadata, chunk_hash, parse_hash,
owner, or a vector the target did not write or cannot use); embed the missing
rows; build the target's indexes when it holds rows of the collection; compare
row by row and, only when the target holds exactly one equal row per source
row, bind the collection to the target and unfreeze it. Ingestion into a frozen
collection is refused or deferred meanwhile. Source rows are never deleted;
nothing is rolled back. The time a collection stays frozen grows with its rows
even when nothing needs filling; --max-fill bounds only the writes.

This run does not change the binding of a collection it does not report as
switched; every report line but a skipped one ends with the binding read when
it was written. Refused reasons: frozen (by a run that holds or left the
window), over-max-fill (run the warm-up, or raise --max-fill; the warm-up
embeds the writable source rows whose key the target does not hold with a
vector it wrote and can use, and touches no other target row: not one without
a source row, not one that differs only in a copied field),
unusable-source-rows (a source row without a core chunk row, owner, text, hash,
or with metadata that is not a JSON object), duplicate-source-rows,
writers-still-active, scope-changed (binding or tables changed between the
count and the re-read under the freeze, including a collection another run
switched meanwhile). Skipped ones are out of scope (unbound, other-engine,
name-not-stripped, no-metadata, ...); one already bound to the target is
already-switched. A summary per outcome ends the run.

The target model's table must exist (ingesting into a collection bound to the
target creates it) with an integer user_id column, or the run never starts. It
is looked up again before every write; if it is gone, that collection fails
with its binding unchanged and the run stops.

A collection still frozen by this run when its line is written is reported LEFT
FROZEN; it, or one a killed run left frozen, is resumed by rerunning with
--collection NAME --take-over, which states that the run holding it is gone.
"window lost": the unfreeze did not find the row as this run last wrote it, and
it is not frozen by this run; rerun (already-switched if the binding moved).
--take-over applies only to the collections named with --collection; one it
takes over is counted under the freeze, not before.

Exit codes: 0 nothing was refused, failed, left frozen or lost its window, and
every skipped collection was unbound, no-metadata or other-engine; 1 a
collection was refused, failed, left frozen, lost its window, or skipped for
any other reason (left as it was; it needs attention), the target's table was
gone before a write, or nothing is in scope; 2 the run never started, including
when the target's table does not exist or its user_id column is of a type the
write path converts."""


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(
        description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", required=True, help="Target embedding model id from the model hub"
    )
    parser.add_argument(
        "--collection",
        action="append",
        dest="collections",
        metavar="NAME",
        help="Restrict to this collection; repeatable",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.0,
        metavar="N",
        help="Texts sent per second at most, on average, each batch counted once; "
        "0 = unlimited, with a warning. A failed provider call is not retried.",
    )
    parser.add_argument(
        "--max-fill",
        type=int,
        default=200,
        metavar="N",
        help="Most rows to embed or delete while a collection is frozen",
    )
    parser.add_argument(
        "--drain-timeout",
        type=float,
        default=300.0,
        metavar="S",
        help="Seconds to wait for admitted writers once frozen",
    )
    parser.add_argument(
        "--take-over",
        action="store_true",
        help="Take over the named collections if frozen by a run that is gone; "
        "requires --collection",
    )
    args = parser.parse_args(argv)
    numbers = (args.rate_limit, args.drain_timeout)
    if args.max_fill < 0 or not all(math.isfinite(n) and n >= 0 for n in numbers):
        logger.error("--rate-limit, --max-fill and --drain-timeout must be >= 0")
        return 2
    options = vars(args)
    try:
        reports = switch(options.pop("model"), **options)
    except Exception as exc:  # noqa: BLE001 - the run never started
        logger.error("Switch did not start: %s", exc)
        return 2
    if not reports:
        logger.warning("No collections in scope")
        return 1
    return 0 if all(report.ok for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
