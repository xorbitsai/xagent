"""Resolve one task's interaction anchor from its checkpoint pointer.

This is the "other half" of anchor resolution ``InteractionAnchor``'s own
docstring (``task_interaction_staging.py``) names: a primary-key read
against ``trace_events``, performed here entirely outside any savepoint
either staging primitive opens, before a caller ever builds an anchor to
hand to ``stage_interaction_request`` or ``interaction_handoff``.

Six-outcome judgment table, evaluated in order -- the order is the contract,
not an implementation detail (see the corrupt-before-legacy note below):

1. ``task.run_id IS NULL`` -> absence.
2. ``task.last_checkpoint_trace_event_id IS NULL`` -> absence.
3. the pointer names no live ``trace_events`` row -> unavailable, reported
   to the caller as absence (see below for why).
4. the row exists but fails any of six self-consistency conditions ->
   corrupt, unless the only condition it fails is the run-partition match
   and its run field is absent entirely rather than merely mismatched --
   that shape reclassifies as absence, not corrupt. That check is not a
   second copy of the conditions: it reads the set of failed condition
   names this table's own predicate returns, and asks whether that set is
   exactly {run partition} and the field is absent
   (``is_missing_run_partition_only``, ``trace_event_staging.py``).
5. the row passes all six conditions but its ``checkpoint_type`` is a
   legacy type -> absence, not corrupt.
6. the row passes all six conditions and its ``checkpoint_type`` is the
   current one -> a resolved ``InteractionAnchor``.

Three of the six outcomes above increment a counter
(``COUNTER_ANCHOR_ABSENT_NO_RUN`` for step 1,
``COUNTER_ANCHOR_UNAVAILABLE_DANGLING_POINTER`` for step 3,
``COUNTER_ANCHOR_ABSENT_LEGACY_CHECKPOINT_TYPE`` for step 5;
``interaction_rollout.py``). Step 4's true-corrupt path has no counter; it
registers ``INTERACTION_ANCHOR_CORRUPT``, an ops degradation signal rather
than a rate metric -- see that signal's own paragraph below. Step 4's
other path, the missing-run-partition reclassification described above,
does increment a counter: the same ``COUNTER_ANCHOR_ABSENT_LEGACY_CHECKPOINT_TYPE``
step 5 uses when the row's ``checkpoint_type`` is a legacy one (both
describe the same kind of row -- a legacy-type checkpoint with no
interaction anchor to resolve -- reached by two different paths through
the row data), or the new ``COUNTER_ANCHOR_ABSENT_MISSING_RUN_PARTITION``
otherwise. Steps 2 and 6 (no checkpoint pointer, and a resolved anchor)
are the two outcomes that remain uninstrumented: nothing downstream
depends on either rate today.

Step 4 must run before step 5: a row belonging to a *different* task, even
one whose ``checkpoint_type`` is legacy, must still be reported corrupt.
Folding the two steps into one (checking legacy-type first) would let a
cross-task row slip through as an ordinary absence instead -- see
``test_task_interaction_anchor.py``'s cross-task legacy-type cell for that
ordering.

``run_id IS NULL`` handling deliberately does not match
``trace_event_staging.checkpoint_run_partition_filter``. That predicate
treats ``run_id IS NULL`` as a legitimate partition (the root-checkpoint
read path it was written for can genuinely have no run id yet) and matches
rows whose own run field is also NULL. This function instead treats
``task.run_id IS NULL`` as absence outright, at step 1, before any row is
even read: ``InteractionAnchor.resume_run_partition`` is typed ``str``, and
``_validate_anchor_fields`` (``task_interaction_staging.py``) raises
``InteractionAnchorCorrupt`` for an empty string -- protocol v1 has no way
to represent a NULL partition inside an anchor at all. Do not "align" this
with the legacy predicate by matching a NULL run field or by reclassifying
it as corrupt; the two functions are answering different questions about
NULL on purpose.

Step 3, the dangling pointer, is reported to the caller as absence, not as
a distinct outcome, and this function does not scan the legacy
``last_checkpoint_event_id`` column the way the trace-read side's fallback
does (``_AnchorFallback``, ``trace_handlers.py``) -- unavailable and absence
get different log levels (``logger.warning`` versus ``logger.info``) and
only the former is counted, but both return ``None`` and both mean the same
thing to this function's caller: there is no anchor to stage against.
Folding them into one legacy-style scan is exactly the shape this function
is built not to reproduce.

The read-direction resolver classifies legacy checkpoint types the
opposite way. ``_resolve_read_direction_anchor``
(``task_interaction_service.py``) accepts every member of
``READABLE_CHECKPOINT_TYPES`` -- the current type and the legacy ones --
where step 5 above rejects the legacy ones. Neither side is wrong and
neither is copying the other: this function decides what may be anchored
*to* when a row is written, and protocol v1 has no representation for a
legacy-anchored request; that one decides whether an already-written
row's anchor still resolves, and it keeps its condition set identical to
trace_handlers' so the two read paths cannot drift. The disagreement is
unreachable in production for exactly one reason -- this function is the
only thing that produces the anchors that resolver later reads, so no
active row anchored to a legacy-type checkpoint can exist for it to
apply to. Reconciling the two into a single classification is deliberately
NOT done here: narrowing this resolver's row-validity judgment would make
it diverge from trace_handlers' for no reachable gain. Whoever does
reconcile it must change both sides in one change, not one alone. That
disagreement is about legacy ``checkpoint_type`` only; how a row missing
the run-partition field is classified is not settled the same way
everywhere, and that is deliberate. This function and lease recovery's
own resolver (``resolve_checkpoint_recovery``, ``task_lease_service.py``)
both reclassify it off the one shared predicate -- lease recovery has no
resumable verdict for an absent checkpoint, so its deferral still ends in
FAILED, but neither calls the row corrupt. The by-primary-key *read*
path (``_load_pk_anchored_checkpoint``, ``trace_handlers.py``) does not:
it still raises ``CheckpointCorruptError`` for the same row shape,
unchanged from its behavior before this module existed, because changing
that verdict is a client-visible behavior change on the resume path and
the right verdict for it depends on the ordering defect tracked in
#2023. #2023 converges the three; this module and lease recovery do not
wait on it.

``INTERACTION_RUN_PARTITION_MISMATCH_DEGRADED`` (``ops_signals.py``) is a
signal owned by ``interaction_handoff``, not by this function: it is
registered when that context manager swallows
``InteractionRunPartitionMismatch``, an exception this function never
raises. It is a pure data-corruption alarm, not an expected degradation
channel the way the other five signals ``interaction_handoff`` registers
are -- a resolved anchor's ``resume_run_partition`` always equals the
``run_id`` this function read it against (step 4's partition-match
condition enforces that), so a caller that passes this function's own
output straight through to ``stage_interaction_request`` should see that
signal at an expected frequency of zero in normal operation. A nonzero rate
means the anchor a caller staged did not come from this function, or the
run identity changed between resolution and staging.

Zero production callers as of this module's introduction: a static test
(``tests/web/services/test_task_interaction_anchor.py``) asserts that no
production module calls ``resolve_interaction_anchor``, mirroring the
existing gate for ``task_interaction_staging.py``'s two entry points
without extending that gate itself -- this module is not one of the two
names it scans for. See that test's own docstring for the removal
condition.

Two kinds of pre-existing row are missing the run-partition field and so
fail the run-partition self-consistency check, though neither one
represents actual data corruption. The first is a row whose checkpoint
pointer was filled in by the migration that added
``last_checkpoint_trace_event_id`` (2026-08): that migration backfills the
pointer by matching a task's legacy event-id column against an existing
``trace_events`` row, and if that row predates the run-partition field
(added before this migration), it carries none. The second is any
legacy-type checkpoint row, on any task, because the function that writes
this field (``stage_trace_event_row``, ``trace_event_staging.py``) only
ever writes it for current-type checkpoints -- a legacy-type row cannot
carry the field regardless of when it was written.

Both kinds classify as absence, not corrupt: step 4's reclassification
catches them before the corrupt verdict is reached, so neither registers
``INTERACTION_ANCHOR_CORRUPT`` (``ops_signals.py``) -- a signal with no
clearing point by design, which would otherwise stay active until the
process restarts over a database state that is expected. The legacy-type
kind increments ``COUNTER_ANCHOR_ABSENT_LEGACY_CHECKPOINT_TYPE``, the same
counter step 5 uses for the row shape it describes; the current-type kind
increments ``COUNTER_ANCHOR_ABSENT_MISSING_RUN_PARTITION``.

The by-primary-key read path does not reach the same classification, and
that is deliberate. ``_load_pk_anchored_checkpoint``
(``api/trace_handlers.py``) reads the same shared predicate to decide
*which* conditions failed, but still raises ``CheckpointCorruptError`` for
this row shape, unchanged from its behavior before this module existed.
Changing the live read's verdict is a client-visible behavior change on
the resume path, and the right verdict for it depends on the ordering
defect tracked in #2023 (a run identity is minted before the checkpoint
is read, so the read is partitioned by an identity the stored row could
not have carried). This module's own reclassification does not wait on
that: it has no production caller yet, so nothing observes its verdict
today. #2023 tracks converging the two; this module does not.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from ...core.agent.checkpoint import LEGACY_CHECKPOINT_TYPES
from ..models.task import Task
from ..models.task import TraceEvent as DatabaseTraceEvent
from .interaction_rollout import (
    COUNTER_ANCHOR_ABSENT_LEGACY_CHECKPOINT_TYPE,
    COUNTER_ANCHOR_ABSENT_MISSING_RUN_PARTITION,
    COUNTER_ANCHOR_ABSENT_NO_RUN,
    COUNTER_ANCHOR_UNAVAILABLE_DANGLING_POINTER,
    increment_counter,
)
from .ops_signals import INTERACTION_ANCHOR_CORRUPT, register_degradation
from .task_interaction_staging import InteractionAnchor
from .trace_event_staging import (
    failed_checkpoint_row_conditions,
    is_missing_run_partition_only,
)

logger = logging.getLogger(__name__)


def resolve_interaction_anchor(db: Session, task: Task) -> InteractionAnchor | None:
    """Resolve ``task``'s interaction anchor from its checkpoint pointer, or
    ``None`` if none applies -- see the module docstring for the six-outcome
    table this function implements, in the order it must run.

    Six self-consistency conditions are checked once a pointed-to row is
    found (step 4); any one failing classifies the row as corrupt. They are
    evaluated by ``failed_checkpoint_row_conditions`` (``trace_event_staging.py``),
    the one definition both this resolver and ``_load_pk_anchored_checkpoint``
    (``trace_handlers.py``) read:

    1. ``row.task_id != task.id`` -- ownership.
    2. ``row.event_type != "system_update_general"`` -- not a checkpoint
       row's event type.
    3. ``row.build_id is not None`` -- must not be a build-partition row.
    4. ``row_data.get("checkpoint_type") not in READABLE_CHECKPOINT_TYPES``
       -- not a checkpoint row at all.
    5. the row's run field does not match ``task.run_id`` -- the row
       belongs to another run. Checked in Python against the fetched row's
       own data, not via ``trace_event_staging.checkpoint_run_partition_filter``
       (that predicate compiles to SQL for a query's ``WHERE`` clause; this
       function already has the one candidate row in hand), and the two no
       longer share the same equality semantics as of #2091. Both
       predicates now live in ``trace_event_staging.py`` -- one compiles to
       SQL for a query's ``WHERE`` clause, the other judges a single
       already-fetched row in Python -- but this resolver compares the
       row's run field against ``task.run_id`` directly, while the
       checkpoint read path instead compares it against the partition
       ``DatabaseTraceHandler._root_checkpoint_read_partition`` resolved
       (``trace_handlers.py``), which since #2091 can be ``None`` -- the
       untagged partition -- while ``task.run_id`` is non-null, for a task
       that has never written a run-tagged checkpoint. In that state the
       read path accepts an untagged row that this resolver classifies as
       corrupt. The two also diverge in the opposite direction: when the
       read path holds the widened partition and the row it examines does
       carry a run tag, that row still fails validation there -- but the
       read's own boundary re-probes once the verdict is produced and
       reclassifies it as a retryable "unavailable" if a run-tagged
       checkpoint now exists, because a row and its tag are written
       together and that combination therefore means its partition
       decision went stale, not that the data is inconsistent. This
       resolver has no such step. Aligning the two is #2122; this resolver
       has no production callers today, so the divergence has no live
       impact yet.
    6. ``row_execution_id and row_execution_id != execution_id`` --
       execution identity mismatch. An empty ``row_execution_id`` (a legacy
       row with no identity field) short-circuits past this condition on
       purpose -- this leniency is copied from the source condition, not a
       gap to close.

    ``InteractionAnchor`` carries no task id (see its own field list in
    ``task_interaction_staging.py``). Once this function returns an anchor,
    nothing downstream re-checks which task the anchored ``trace_events``
    row belongs to -- ``_validate_anchor_fields`` only checks the dataclass
    against itself, and the staging primitive's run-partition comparison is
    about ``run_id``, not ownership. **This function is therefore the
    single execution point of the ownership check.** Condition 1 below
    (``row.task_id != task.id``) is not a redundant sanity assert; it is
    the only one there is.

    The anchor's ``resume_execution_id`` and the condition-6 ``execution_id``
    it is checked against are both ``str(task.id)``: this function takes no
    separate execution-identity argument the way
    ``_load_pk_anchored_checkpoint`` does, because the web read path that
    function serves passes it the task id already (see that function's own
    docstring: "web's execution_id is the task id"). Using anything else
    here (e.g. the row's own, possibly-empty ``row_execution_id``) could
    persist an empty string into a column
    (``ck_task_interaction_requests_resume_execution_id_nonempty``,
    ``models/task_interaction.py``) that must never be empty.

    When any condition fails, the body asks a narrower question before
    deciding the row is corrupt: is the run-partition match the only failed
    condition, and did it fail because the row's run field is absent rather
    than wrong? ``is_missing_run_partition_only`` (``trace_event_staging.py``)
    answers it from the same failure set. A row of that shape is a
    pre-existing row whose checkpoint predates the run-partition field (see
    the module docstring's paragraph on the two kinds), not a corrupt one,
    and is reclassified as absence instead.
    """

    if task.run_id is None:
        logger.info("task %s has no run_id; no interaction anchor to resolve", task.id)
        increment_counter(COUNTER_ANCHOR_ABSENT_NO_RUN)
        return None

    pointer_id = task.last_checkpoint_trace_event_id
    if pointer_id is None:
        logger.info(
            "task %s has no checkpoint pointer; no interaction anchor to resolve",
            task.id,
        )
        return None

    row = db.get(DatabaseTraceEvent, pointer_id)
    if row is None:
        logger.warning(
            "task %s's checkpoint pointer %s has no matching trace_events row",
            task.id,
            pointer_id,
        )
        increment_counter(COUNTER_ANCHOR_UNAVAILABLE_DANGLING_POINTER)
        return None

    row_data: dict[str, Any] = row.data if isinstance(row.data, dict) else {}
    execution_id = str(task.id)
    # task.run_id was already checked non-None above (step 1); str() here
    # is the same explicit-cast convention this function uses elsewhere for
    # a Column-typed attribute passed where a plain type is expected (see
    # execution_id above, and trace_event_id=int(row.id) below).
    failed = failed_checkpoint_row_conditions(
        row,
        row_data,
        task_id=int(task.id),
        run_id=str(task.run_id),
        execution_id=execution_id,
    )
    if failed:
        if is_missing_run_partition_only(failed, row_data):
            row_checkpoint_type = row_data.get("checkpoint_type")
            if row_checkpoint_type in LEGACY_CHECKPOINT_TYPES:
                logger.info(
                    "task %s's checkpoint pointer %s is missing its "
                    "run-partition field and names a legacy checkpoint "
                    "type %r; no interaction anchor to resolve",
                    task.id,
                    pointer_id,
                    row_checkpoint_type,
                )
                increment_counter(COUNTER_ANCHOR_ABSENT_LEGACY_CHECKPOINT_TYPE)
            else:
                logger.info(
                    "task %s's checkpoint pointer %s is missing its "
                    "run-partition field; treating it as a pre-existing "
                    "row with no interaction anchor to resolve, not as "
                    "corrupt",
                    task.id,
                    pointer_id,
                )
                increment_counter(COUNTER_ANCHOR_ABSENT_MISSING_RUN_PARTITION)
            return None
        register_degradation(
            INTERACTION_ANCHOR_CORRUPT,
            f"task {task.id}: checkpoint pointer {pointer_id} does not "
            "match the row it anchors",
        )
        logger.error(
            "task %s's checkpoint pointer %s failed anchor validation",
            task.id,
            pointer_id,
        )
        return None

    # Every row that survives the six conditions above has a
    # checkpoint_type in READABLE_CHECKPOINT_TYPES (condition 4), which is
    # exactly {CHECKPOINT_TYPE} | LEGACY_CHECKPOINT_TYPES -- so the branch
    # below is exhaustive between steps 5 and 6, with nothing left over.
    # No legacy-type row reaches this branch under the current write path.
    # Getting here means all six conditions passed, which for a non-null task
    # run id means the row carries a matching run-partition field -- and
    # ``stage_trace_event_row`` (trace_event_staging.py) writes that field
    # only for a current-type checkpoint, never for a legacy-type one. Every
    # legacy-type row therefore leaves through the
    # is_missing_run_partition_only branch above. The branch below is kept
    # for exhaustiveness over READABLE_CHECKPOINT_TYPES, not because it
    # covers a shape the writer produces: deleting it would make this
    # function's exhaustiveness depend on the writer's current behavior
    # rather than on the condition set it actually reads.
    checkpoint_type = row_data.get("checkpoint_type")
    if checkpoint_type in LEGACY_CHECKPOINT_TYPES:
        logger.info(
            "task %s's checkpoint pointer %s names a legacy checkpoint "
            "type %r; no interaction anchor to resolve",
            task.id,
            pointer_id,
            checkpoint_type,
        )
        increment_counter(COUNTER_ANCHOR_ABSENT_LEGACY_CHECKPOINT_TYPE)
        return None

    logger.debug(
        "task %s's checkpoint pointer %s resolved to an interaction anchor",
        task.id,
        pointer_id,
    )
    return InteractionAnchor(
        trace_event_id=int(row.id),
        resume_event_id=str(row.event_id),
        resume_execution_id=execution_id,
        resume_run_partition=str(task.run_id),
    )
