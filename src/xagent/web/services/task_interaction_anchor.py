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
   corrupt.
5. the row passes all six conditions but its ``checkpoint_type`` is a
   legacy type -> absence, not corrupt.
6. the row passes all six conditions and its ``checkpoint_type`` is the
   current one -> a resolved ``InteractionAnchor``.

Step 4 must run before step 5: a row belonging to a *different* task, even
one whose ``checkpoint_type`` is legacy, must still be reported corrupt.
Folding the two steps into one (checking legacy-type first) would let a
cross-task row slip through as an ordinary absence instead -- see
``test_task_interaction_anchor.py``'s mutation test for that ordering.

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
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from ...core.agent.checkpoint import (
    LEGACY_CHECKPOINT_TYPES,
    READABLE_CHECKPOINT_TYPES,
    checkpoint_execution_id,
)
from ..models.task import Task
from ..models.task import TraceEvent as DatabaseTraceEvent
from .interaction_rollout import (
    COUNTER_ANCHOR_ABSENT_LEGACY_CHECKPOINT_TYPE,
    COUNTER_ANCHOR_ABSENT_NO_RUN,
    COUNTER_ANCHOR_UNAVAILABLE_DANGLING_POINTER,
    increment_counter,
)
from .ops_signals import INTERACTION_ANCHOR_CORRUPT, register_degradation
from .task_interaction_staging import InteractionAnchor
from .task_lease_service import TASK_RUN_ID_TRACE_FIELD

logger = logging.getLogger(__name__)


def resolve_interaction_anchor(db: Session, task: Task) -> InteractionAnchor | None:
    """Resolve ``task``'s interaction anchor from its checkpoint pointer, or
    ``None`` if none applies -- see the module docstring for the six-outcome
    table this function implements, in the order it must run.

    Six self-consistency conditions are checked once a pointed-to row is
    found (step 4); any one failing classifies the row as corrupt, copied
    verbatim from ``_load_pk_anchored_checkpoint``'s own six conditions
    (``trace_handlers.py``) rather than re-derived:

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
       function already has the one candidate row in hand), but the same
       equality semantics.
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
    run_field = row_data.get(TASK_RUN_ID_TRACE_FIELD)
    partition_matches = run_field == task.run_id
    execution_id = str(task.id)
    row_execution_id = checkpoint_execution_id(row_data)
    if (
        row.task_id != task.id
        or row.event_type != "system_update_general"
        or row.build_id is not None
        or row_data.get("checkpoint_type") not in READABLE_CHECKPOINT_TYPES
        or not partition_matches
        or (row_execution_id and row_execution_id != execution_id)
    ):
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
        resume_run_partition=task.run_id,
    )
