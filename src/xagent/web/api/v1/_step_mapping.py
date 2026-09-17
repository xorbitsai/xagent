"""Pure mapping from internal trace events to public SDK step types.

Background:

    Internally the agent runtime emits a rich tree of ~32 ``event_type``
    strings (see ``ws_trace_handlers.get_event_type_mapping``) that
    capture every phase, sub-phase, tool call, LLM call, memory op,
    visualization tick, etc. That granularity is useful for the web
    UI's live trace view and for internal debugging, but it's far too
    much surface area to commit to in a stable public SDK contract.

    This module collapses those 32 internal types into the **4 public
    step types** the SDK promises:

      - ``thinking``         (reasoning phases: planning / step / action)
      - ``tool_call``        (tool invocations or skill selections)
      - ``agent_delegation`` (a sub-agent invocation -- surfaced
                              separately because the call shape is
                              meaningfully different from a flat tool)
      - ``message``          (one user or assistant message)

    Everything else (llm_call_*, memory_*, react_task_*, react_step_*,
    visualization_update, task_completion, trace_error,
    action_*_compact) is **intentionally not exposed** so the SDK
    surface can evolve without breaking clients. ``dag_execute_*`` is
    a partial exception -- see the "Pairing rule" section below.

Pure-function design:

    ``map_trace_events_to_public_steps`` takes a list of DB
    ``TraceEvent`` rows and returns a list of :class:`PublicStep`
    dicts. It has no DB / FastAPI / async dependencies so it's
    independently unit-testable against synthetic events and the only
    place SDK clients can observe a behavior change is through this
    one function's output -- which makes regressions easy to gate.

Incremental projection:

    The state machine that implements the pairing rules below lives in
    :class:`PublicStepProjector`, not in this function. The projector
    holds all the folding state a fold needs -- the pending
    (start-seen, end-not-yet-seen) table, the ``dag_plan_*`` replan
    counter, and the independent ``dag_execution`` planning-phase
    counter -- so a caller can feed it one event at a time and read
    back the steps that changed. ``map_trace_events_to_public_steps``
    stays pure by constructing one fresh, throwaway projector per
    call and reading back its materialized result; it keeps no
    folding state of its own.

Pairing rule:

    Start / end events are paired by a stable ``key``:

      - ``tool_execution_*`` events pair on ``data['tool_call_id']``
        (the provider-assigned call id), falling back to ``step_id``.
        ``tool_execution_id`` is accepted ahead of it for compatibility
        but has no current producer. A per-invocation id is the only
        safe key when the same tool is called twice in the same step.
      - ``react_action_*`` events pair on ``step_id``. ``react_action_end``
        has no current producer -- the (ACTION, END, REACT) combination
        is unreachable in this codebase today -- so the rule below is
        untested for this family, not an observed fact.
      - ``dag_step_*`` events pair on ``step_id``. Its end events
        project their own ``data['status']``: ``"failed"`` stays
        failed, anything else (including a missing key) becomes
        ``"completed"``. ``react_action_end`` would be folded through
        this same rule if one were ever emitted, but the rest of the
        ACTION family (``tool_execution_end``, below) instead reads
        ``data['success']`` -- so a future ``react_action_end``
        producer that follows that existing sibling convention rather
        than ``dag_step_*``'s would have its failures silently
        projected as ``"completed"``.
      - ``dag_plan_*`` events pair on ``task_id`` (single planning
        phase per task; no per-plan identifier available).
      - ``dag_execution`` events pair on ``data['phase']``: a
        ``planning``/``replanning`` value opens a planning-phase
        thinking step, an ``executing`` value closes the currently
        open one. This is a second, independent source of the same
        public phase as ``dag_plan_*`` above -- it keeps its own
        counter and open key so the two families never collide, even
        when both appear in the same event stream.
      - ``skill_select_*`` events pair on ``task_id`` (single skill
        selection phase per task).
      - ``dag_execute_*`` events carry the whole DAGPattern's lifecycle,
        not a single step's, so they never open their own pending
        entry. ``dag_execute_end`` instead reaches into whichever
        ``dag_execution`` planning key is currently open and clears
        that key -- the round it belongs to has now ended, one way or
        another -- closing the step as ``"failed"`` only when
        ``data['status']`` is the literal string ``"failed"``. This
        branch only has a key left to close while plan generation
        itself is still open: the ``dag_execution`` ``executing``-phase
        event already clears the key the moment plan generation
        finishes, so an ``"interrupted"`` or ``"waiting_for_user"``
        round reported from that point on (during step execution)
        can't reach back here. Only a round that is interrupted (or
        reports ``"waiting_for_user"``, ``"completed"``, a missing
        key, or ``None``) *during plan generation* leaves the planning
        step running, the opposite fallback direction from
        ``dag_step_*`` above (which cannot be reused here for exactly
        that reason) -- but the key is cleared regardless, so a
        *later* round's ``dag_execute_end`` can never reach back
        through it and misattribute its own failure to this round's
        stranded step. A plan-generation exception that escapes
        ``DAGPattern.run()`` before it ever calls ``on_pattern_end``
        emits no ``dag_execute_end`` at all, so the planning step is
        left running indefinitely too -- out of scope for this
        module. ``dag_execute_start`` additionally clears a stale open
        planning key left by a prior round whose terminal
        ``dag_execute_end`` was itself dropped (never observed at
        all) -- the one case ``dag_execute_end``'s own clearing above
        cannot already have handled, since there was no such event to
        clear it.

    Orphan ends (end with no matching start) are dropped -- they
    represent malformed data and the SDK contract is "every step has
    a start"; emitting an orphan would make ``started_at`` synthetic
    and confusing.

    Orphan starts (start with no matching end) are emitted with
    ``status='running'`` and ``completed_at=None``. This naturally
    handles the case where the SDK polls ``/steps`` mid-task.

    Key collision -- two starts sharing one pairing key before either
    ends -- is last-write-wins: the second start replaces the first
    pending entry, so only the second reaches a terminal state. Both
    carry the same public step id (id and pairing key are derived from
    the same (type, key) pair), so a consumer folding ``feed`` results
    by id converges on the second step rather than being left with a
    stranded ``running`` one. ``dag_plan_*`` and the planning-phase
    steps ``dag_execution`` produces are the exception: neither has a
    natural key at all, so each family keeps its own counter (and its
    own ``plan:`` / ``planning:`` key prefix) to generate a distinct
    one per occurrence.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from xagent.core.tools.adapters.vibe.agent_tool_names import (
    LEGACY_AGENT_TOOL_NAME_PREFIX,
    is_agent_tool_name,
)
from xagent.core.tools.adapters.vibe.connector_runtime import (
    redact_runtime_sensitive_payload,
)

logger = logging.getLogger(__name__)


# Agent tool names are routed to the public ``agent_delegation`` type
# instead of ``tool_call`` so SDK consumers can render nested
# timelines without pattern-matching tool names client-side.


class PublicStepProjector:
    """Incremental folding state machine: trace events -> public steps.

    Holds all the folding state a fold needs -- the pending
    (start-seen, end-not-yet-seen) pairing table, the ``dag_plan_*``
    counter/open-key used to disambiguate replan, and the
    ``dag_execution`` counter/open-key (``_dag_execution_counter`` /
    ``_open_dag_execution_key``) that tracks the same planning phase
    from its second, independent event source. The batch driver
    (``map_trace_events_to_public_steps``) keeps none of its own: it
    builds one instance and reads back the result. Holding this state
    on an instance is what lets a caller feed a live event stream one
    event at a time (see the module docstring's pairing rules for what
    "pairing" means per event family).

    Two ways to build one:

      - ``PublicStepProjector()`` then repeated ``feed(event)`` calls, for
        a live/incremental consumer.
      - :meth:`from_history` to replay a full event list in one call and
        get a projector whose state is exactly where it would be had it
        been fed those events live -- this is what
        ``map_trace_events_to_public_steps`` uses, and it's also how a
        late-attaching consumer pre-warms pairing state instead of seeing
        orphan ends for steps that started before it attached.

    Two retention modes, chosen at construction:

      - ``retain_finished=True`` (the default): every finalized step is
        kept so :meth:`materialized_steps` can return the task's whole
        projected timeline at the end. A one-shot request needs this,
        and ``map_trace_events_to_public_steps`` -- and therefore
        ``GET /v1/chat/tasks/{task_id}/steps`` -- reads its result
        *only* through :meth:`materialized_steps`, so that driver must
        never be built with the other mode.
      - ``retain_finished=False``: a finalized step is returned by
        :meth:`feed` and then forgotten. For a consumer that acts on
        each ``feed`` result immediately and never calls
        :meth:`materialized_steps` -- the v1 SSE sink serializes each
        changed step to a frame as it comes and holds one projector for
        as long as its connection lives, so retaining would accumulate
        every step's untruncated ``data`` for that whole time in a list
        nothing reads. :meth:`materialized_steps` raises in this mode
        rather than returning a silently partial timeline.

    ``feed``'s return value is identical in both modes; the mode only
    decides whether the projector also keeps the step afterwards. The
    pending table is kept either way -- the pairing rules need it.

    :meth:`materialized_steps` never sorts by ``started_at``. That global
    resort is a batch-only concern (see ``map_trace_events_to_public_steps``):
    a live consumer wants steps in the order they actually resolved, not
    resorted on every event.

    Preconditions for the incremental path: one instance per task, fed
    in event order, from one thread/task at a time. ``feed`` mutates
    the pairing tables without any locking, and the ``dag_plan_*`` open
    key is a single slot, so two interleaved tasks would cross-pair.
    The batch driver satisfies all three trivially -- it builds a fresh
    throwaway instance per call.
    """

    def __init__(self, *, retain_finished: bool = True) -> None:
        # In-progress (start seen, end not yet seen) steps keyed by
        # (public_type, pairing_key). Order of insertion is preserved by
        # Python 3.7+ dict semantics, which is what we use to emit final
        # output in the order steps were started.
        self._pending: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Completed-or-emitted-immediately steps. Filled in either by an
        # end event matching a pending start, or by a one-shot event like
        # ``user_message`` / ``ai_message``. ``None`` when the caller
        # opted out of retention (see the class docstring): those steps
        # are still returned by ``feed``, they are just not kept after.
        self._finished: Optional[List[Dict[str, Any]]] = [] if retain_finished else None
        # ``dag_plan_*`` has no per-plan identifier in the event data, so
        # we synthesize one by counting starts and remembering the
        # currently-open key. Replan in a single task (rare but legal)
        # produces N >= 2 pairs; without this counter the second
        # dag_plan_start would silently overwrite the first's pending
        # entry. We assume plans don't nest (only one in flight at a
        # time); nesting would require a stack, which DAG doesn't emit.
        self._plan_counter = 0
        self._open_plan_key: Optional[str] = None
        # ``dag_execution`` (phase planning/replanning/executing) is a
        # second source of the same public planning phase, translated
        # further down in ``feed``. It gets its own counter and open
        # key so it never shares pairing state with ``dag_plan_*``
        # above -- both families can appear in the same stream without
        # cross-talk or id collisions.
        self._dag_execution_counter = 0
        self._open_dag_execution_key: Optional[str] = None

    @classmethod
    def from_history(cls, events: List[Any]) -> "PublicStepProjector":
        """Build a projector pre-warmed by replaying a full event history.

        Feeds every event into a fresh instance, in order. The resulting
        state (the accumulated timeline plus any still-unpaired
        starts) is identical to what a live consumer would have
        accumulated by that point -- there is no separate "batch" code
        path for the folding logic itself, only this replay.
        """
        projector = cls()
        for event in events:
            projector.feed(event)
        return projector

    def materialized_steps(self) -> List[Dict[str, Any]]:
        """Return every step currently known, finished or still running.

        Order: finished steps in the order they resolved, then any
        still-``pending`` (``running``) steps in the order they started.
        Callers that need the public started_at-sorted order (currently
        only the ``map_trace_events_to_public_steps`` batch driver) sort
        this themselves.

        The list itself is a fresh snapshot, but the step dicts inside
        it are the projector's live objects: a still-``running`` step
        is mutated in place when its end event arrives (see
        :meth:`feed`). Treat them as read-only views -- a caller that
        needs a stable picture must copy before storing.

        Raises ``RuntimeError`` on a projector built with
        ``retain_finished=False``: the finished steps it would need were
        never kept, so there is no partial answer worth returning.
        """
        if self._finished is None:
            raise RuntimeError(
                "materialized_steps() needs the finished-step history; this "
                "projector was built with retain_finished=False"
            )
        steps = list(self._finished)
        steps.extend(self._pending.values())
        return steps

    def feed(self, event: Any) -> List[Dict[str, Any]]:
        """Fold one trace event, returning the step(s) it changed.

        A start event returns the new ``running`` step; an end event
        (or failure event) returns the step now in its terminal state;
        a one-shot event (``user_message`` / ``ai_message``) returns the
        step it produces. Events that don't affect any public step --
        unexposed types, orphan ends, non-terminal sub-events of an
        exposed family -- return ``[]``.

        The returned dict is the same object subsequently held in
        :meth:`materialized_steps`'s backing storage (mutated in place
        when a pending start is later finalized), so a caller folding
        successive ``feed`` results by step id always ends up with the
        current state of each step (when this projector retains them;
        see the class docstring).
        """
        event_type = _safe_get(event, "event_type")
        if not event_type:
            return []

        # ===== messages: one event per message, no pairing =====
        if event_type == "user_message":
            step = _build_message_step(event, role="user")
            if self._finished is not None:
                self._finished.append(step)
            return [step]
        if event_type == "ai_message":
            step = _build_message_step(event, role="assistant")
            if self._finished is not None:
                self._finished.append(step)
            return [step]

        # ===== thinking: paired start/end =====
        thinking_phase = _thinking_phase_for(event_type)
        if thinking_phase == "planning":
            # Special-cased because plan events have no per-plan id;
            # we generate one from a counter and remember the open
            # key so the next dag_plan_end pairs with the latest start.
            # The dag_execution branch below keeps its own independent
            # counter and "planning:" key prefix; keep pairing
            # semantics changes in lockstep between the two branches.
            if event_type.endswith("_start"):
                self._plan_counter += 1
                task_ref = (
                    _safe_get(event, "task_id")
                    or _safe_get(event, "event_id")
                    or "anon"
                )
                self._open_plan_key = f"plan:{task_ref}:{self._plan_counter}"
                step = _build_thinking_start(
                    event, phase="planning", key=self._open_plan_key
                )
                self._pending[("thinking", self._open_plan_key)] = step
                return [step]
            if event_type.endswith("_end") and self._open_plan_key is not None:
                finalized = _finalize_pending(
                    self._pending,
                    self._finished,
                    ("thinking", self._open_plan_key),
                    end_event=event,
                    status="completed",
                )
                self._open_plan_key = None
                return [finalized] if finalized is not None else []
            # Orphan end with no open plan: drop silently (same policy
            # as orphan tool_execution_end).
            return []

        if thinking_phase is not None:
            # action / step branch -- step_id is the natural pair key.
            key = _thinking_pair_key(event, thinking_phase)
            if event_type.endswith("_start"):
                step = _build_thinking_start(event, phase=thinking_phase, key=key)
                self._pending[("thinking", key)] = step
                return [step]
            if event_type.endswith("_end"):
                # dag_step_end always carries its own data['status']
                # ("failed" or "completed"); react_action_end has no
                # current producer but is covered by the same rule for
                # when one appears. A missing key defaults to
                # "completed" -- see _terminal_status_from_event.
                # No extra_data_fn here on purpose: a failed thinking
                # step doesn't carry an "error" field. Its data shape
                # is the existing contract ({"phase": ...} only), unlike
                # the tool family below, which does attach "error".
                finalized = _finalize_pending(
                    self._pending,
                    self._finished,
                    ("thinking", key),
                    end_event=event,
                    status=_terminal_status_from_event(event),
                )
                return [finalized] if finalized is not None else []
            # Events in these families that are neither a start nor an
            # end carry no step transition.
            return []

        # ===== tool_call / agent_delegation: paired start/end + failure =====
        if event_type in (
            "tool_execution_start",
            "tool_execution_end",
            "tool_execution_failed",
        ):
            tool_name = _data_get(event, "tool_name")
            is_delegation = is_agent_tool_name(tool_name)
            public_type = "agent_delegation" if is_delegation else "tool_call"
            # Pair on a per-invocation id (unique even when one step
            # invokes the same tool twice). The effective key today is
            # ``tool_call_id``; ``tool_execution_id`` is accepted ahead
            # of it for compatibility but has no current producer.
            # step_id alone is unsafe because one step may invoke
            # multiple tools.
            key = (
                _data_get(event, "tool_execution_id")
                or _data_get(event, "tool_call_id")
                or _safe_get(event, "step_id")
                or _safe_get(event, "event_id")
            )
            if not key:
                return []

            if event_type == "tool_execution_start":
                step = _build_tool_start(
                    event,
                    public_type=public_type,
                    tool_name=tool_name,
                    key=str(key),
                )
                self._pending[(public_type, str(key))] = step
                return [step]
            if event_type == "tool_execution_end":
                success = _data_get(event, "success", default=True)
                status = "completed" if success else "failed"
                # ``tool_call`` and ``agent_delegation`` use different keys
                # on the public schema: tool_call exposes ``result``
                # (generic tool return), agent_delegation exposes
                # ``output`` (mirroring ``input`` on the start side). The
                # underlying internal field is still ``data['result']`` --
                # we only rename on the public surface. ``error`` is the
                # same on both for failures.
                success_key = (
                    "output" if public_type == "agent_delegation" else "result"
                )
                finalized = _finalize_pending(
                    self._pending,
                    self._finished,
                    (public_type, str(key)),
                    end_event=event,
                    status=status,
                    extra_data_fn=lambda ev, succ=success, k=success_key: (
                        {k: _data_get(ev, "result")}
                        if succ
                        else {
                            "error": _data_get(ev, "error") or "Tool execution failed"
                        }
                    ),
                )
                return [finalized] if finalized is not None else []
            # tool_execution_failed: v2 runtime emits a dedicated failure
            # event (TraceCategory.TOOL + TraceAction.ERROR) instead of
            # tool_execution_end with success=False. Without this branch
            # the pending start was never finalized and the public step
            # stayed at status='running' indefinitely.
            finalized = _finalize_pending(
                self._pending,
                self._finished,
                (public_type, str(key)),
                end_event=event,
                status="failed",
                extra_data_fn=lambda ev: {
                    "error": _data_get(ev, "error")
                    or _data_get(ev, "error_message")
                    or "Tool execution failed"
                },
            )
            return [finalized] if finalized is not None else []

        # ===== skill_select_*: surface as tool_call with skill name =====
        if event_type in ("skill_select_start", "skill_select_end"):
            key = (
                _data_get(event, "skill_name")
                or _safe_get(event, "step_id")
                or str(_safe_get(event, "task_id") or "skill")
            )
            if event_type == "skill_select_start":
                step = _build_tool_start(
                    event,
                    public_type="tool_call",
                    tool_name=_data_get(event, "skill_name") or "skill_select",
                    key=str(key),
                )
                self._pending[("tool_call", str(key))] = step
                return [step]
            finalized = _finalize_pending(
                self._pending,
                self._finished,
                ("tool_call", str(key)),
                end_event=event,
                status="completed",
                extra_data_fn=lambda ev: {"result": _data_get(ev, "result")},
            )
            return [finalized] if finalized is not None else []

        # ===== dag_execution: translate an existing signal into the
        # same public planning phase ``dag_plan_*`` produces =====
        if event_type == "dag_execution":
            phase = _data_get(event, "phase")
            if not isinstance(phase, str):
                # Missing/malformed phase: nothing to translate.
                return []
            if phase in ("planning", "replanning"):
                self._dag_execution_counter += 1
                task_ref = (
                    _safe_get(event, "task_id")
                    or _safe_get(event, "event_id")
                    or "anon"
                )
                key = f"planning:{task_ref}:{self._dag_execution_counter}"
                self._open_dag_execution_key = key
                step = _build_thinking_start(event, phase="planning", key=key)
                self._pending[("thinking", key)] = step
                return [step]
            if phase == "executing":
                if self._open_dag_execution_key is None:
                    # No open planning step to close: same policy as an
                    # orphan end elsewhere in this module -- drop.
                    return []
                finalized = _finalize_pending(
                    self._pending,
                    self._finished,
                    ("thinking", self._open_dag_execution_key),
                    end_event=event,
                    status="completed",
                )
                self._open_dag_execution_key = None
                return [finalized] if finalized is not None else []
            # Other phases (e.g. completion_assessment): not exposed.
            return []

        # ===== dag_execute_start / dag_execute_end: clear the
        # dag_execution planning key at each round boundary, closing
        # the planning step only on outright plan failure. See the
        # module docstring's "Pairing rule" section for the full
        # rationale. =====
        if event_type == "dag_execute_start":
            # Drop a stale open key left by a prior round whose
            # terminal dag_execute_end never reached this projector.
            # The only live path to that today is a plan-generation
            # exception other than RequiredToolCallError escaping
            # DAGPattern.run(): it re-raises past the try/except
            # instead of going through on_pattern_end, so no
            # dag_execute_end is ever emitted for that round. A round
            # whose dag_execute_end did arrive -- whatever its status
            # -- already cleared its own key below; this guard only
            # catches that never-emitted-terminal-event case. The
            # stale pending step itself is left as-is (still
            # "running").
            self._open_dag_execution_key = None
            return []
        if event_type == "dag_execute_end":
            # A round's key is scoped to that round: clear it the
            # moment this round's terminal event arrives, regardless of
            # status. Otherwise a non-"failed" end (typically
            # "interrupted") leaves the key open, and a *later* round's
            # dag_execute_end -- observed without ever seeing that
            # round's own dag_execute_start/planning events -- would
            # reach back through the stale key and misattribute its
            # failure to this round's still-pending planning step.
            dag_execution_key = self._open_dag_execution_key
            self._open_dag_execution_key = None
            if dag_execution_key is None:
                return []  # No open planning step to close.
            # Deliberately not _terminal_status_from_event: that
            # helper's fallback direction (anything but "failed"
            # becomes "completed") is the opposite of what this branch
            # needs -- only the literal "failed" may close the step.
            if _data_get(event, "status") != "failed":
                return []
            finalized = _finalize_pending(
                self._pending,
                self._finished,
                ("thinking", dag_execution_key),
                end_event=event,
                status="failed",
                # No extra_data_fn: data['result'] is the DAG
                # pattern's full output and must not reach the public
                # surface. Keeps data == {"phase": "planning"}.
            )
            return [finalized] if finalized is not None else []

        # Everything else (llm_call_*, memory_*, react_task_*,
        # react_step_*, visualization_update, task_completion,
        # trace_error, action_*_compact) -- not exposed in the SDK
        # contract. Silently drop.
        return []


def map_trace_events_to_public_steps(
    events: List[Any],
) -> List[Dict[str, Any]]:
    """Collapse internal trace events into the 4 public SDK step types.

    Args:
        events: List of ``TraceEvent`` ORM rows (or any objects with
            ``event_type``, ``data``, ``step_id``, ``timestamp``, and
            ``event_id`` attributes). Caller is responsible for
            filtering by ``task_id`` and ordering by ``id`` ASC; this
            function preserves the input order in its output.

    Returns:
        List of public-step dicts in the order their **start** events
        first appeared. Each dict has the shape:

            {
                "id": str,
                "type": "thinking" | "tool_call" | "agent_delegation" | "message",
                "status": "running" | "completed" | "failed",
                "started_at": datetime,
                "completed_at": Optional[datetime],
                "data": dict (type-specific fields, see below)
            }

        Type-specific ``data`` fields:

          - ``thinking``: ``{"phase": "planning" | "step" | "action"}``
          - ``tool_call``: ``{"name", "args", "result"?, "error"?}``
            (``result`` populated on success, ``error`` on failure)
          - ``agent_delegation``: ``{"sub_agent_name", "input"?, "output"?}``
            (``sub_agent_name`` is derived from the agent tool name;
            ``input`` from ``tool_args`` if available, ``output`` from
            end event's ``result``)
          - ``message``: ``{"role": "user"|"assistant", "content": str}``

    Notes:
        - This function is pure (no I/O, no global state). All test
          cases live in ``tests/web/api/v1/test_steps_mapping.py``.
        - Unknown event types (anything not listed in the mapping
          tables in this module) are silently dropped. Adding a new
          public type is a deliberate per-type opt-in.
    """
    # The folding state machine itself lives in ``PublicStepProjector``;
    # this is a thin batch driver that replays the full event list
    # through a fresh instance and applies the one sort step that is a
    # batch-only concern (see ``PublicStepProjector.materialized_steps``).
    output = PublicStepProjector.from_history(events).materialized_steps()
    # Final sort by ``started_at`` so output is monotonic regardless of
    # whether a step finishes before the next one starts. Stable sort
    # preserves insertion order for ties.
    output.sort(key=lambda s: s["started_at"])
    return output


# ===== thinking helpers =====


_THINKING_PHASE_BY_PREFIX: Tuple[Tuple[str, str], ...] = (
    # Order matters: longer / more-specific prefixes first so
    # ``react_action_*`` doesn't accidentally match a future
    # ``react_*`` general rule.
    ("react_action_", "action"),
    ("dag_step_", "step"),
    ("dag_plan_", "planning"),
)


def _thinking_phase_for(event_type: str) -> Optional[str]:
    """Return the public ``thinking.phase`` value for an internal event,
    or ``None`` if this event is not a thinking event.
    """
    for prefix, phase in _THINKING_PHASE_BY_PREFIX:
        if event_type.startswith(prefix):
            return phase
    return None


def _thinking_pair_key(event: Any, phase: str) -> str:
    """Pairing key for a non-planning thinking start/end event.

    ``react_action_*`` and ``dag_step_*`` always carry a step_id
    which is the natural pairing key. Planning events are handled
    inline in :meth:`PublicStepProjector.feed` because they lack a
    per-plan identifier and need a synthesized counter.
    """
    return str(_safe_get(event, "step_id") or _safe_get(event, "event_id") or "")


def _build_thinking_start(event: Any, *, phase: str, key: str) -> Dict[str, Any]:
    return {
        "id": f"thinking:{key}",
        "type": "thinking",
        "status": "running",
        "started_at": _ts(event),
        "completed_at": None,
        "data": {"phase": phase},
    }


# ===== tool_call / agent_delegation helpers =====


def _build_tool_start(
    event: Any,
    *,
    public_type: str,
    tool_name: Optional[str],
    key: str,
) -> Dict[str, Any]:
    """Build the start side of a tool_call or agent_delegation step.

    For legacy ``call_agent_<name>`` delegation tool names we extract
    the suffix for backward-compatible display. Canonical ``agent_<id>``
    and semantic Workforce tool names are already stable public identifiers.

    The args/input value lives under different keys depending on which
    runtime emitted the event: v1 uses ``tool_args``, v2 uses
    ``tool_params``. We read whichever is present so the public step
    surface stays uniform across runtimes.
    """
    args = _data_get(event, "tool_args")
    if args is None:
        args = _data_get(event, "tool_params")
    args = redact_runtime_sensitive_payload(args)
    assistant_content = _data_get(event, "assistant_content")
    assistant_content = (
        assistant_content.strip()
        if isinstance(assistant_content, str) and assistant_content.strip()
        else None
    )
    if public_type == "agent_delegation" and isinstance(tool_name, str):
        sub_agent_name = (
            tool_name.removeprefix(LEGACY_AGENT_TOOL_NAME_PREFIX)
            if tool_name.startswith(LEGACY_AGENT_TOOL_NAME_PREFIX)
            else tool_name
        )
        data = {
            "sub_agent_name": sub_agent_name,
            "input": args,
        }
        if assistant_content:
            data["assistant_content"] = assistant_content
        return {
            "id": f"agent_delegation:{key}",
            "type": "agent_delegation",
            "status": "running",
            "started_at": _ts(event),
            "completed_at": None,
            "data": data,
        }
    data = {
        "name": tool_name,
        "args": args,
    }
    if assistant_content:
        data["assistant_content"] = assistant_content
    return {
        "id": f"tool_call:{key}",
        "type": "tool_call",
        "status": "running",
        "started_at": _ts(event),
        "completed_at": None,
        "data": data,
    }


# ===== message helpers =====


def _build_message_step(event: Any, *, role: str) -> Dict[str, Any]:
    """One-shot message step (no pairing).

    user_message stores its text in ``data['message']`` (see
    ``trace_user_message``); ai_message uses ``data['content']`` (see
    ``trace_ai_message``). We normalize both into ``content`` here so
    SDK consumers don't have to know about the asymmetry.
    """
    content = _data_get(event, "content")
    if content is None:
        content = _data_get(event, "message")
    ts = _ts(event)
    return {
        "id": f"message:{_safe_get(event, 'event_id') or _safe_get(event, 'id')}",
        "type": "message",
        "status": "completed",
        "started_at": ts,
        "completed_at": ts,
        "data": {
            "role": role,
            "content": content or "",
        },
    }


# ===== shared finalization =====


def _terminal_status_from_event(event: Any) -> str:
    """Project a thinking-family end event's own ``data['status']``.

    ``dag_step_end`` (and, theoretically, ``react_action_end``) carries
    the step's own terminal status: ``"failed"`` or ``"completed"``. We
    surface it as-is, mapping anything other than the literal string
    ``"failed"`` -- including a missing key -- to ``"completed"``. Every
    current ``dag_step_end`` producer sets the field explicitly; the
    completed fallback only guards a malformed or legacy event, not a
    real failure being swallowed.
    """
    return "failed" if _data_get(event, "status") == "failed" else "completed"


def _finalize_pending(
    pending: Dict[Tuple[str, str], Dict[str, Any]],
    finished: Optional[List[Dict[str, Any]]],
    key: Tuple[str, str],
    *,
    end_event: Any,
    status: str,
    extra_data_fn: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Move ``pending[key]`` out of the pending table and patch it with
    end metadata, appending it to ``finished`` when the caller keeps one
    (``None`` for a projector built with ``retain_finished=False`` -- see
    :class:`PublicStepProjector`).

    Orphan end (no matching start in ``pending``) is dropped on
    purpose -- see module docstring. Returns the finalized step dict
    (the same object appended to ``finished`` when one is kept), or
    ``None`` for a dropped orphan end, so callers that need to report
    "what changed" (:meth:`PublicStepProjector.feed`) don't have to
    re-derive it.
    """
    step = pending.pop(key, None)
    if step is None:
        # Orphan end event; skip.
        return None
    step["status"] = status
    step["completed_at"] = _ts(end_event)
    if extra_data_fn is not None:
        try:
            extra = extra_data_fn(end_event) or {}
            step["data"].update(redact_runtime_sensitive_payload(extra))
        except Exception as exc:  # defensive; data shape is external
            logger.debug("step extra_data_fn failed: %s", exc)
    if finished is not None:
        finished.append(step)
    return step


# ===== attribute / data accessors =====


def _safe_get(event: Any, name: str, default: Any = None) -> Any:
    """Read an attribute that may exist on the ORM row OR in event.data.

    Handles both real ``TraceEvent`` rows and the lightweight dict-like
    stubs used in unit tests, without forcing either to mimic the other.
    """
    if hasattr(event, name):
        return getattr(event, name)
    if isinstance(event, dict):
        return event.get(name, default)
    return default


def _data_get(event: Any, name: str, default: Any = None) -> Any:
    """Read a field from ``event.data`` regardless of whether ``data``
    is a JSON column dict or already-deserialized dict on a stub.
    """
    data = _safe_get(event, "data")
    if isinstance(data, dict):
        return data.get(name, default)
    return default


def _ts(event: Any) -> datetime:
    """Coerce the event's timestamp into a tz-aware datetime."""
    ts = _safe_get(event, "timestamp")
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return datetime.now(timezone.utc)
