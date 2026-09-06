"""ReAct execution pattern for the agent runtime.

In-turn tool concurrency (off by default; gated by ``tool_parallel_enabled``)
lets consecutive concurrency-safe tool calls in a single turn run as a bounded
concurrent batch instead of strictly serially. The implementation preserves
these invariants — referenced as I1–I6 in the code below — so that enabling the
flag never changes observable results, only latency:

- I1 (ordered backfill): tool results are written to the context in the model's
  original tool-call order, regardless of which tool finishes first.
- I2 (one result per call): every ``tool_call_id`` gets exactly one result,
  including failures.
- I3 (ledger order): ``tool_ledger`` insertion order matches input order after a
  batch, because the consecutive-count walks read it in reverse insertion order.
- I4 (control segment): a control tool (final_answer / send_message /
  ask_user_question) owns its segment and never shares a concurrent batch.
  What happens to the rest of the batch depends on the control result:
    * final_answer with answer text finalizes the run and clears the queue;
    * final_answer with an empty answer is rejected instead - the call gets a
      failure result, every sibling still queued is cancelled with a result of
      its own, and the next turn is forced back to final_answer;
    * ask_user_question, and send_message with expect_response=True, suspend
      for the user and discard the rest of the plan;
    * send_message with expect_response=False returns "message_sent" and
      execution continues with the next segment.
  So every branch except the last ends the turn's tool execution. The reject,
  suspend, and message branches settle every queued sibling with a result of
  its own; the finalize branch does not - it clears the queue without
  cancelling siblings, which is exactly why the strip removes a bundled
  final_answer before the batch is recorded.

  A batch that arrives here from a fresh LLM response carries no final_answer
  alongside a work tool: response normalization removes it first, because its
  answer text was written before those tools ran. That holds for fresh
  responses only - pending_tool_calls restored from a checkpoint are replayed
  without re-normalization, so a batch written by an earlier build can still
  reach this loop carrying one, and takes the branches above unchanged.
- I5 (interrupt / resume): an interrupt during a concurrent batch preserves
  calls that already completed and leaves only interrupted calls pending. A
  cancelled call may still have committed externally before cancellation was
  observed, so ``concurrency_safe`` is an explicit idempotency contract as well
  as a concurrency contract. A crash before backfill leaves the whole segment
  pending for resume under the same contract.
- I6 (trace pairing): tool trace spans pair START with END/ERROR by
  ``tool_call_id`` rather than tool name, so concurrent same-name calls do not
  cross-attribute (see ``tracing/langfuse/handler.py``).

When ``tool_parallel_enabled`` is False every segment is a single serial call,
making the loop byte-for-byte equivalent to the pre-concurrency behavior.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
from dataclasses import dataclass, replace
from datetime import timezone
from enum import Enum
from typing import Any, cast

from ....file_ref import (
    WORKSPACE_OUTPUT_FILES_TOOL_NAME,
    final_deliverable_file_reference_instructions,
)
from ....model.chat.exceptions import LLMToolProtocolError
from ....model.chat.response_shape import classify_chat_response
from ....model.chat.tool_protocol import get_tool_protocol_error
from ....tools.adapters.vibe.interaction_types import INTERACTION_TYPES
from ....tools.user_interaction import (
    tool_result_waits_for_user,
    user_interaction_resume_callable,
)
from ...clarification import draft_from_waiting_request
from ...context.enrichment import (
    IMAGE_EDIT_UNAVAILABLE_METADATA_KEY,
    enrich_context_with_memory,
    latest_user_text,
)
from ...context.memory_tool import build_memory_tools
from ...context.skill_tool import build_load_skill_tool
from ...grounding import grounding_rule
from ...language import final_answer_language_rule
from ...result import (
    CONTROL_TOOL_NAMES,
    tool_result_succeeded,
    unwrap_final_answer_content,
)
from ...runtime import (
    DISCARDED_BUNDLED_FINAL_ANSWER_REASON,
    INTERRUPTED_DURING_LLM_STREAM_REASON,
    INVALID_TOOL_PROTOCOL_AFTER_RECOVERY_REASON,
    INVALID_TOOL_PROTOCOL_AFTER_RETRY_REASON,
    INVALID_TOOL_PROTOCOL_RETRYING_REASON,
    NO_DELIVERABLE_FINAL_ANSWER_REASON,
    UNAVAILABLE_TOOL_CALL_RESTORING_TOOLS_REASON,
    ExecutionInterrupted,
    LLMCallInterrupted,
    PatternRuntime,
    ToolCallInterrupted,
    prepare_llm_for_context,
    resolved_llm_metadata,
)
from ..base import AgentPattern, PatternResult, truncate_prompt_preview
from ..final_answer_stream import ReActFinalAnswerStreamer

logger = logging.getLogger(__name__)


class ReActReasoningMode(str, Enum):
    """Reasoning strategy used by ReActPattern."""

    TOOL_CALLING = "tool_calling"
    REASONING_ACTION = "reasoning_action"


REPEATED_TOOL_DECISION_REQUESTED_STATUS = "repeated_tool_decision_requested"
DEFAULT_REPEATED_TOOL_DECISION_CONSECUTIVE_TOOL_CALLS = 4
DEFAULT_REPEATED_TOOL_DECISION_CONSECUTIVE_WORK_TOOL_CALLS = 10
REACT_DECISION_TOOL_NAME = "react_decision"
REACT_DECISION_FINAL_ANSWER = "final_answer"
USER_INTERACTION_CONTROL_TOOL_NAMES = CONTROL_TOOL_NAMES - {REACT_DECISION_FINAL_ANSWER}
REACT_DECISION_TOOL_CALL = "tool_call"
UNGROUPED_TOOL_DECISION_CATEGORIES = frozenset({"basic", "other"})
# Bounds for the final_answer-strip warning's tool-name list. Tool names come
# straight from the model, so the log line is shaped like the rest of this
# module's untrusted-input logging: bounded length, escaped, never raw.
STRIP_LOG_MAX_TOOL_NAMES = 8
STRIP_LOG_MAX_TOOL_NAME_CHARS = 64
REACT_RESPONSE_LANGUAGE_DESCRIPTION = (
    "Target natural language for user-facing prose in this ReAct response, "
    "for example English, Simplified Chinese, Traditional Chinese, or Spanish. "
    "For Chinese requests, choose Simplified Chinese or Traditional Chinese to "
    "match the request script; do not use generic Chinese. If the current user "
    "request explicitly asks to answer in another language, use that requested "
    "target language."
)


@dataclass
class ToolCallRecord:
    """Serializable ledger entry for a tool call."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    args_hash: str
    status: str
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "args": self.args,
            "args_hash": self.args_hash,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCallRecord":
        return cls(
            tool_call_id=str(data["tool_call_id"]),
            tool_name=str(data["tool_name"]),
            args=dict(data.get("args") or {}),
            args_hash=str(data.get("args_hash", "")),
            status=str(data.get("status", "pending")),
            result=data.get("result"),
            error=data.get("error"),
        )


# Every code point that Python's str.strip() or JavaScript's
# String.prototype.trim() treats as trimmable: ECMA-262 WhiteSpace (TAB VT FF
# ZWNBSP + Unicode Zs) and LineTerminator (LF CR LS PS), unioned with the five
# extra code points CPython's str.strip() treats as whitespace (U+001C-U+001F,
# U+0085).
#
# The table is frozen as a literal instead of derived from CPython's
# whitespace table for two reasons: (1) this invariant runs in the direction
# "whatever JavaScript trims, we must also trim", and CPython's whitespace
# table shifts with the Unicode version bundled in each interpreter release;
# (2) the normalized value is written back into item["field"], and the
# frontend's own trim() must be a no-op on the result -- that only holds
# while this table is a superset of the JavaScript table, which the coverage
# test in tests/core/agent/test_react.py pins down.
#
# Every code point is written as an escape, never a literal: several of them
# (U+2028/U+2029 in particular) are silently rewritten by some editors and
# transports when they appear as literal bytes.
_INTERACTION_TRIM_CHARS = (
    "\t\n\v\f\r\x1c\x1d\x1e\x1f\x20\x85\xa0"
    "\u1680\u2000\u2001\u2002\u2003\u2004\u2005"
    "\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000\ufeff"
)  # 30 code points


def _normalize_interaction_text(value: str) -> str:
    """Strip every code point either Python or JavaScript treats as trimmable.

    One pass over one union table, deliberately -- not ``value.strip()``
    followed by a second pass over the JavaScript-only characters.

    One pass is a fixed point by construction: ``str.strip(chars)`` deletes
    from both ends up to the first character not in ``chars``, so the
    returned value's first and last characters are, by definition, not in
    ``chars``; stripping the same ``chars`` again is the identity. That is
    what lets the caller write the result back into ``item["field"]`` and
    rely on the frontend's own ``trim()`` being a no-op on it.

    Two passes over two different tables would not be a fixed point: each
    pass stops at a character its own table does not contain, and that
    stopping point says nothing about the other table -- e.g. a value
    starting with U+FEFF then U+001C would have the first pass halt
    immediately on U+FEFF (Python does not treat it as space), then a second
    pass over the JavaScript-only characters would remove U+FEFF and halt on
    U+001C (JavaScript does not trim it), leaving U+001C behind. Do not
    "optimize" this back into two passes.
    """
    return value.strip(_INTERACTION_TRIM_CHARS)


def _is_non_blank_str(value: Any) -> bool:
    """True when value is a string that stays non-empty after
    _normalize_interaction_text -- the blankness judgment shared by option
    label/value filtering and field-name fallback. Takes Any (not str) so
    the isinstance check and the trim happen together, on the same value:
    calling _normalize_interaction_text directly on a fresh dict.get(...)
    expression defeats type-narrowing across the two calls.
    """
    return isinstance(value, str) and bool(_normalize_interaction_text(value))


def _normalize_ask_user_interactions(interactions: Any) -> list[dict[str, Any]]:
    """Normalize common model variants into the frontend interaction contract.

    A label or value that is blank after ``_normalize_interaction_text`` is
    treated the same as missing: the option is dropped. A field name that is
    blank after normalization falls back to ``response_{index}``; a
    well-formed field name is normalized and written back so the frontend's
    own ``trim()`` is a no-op on it. Survivors are otherwise kept verbatim --
    only blankness is judged here, not content.

    The alias chain ``field or id or name`` intentionally keeps its raw
    truthiness check; it is not normalization-aware. The frontend's own
    alias chains (clarification-form.tsx, app-context-chat.tsx) make the
    same raw-truthiness choice, and because this function always writes its
    result back into ``item["field"]``, the frontend never evaluates its own
    ``id``/``name`` fallback for a field this function has already resolved
    -- so this stays consistent with the frontend regardless of which one
    changes first.

    This function does not deduplicate field names within a single call: it
    keeps every colliding entry as its own interaction rather than dropping
    or renaming one. Each one still goes through every other step above --
    its field is trimmed, its options are filtered, ``actions`` is stripped
    -- only the name collision itself is left as-is, and a warning is
    logged. Both call sites deduplicate afterward, in the same shape
    (append ``_2``, ``_3`` to a repeated base) but at different scope: the
    single-tool call site (``ask_user_question`` in ``_handle_control_tool``)
    deduplicates within that one call's own interactions only. The
    multi-tool call site (``_pause_for_tool_results``) runs its own
    deduplication across all tools' interactions after calling this
    function once per tool, so a base already used by an earlier tool in
    the same batch is not reused by a later one either.

    The output never carries an ``actions`` key: ``actions`` is a model
    alias for ``options`` (consumed above whenever ``options`` itself is
    missing or not a list, and ``actions`` is itself a list), and leaving
    the raw, unfiltered alias in the output would give the persisted row a
    second, never-filtered carrier of the same option list.
    """

    if not isinstance(interactions, list):
        return []

    normalized: list[dict[str, Any]] = []
    for index, interaction in enumerate(interactions):
        if not isinstance(interaction, dict):
            continue

        item = dict(interaction)
        field = item.get("field") or item.get("id") or item.get("name")
        normalized_field = (
            _normalize_interaction_text(field) if isinstance(field, str) else ""
        )
        item["field"] = normalized_field or f"response_{index}"

        # Widened from "options" not in item: an interaction can carry both
        # a malformed options (present but not a list) and a well-formed
        # actions alias, and the alias is the only place the real data
        # lives in that shape -- narrower than "not in item" would leave
        # the alias unconsumed and drop every option for that interaction
        # (verified: the malformed-options-plus-actions case loses all its
        # options under the narrower condition).
        if not isinstance(item.get("options"), list) and isinstance(
            item.get("actions"), list
        ):
            item["options"] = item["actions"]

        options = item.get("options")
        if isinstance(options, list):
            filtered_options = [
                {
                    key: value
                    for key, value in {
                        "label": option.get("label"),
                        "value": option.get("value"),
                        "description": option.get("description"),
                        "action_type": option.get("action_type"),
                    }.items()
                    if value is not None
                }
                for option in options
                if isinstance(option, dict)
                and _is_non_blank_str(option.get("label"))
                and _is_non_blank_str(option.get("value"))
            ]
            if options and not filtered_options:
                # All options for this interaction were blank. The
                # interaction is still emitted (the question still goes
                # out), so this is the only signal that it happened. This
                # warning's payload is bounded to integer counts, never the
                # model-controlled field name -- the same discipline
                # STRIP_LOG_MAX_TOOL_NAMES above applies to tool names, but
                # that is this warning's own choice, not a blanket rule for
                # every log call in this module (several elsewhere put raw
                # tool names or argument keys straight into the message).
                logger.warning(
                    "ask_user_question dropped all %d option(s) for interaction %d",
                    len(options),
                    index,
                    extra={
                        "dropped": len(options),
                        "total": len(options),
                        "interaction_index": index,
                    },
                )
            item["options"] = filtered_options
        elif "options" in item:
            # options is present but neither a list nor rescued by the
            # actions alias above -- a malformed shape this function has
            # always left untouched, but silently: nothing signaled that
            # it happened. Same payload discipline as the other two
            # warnings in this function: bounded, integer-only.
            logger.warning(
                "ask_user_question interaction %d has a non-list options value",
                index,
                extra={"interaction_index": index},
            )

        # Leave only one carrier of the option list in the output. Whether
        # or not the alias branch above used actions to seed options,
        # item["actions"] itself is never touched by the filter step above
        # (that step reassigns item["options"] to a new, filtered list and
        # leaves the actions key exactly as the model gave it) -- so
        # whatever is left under item["actions"] is always the original,
        # unfiltered list: either the same content options was seeded
        # from, pre-filter, or, when options was itself already a list, a
        # completely unrelated list the filter never saw. Either way,
        # leaving it in the output gives the persisted row a second,
        # unfiltered carrier of option data that gets stored verbatim into
        # task_chat_messages.interactions and replayed unchanged.
        # Unconditional so this holds regardless of whether options ended
        # up a list -- must run after the alias branch above, not before:
        # popping actions first would delete the only place a malformed
        # options's real data lives before the alias branch can consume it,
        # dropping every option for that interaction.
        item.pop("actions", None)

        normalized.append(item)

    field_counts: dict[str, int] = {}
    for item in normalized:
        field_counts[item["field"]] = field_counts.get(item["field"], 0) + 1
    colliding_field_count = sum(1 for count in field_counts.values() if count > 1)
    if colliding_field_count:
        # Same payload discipline as the other warnings in this function:
        # integer counts only, never the colliding field name itself.
        logger.warning(
            "ask_user_question interactions have %d colliding field name(s) out of %d",
            colliding_field_count,
            len(normalized),
            extra={
                "colliding_field_count": colliding_field_count,
                "total": len(normalized),
            },
        )

    return normalized


class ReActPattern(AgentPattern):
    """Minimal ReAct loop for the execution runtime."""

    def __init__(
        self,
        llm: Any | None = None,
        *,
        # Intentionally high for interactive and long-running agent tasks; callers
        # can pass a lower value when they need stricter cost or latency bounds.
        max_iterations: int = 200,
        tool_choice: str | dict[str, Any] | None = "required",
        reasoning_mode: ReActReasoningMode | str = ReActReasoningMode.TOOL_CALLING,
        finalize_after_tool_result: bool = False,
        repeated_tool_decision_after_consecutive_tool_calls: int | None = (
            DEFAULT_REPEATED_TOOL_DECISION_CONSECUTIVE_TOOL_CALLS
        ),
        repeated_tool_decision_after_consecutive_work_tool_calls: int | None = (
            DEFAULT_REPEATED_TOOL_DECISION_CONSECUTIVE_WORK_TOOL_CALLS
        ),
        tool_parallel_enabled: bool = False,
        tool_max_concurrency: int = 3,
        user_interaction_enabled: bool = True,
    ) -> None:
        self.llm = llm
        self.max_iterations = max_iterations
        self.tool_choice = tool_choice
        self.reasoning_mode = ReActReasoningMode(reasoning_mode)
        self.finalize_after_tool_result = finalize_after_tool_result
        # In-turn tool concurrency (default off). When enabled, consecutive
        # concurrency-safe tool calls in a single turn run as a concurrent
        # batch bounded by ``tool_max_concurrency``.
        self.tool_parallel_enabled = tool_parallel_enabled
        self.tool_max_concurrency = max(1, int(tool_max_concurrency))
        self.user_interaction_enabled = user_interaction_enabled
        self.repeated_tool_decision_after_consecutive_tool_calls = (
            repeated_tool_decision_after_consecutive_tool_calls
        )
        self.repeated_tool_decision_after_consecutive_work_tool_calls = (
            repeated_tool_decision_after_consecutive_work_tool_calls
        )
        self.status = "idle"
        self.current_iteration = 0
        self.last_response: Any = None
        self.pending_tool_calls: list[dict[str, Any]] = []
        self.pending_tool_call_content: dict[str, str] = {}
        self.tool_ledger: dict[str, ToolCallRecord] = {}
        self.force_final_answer_next = False
        self.repeated_tool_decision: dict[str, Any] | None = None
        self.waiting_for_user_request: dict[str, Any] | None = None
        self.pending_tool_interaction_responses: list[dict[str, str]] = []
        self.task_text: str | None = None
        self.memory_input_text: str | None = None
        self._memory_store: Any | None = None
        self._tool_decision_groups_by_name: dict[str, str] = {}

    async def run(
        self,
        context: Any,
        tools: list[Any],
        llm: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        runtime = kwargs.get("runtime")
        if runtime is None:
            runtime = PatternRuntime(
                execution_id=getattr(context, "execution_id", None)
            )
        elif getattr(runtime, "execution_id", None) is None:
            setattr(runtime, "execution_id", getattr(context, "execution_id", None))

        active_llm = llm or self.llm
        compact_llm = kwargs.get("compact_llm")
        if active_llm is None:
            return PatternResult(
                success=False,
                error="ReActPattern requires an llm instance.",
            ).to_dict()

        await runtime.on_pattern_start(context=context, pattern=self)
        waiting_result = await self._resume_waiting_for_user_if_needed(
            context=context,
            runtime=runtime,
            tools=tools,
        )
        if waiting_result is not None:
            await runtime.on_pattern_end(
                context=context,
                pattern=self,
                result=waiting_result,
            )
            return waiting_result

        if self.reasoning_mode == ReActReasoningMode.REASONING_ACTION:
            self.status = "failed"
            result = PatternResult(
                success=False,
                error=(
                    "ReActPattern reasoning_action mode is reserved for a future "
                    "implementation; use tool_calling mode for this release."
                ),
                metadata={
                    "status": self.status,
                    "reasoning_mode": self.reasoning_mode.value,
                    "error_type": "not_implemented",
                },
            ).to_dict()
            await runtime.on_pattern_end(context=context, pattern=self, result=result)
            return result

        try:
            task_text = self._task_text(context)
            memory_text = self._memory_text(context, execution_text=task_text)
            self._memory_store = kwargs.get("memory_store")
            # DAG steps skip the automatic retrieval: the root run already
            # retrieved for the whole task, and steps can search_memory on
            # demand.
            if not context.metadata.get("dag_step_id"):
                await enrich_context_with_memory(
                    context=context,
                    query=memory_text,
                    category="react_memory",
                    memory_store=self._memory_store,
                    runtime=runtime,
                    similarity_threshold=kwargs.get("memory_similarity_threshold"),
                )
            context_tools = await self._with_context_tools(
                tools=tools,
                context=context,
                task_text=memory_text,
                runtime=runtime,
                skill_manager=kwargs.get("skill_manager"),
                allowed_skills=kwargs.get("allowed_skills"),
            )
            await self._deliver_pending_tool_interaction_responses(
                tools=context_tools,
                context=context,
                runtime=runtime,
            )
            result = await self._run_tool_calling_loop(
                context=context,
                tools=context_tools,
                llm=active_llm,
                compact_llm=compact_llm,
                runtime=runtime,
            )
        except ExecutionInterrupted as exc:
            interrupted = await self._interrupt_if_requested(
                runtime=runtime,
                context=context,
                label=(
                    "during_tool"
                    if isinstance(exc, ToolCallInterrupted)
                    else "during_enrichment"
                ),
            )
            if interrupted is None:
                raise
            result = interrupted
        except Exception as exc:
            await runtime.on_pattern_error(context=context, pattern=self, error=exc)
            raise

        await runtime.on_pattern_end(context=context, pattern=self, result=result)
        return result

    async def _with_context_tools(
        self,
        *,
        tools: list[Any],
        context: Any,
        task_text: str,
        runtime: PatternRuntime,
        skill_manager: Any | None,
        allowed_skills: list[str] | None,
    ) -> list[Any]:
        """Expose the memory tool set and ``load_skill`` for this run.

        Skipped when ``finalize_after_tool_result`` is set (single_call mode):
        that mode forces a final answer after the first successful tool call,
        so the only tool round must go to the actual task.
        """
        if self.finalize_after_tool_result:
            return tools
        extra_tools: list[Any] = build_memory_tools(
            memory_store=self._memory_store,
            task=task_text,
            runtime=runtime,
            context=context,
        )
        skill_tool = await build_load_skill_tool(
            skill_manager=skill_manager,
            context=context,
            allowed_skills=allowed_skills,
        )
        if skill_tool is not None:
            extra_tools.append(skill_tool)
        if not extra_tools:
            return tools
        return [*tools, *extra_tools]

    async def _run_tool_calling_loop(
        self,
        *,
        context: Any,
        tools: list[Any],
        llm: Any,
        compact_llm: Any | None,
        runtime: PatternRuntime,
    ) -> dict[str, Any]:
        self.status = "thinking"
        self._tool_decision_groups_by_name = self._tool_decision_groups_for_tools(tools)
        # Read by get_system_context to contradict skill text naming edit_image.
        context.metadata[IMAGE_EDIT_UNAVAILABLE_METADATA_KEY] = (
            "generate_image" in self._tool_decision_groups_by_name
            and "edit_image" not in self._tool_decision_groups_by_name
        )
        base_tool_schemas = (
            []
            if self.tool_choice == "none"
            else self._tool_schemas_with_builtin_controls(tools)
        )

        for iteration in range(self.current_iteration, self.max_iterations):
            self.current_iteration = iteration
            if self.pending_tool_calls:
                self._ensure_pending_tool_call_envelope(context)
                pending_result = await self._execute_pending_tool_calls(
                    context=context,
                    tools=tools,
                    llm=llm,
                    runtime=runtime,
                )
                if pending_result is not None:
                    return pending_result
                self.current_iteration = iteration + 1
                self.status = "thinking"
                continue

            if self.repeated_tool_decision:
                decision_result = await self._run_repeated_tool_decision(
                    context=context,
                    llm=llm,
                    runtime=runtime,
                )
                if decision_result is not None:
                    return decision_result

            force_final_answer_now = self.force_final_answer_next or (
                self.finalize_after_tool_result
                and not self.pending_tool_calls
                and self._latest_tool_result_success(context)
            )
            tool_schemas = (
                [self._final_answer_tool_schema()]
                if force_final_answer_now
                else base_tool_schemas
            )
            interrupted = await self._interrupt_if_requested(
                runtime=runtime,
                context=context,
                label="before_llm",
            )
            if interrupted is not None:
                return interrupted

            route_messages = self._messages_for_llm(
                context,
                has_tools=bool(tool_schemas),
                force_final_answer=force_final_answer_now,
                tool_names=self._schema_tool_names(tool_schemas),
            )
            call_llm = await prepare_llm_for_context(
                llm=llm,
                messages=route_messages,
                context=context,
            )
            llm_metadata = {
                "iteration": iteration,
                **resolved_llm_metadata(call_llm),
            }
            await runtime.compact_context_if_needed(
                context=context,
                # Fall back to the main model when no compact model is
                # configured. PatternRuntime skips summarization entirely
                # without one and drops all but the last few messages
                # instead, losing what the agent actually did; agent preview
                # and delegated sub-agents resolve the compact slot on their
                # own and validate only the default model, so an empty slot
                # is ordinary rather than exceptional.
                #
                # Substituting here, rather than defaulting the field further
                # up, keeps "unset" distinguishable from "explicitly set to
                # the main model" -- and hands compaction the *resolved*
                # per-call model, so a virtual model reuses this turn's
                # routing decision instead of routing again on the compaction
                # prompt, whose only user message is the whole transcript.
                llm=compact_llm if compact_llm is not None else call_llm,
                metadata={"iteration": iteration},
            )

            messages = self._messages_for_llm(
                context,
                has_tools=bool(tool_schemas),
                force_final_answer=force_final_answer_now,
                tool_names=self._schema_tool_names(tool_schemas),
            )
            await runtime.checkpoint("before_llm", context=context, pattern=self)
            await runtime.on_llm_start(
                context=context,
                messages=messages,
                tools=tool_schemas or None,
                metadata=llm_metadata,
            )
            answer_streamer: ReActFinalAnswerStreamer | None = None
            protocol_retry_performed = False
            response_already_traced = False
            try:
                effective_tool_choice = (
                    "required"
                    if force_final_answer_now
                    else self.tool_choice
                    if tool_schemas
                    else None
                )
                llm_kwargs = {
                    "messages": messages,
                    "tools": tool_schemas or None,
                    "tool_choice": effective_tool_choice,
                }
                if tool_schemas:
                    answer_streamer = ReActFinalAnswerStreamer(runtime)
                    response = await runtime.run_streaming_llm_call(
                        call_llm,
                        on_chunk=answer_streamer.handle_chunk,
                        **llm_kwargs,
                    )
                else:
                    response = await runtime.stream_final_answer(call_llm, **llm_kwargs)
            except LLMCallInterrupted:
                if answer_streamer is not None:
                    await answer_streamer.fail(INTERRUPTED_DURING_LLM_STREAM_REASON)
                interrupted = await self._interrupt_if_requested(
                    runtime=runtime,
                    context=context,
                    label="during_llm",
                )
                if interrupted is not None:
                    return interrupted
                raise
            except LLMToolProtocolError as exc:
                unavailable_tool_call = exc.code == "unavailable_tool_call"
                if answer_streamer is not None:
                    await answer_streamer.fail(
                        UNAVAILABLE_TOOL_CALL_RESTORING_TOOLS_REASON
                        if unavailable_tool_call
                        else f"invalid {exc.code} tool protocol, retrying"
                    )
                await runtime.on_llm_error(
                    context=context,
                    error=exc,
                    metadata={
                        **llm_metadata,
                        "phase": exc.code,
                        "protocol_code": exc.code,
                    },
                )
                try:
                    (
                        response,
                        answer_streamer,
                    ) = await self._retry_tool_protocol_response(
                        context=context,
                        llm=call_llm,
                        runtime=runtime,
                        iteration=iteration,
                        tool_schemas=base_tool_schemas,
                        force_final_answer=(
                            force_final_answer_now and not unavailable_tool_call
                        ),
                        recovery_reason=exc.code,
                    )
                except LLMCallInterrupted:
                    interrupted = await self._interrupt_if_requested(
                        runtime=runtime,
                        context=context,
                        label="during_llm",
                    )
                    if interrupted is not None:
                        return interrupted
                    raise
                if unavailable_tool_call:
                    self.force_final_answer_next = False
                    force_final_answer_now = False
                protocol_retry_performed = True
                response_already_traced = True
            except Exception as exc:
                if answer_streamer is not None:
                    await answer_streamer.fail(str(exc))
                raise
            self.repeated_tool_decision = None
            self.last_response = response
            normalized = self._normalize_llm_response(response)
            requires_protocol_retry = self._response_requires_tool_protocol_retry(
                normalized,
                force_final_answer=force_final_answer_now,
                reject_mixed_control_calls=protocol_retry_performed,
            )
            end_metadata: dict[str, Any] = dict(llm_metadata)
            if requires_protocol_retry:
                end_metadata.update(
                    success=False,
                    phase="discarded_invalid_tool_protocol",
                )
            if not response_already_traced:
                await runtime.on_llm_end(
                    context=context,
                    response=response,
                    metadata=end_metadata,
                )
            if requires_protocol_retry:
                if protocol_retry_performed:
                    return await self._invalid_tool_protocol_result(
                        runtime=runtime,
                        context=context,
                        iteration=iteration,
                        answer_streamer=answer_streamer,
                        stream_failure_message=INVALID_TOOL_PROTOCOL_AFTER_RECOVERY_REASON,
                        empty_final_answer=(
                            self._empty_final_answer_call(normalized) is not None
                        ),
                    )
                if answer_streamer is not None:
                    await answer_streamer.fail(INVALID_TOOL_PROTOCOL_RETRYING_REASON)
                # Rejecting the whole response drops any assistant preamble it
                # carried: the response is discarded before ``add_assistant_message``,
                # so the retry rebuilds from context without it. Deliberate - the
                # preamble arrived attached to a protocol violation, so it is not a
                # vetted user-facing answer, and replaying the model's own discarded
                # prose invites it to treat that text as already committed. The cost
                # is that a model which keeps emitting "preamble + empty answer"
                # fails the run without the user seeing the preamble.
                recover_full_tool_set = self._requires_full_tool_set_recovery(
                    normalized,
                    force_final_answer=force_final_answer_now,
                )
                empty_final_answer = self._empty_final_answer_call(normalized)
                if empty_final_answer is not None:
                    logger.warning(
                        "ReAct final_answer carried no answer text; discarding the "
                        "response and retrying. iteration=%s arg_keys=%s",
                        iteration,
                        sorted(self._tool_call_args_dict(empty_final_answer), key=str),
                    )
                if recover_full_tool_set:
                    recovery_reason: str | None = "unavailable_tool_call"
                elif empty_final_answer is not None:
                    recovery_reason = "empty_final_answer"
                else:
                    recovery_reason = None
                try:
                    (
                        response,
                        answer_streamer,
                    ) = await self._retry_tool_protocol_response(
                        context=context,
                        llm=call_llm,
                        runtime=runtime,
                        iteration=iteration,
                        tool_schemas=base_tool_schemas,
                        force_final_answer=(
                            force_final_answer_now and not recover_full_tool_set
                        ),
                        recovery_reason=recovery_reason,
                        empty_final_answer=empty_final_answer is not None,
                    )
                except LLMCallInterrupted:
                    interrupted = await self._interrupt_if_requested(
                        runtime=runtime,
                        context=context,
                        label="during_llm",
                    )
                    if interrupted is not None:
                        return interrupted
                    raise
                if recover_full_tool_set:
                    self.force_final_answer_next = False
                    force_final_answer_now = False
                self.last_response = response
                normalized = self._normalize_llm_response(response)
                if self._response_requires_tool_protocol_retry(
                    normalized,
                    force_final_answer=force_final_answer_now,
                    reject_mixed_control_calls=recover_full_tool_set,
                ):
                    return await self._invalid_tool_protocol_result(
                        runtime=runtime,
                        context=context,
                        iteration=iteration,
                        answer_streamer=answer_streamer,
                        stream_failure_message=INVALID_TOOL_PROTOCOL_AFTER_RETRY_REASON,
                        empty_final_answer=(
                            self._empty_final_answer_call(normalized) is not None
                        ),
                    )
            original_tool_calls = normalized.get("tool_calls") or []
            kept_tool_calls, stripped_final_answers = (
                self._strip_final_answer_bundled_with_work_tools(original_tool_calls)
            )
            if stripped_final_answers:
                logger.warning(
                    "ReAct discarding %d final_answer call(s) bundled with work "
                    "tools; the answer text predates their results. "
                    "iteration=%s model=%s batch_size=%d tools=[%s]",
                    len(stripped_final_answers),
                    iteration,
                    llm_metadata.get("selected_model"),
                    len(original_tool_calls),
                    self._tool_names_for_log(original_tool_calls),
                )
                normalized["tool_calls"] = kept_tool_calls
                if answer_streamer is not None:
                    # The discarded text may already be streaming to the UI.
                    # Nothing downstream closes that stream once the batch no
                    # longer carries a final_answer, so close it here.
                    await answer_streamer.fail(DISCARDED_BUNDLED_FINAL_ANSWER_REASON)
            if force_final_answer_now and not normalized.get("tool_calls"):
                normalized["done"] = True

            assistant_content = normalized.get("content")
            tool_calls = normalized.get("tool_calls", [])
            if assistant_content is not None or normalized.get("tool_calls"):
                # A tool-protocol error response never carries tool_calls (see
                # tool_protocol_error_response), so this guard never mistakes
                # a protocol violation for a real tool-call turn worth saving
                # provider state for.
                metadata = (
                    self._provider_state_for_context(normalized) if tool_calls else {}
                )
                context.add_assistant_message(
                    assistant_content or "",
                    tool_calls=[
                        self._tool_call_for_context(tool_call)
                        for tool_call in tool_calls
                    ],
                    **({"metadata": metadata} if metadata else {}),
                )

            if answer_streamer is not None:
                await self._close_streamed_answer(
                    answer_streamer=answer_streamer,
                    assistant_content=assistant_content,
                    tool_calls=tool_calls,
                )
            if tool_calls:
                self._remember_tool_call_content(tool_calls, assistant_content)
                self.status = "acting"
                self.pending_tool_calls = list(tool_calls)
                await runtime.checkpoint("after_llm", context=context, pattern=self)
                pending_result = await self._execute_pending_tool_calls(
                    context=context,
                    tools=tools,
                    llm=llm,
                    runtime=runtime,
                )
                if pending_result is not None:
                    return pending_result
                self.current_iteration = iteration + 1
                self.status = "thinking"
                continue

            await runtime.checkpoint("after_llm", context=context, pattern=self)
            if normalized.get("done", True):
                # Fall back to the raw response's usable *text*, never the
                # raw value itself: for envelope adapters the raw is the
                # whole envelope dict, and stringifying it downstream would
                # leak an internal repr into the user-visible transcript.
                response = assistant_content
                if not response:
                    raw_shape = classify_chat_response(normalized.get("raw"))
                    response = raw_shape.text if raw_shape.kind == "text" else ""
                return await self._finalize_success(
                    context=context,
                    runtime=runtime,
                    response=response,
                )

        self.status = "max_iterations"
        await runtime.checkpoint("max_iterations", context=context, pattern=self)
        return PatternResult(
            success=False,
            error="ReActPattern reached max iterations without a final answer.",
            metadata={"iterations": self.max_iterations, "status": self.status},
        ).to_dict()

    async def _invalid_tool_protocol_result(
        self,
        *,
        runtime: PatternRuntime,
        context: Any,
        iteration: int,
        answer_streamer: ReActFinalAnswerStreamer | None,
        stream_failure_message: str,
        empty_final_answer: bool = False,
    ) -> dict[str, Any]:
        """Abandon the run after the one repair attempt failed.

        ``empty_final_answer`` distinguishes "the model never produced an answer"
        from the status's other producers (provider protocol errors, mixed
        control calls, a non-``final_answer`` tool on a forced turn), whose
        ``error`` text is the only signal a caller has. Delegated-child
        classification reads it to avoid collapsing all four into "never
        produced an answer" - see ``agent_tool._classify_delegated_failure``.
        """

        logger.warning(
            "ReAct failing the run after an invalid tool protocol: %s "
            "(iteration=%s empty_final_answer=%s)",
            stream_failure_message,
            iteration,
            empty_final_answer,
        )
        if answer_streamer is not None:
            await answer_streamer.fail(stream_failure_message)
        await runtime.checkpoint(
            "invalid_tool_protocol",
            context=context,
            pattern=self,
            metadata={"iteration": iteration},
        )
        error = (
            "The model called final_answer without an answer twice, so the run "
            "produced no response."
            if empty_final_answer
            else "The model returned an invalid tool protocol response "
            "after one repair attempt."
        )
        return PatternResult(
            success=False,
            error=error,
            metadata={
                "iterations": iteration + 1,
                "status": "invalid_tool_protocol",
                "empty_final_answer": empty_final_answer,
            },
        ).to_dict()

    async def _close_streamed_answer(
        self,
        *,
        answer_streamer: ReActFinalAnswerStreamer,
        assistant_content: Any,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """Ensure a started answer stream reaches exactly one terminal event.

        The branches below are R0 (nothing streamed - no-op), R1
        (``tool_calls[0]`` is a ``final_answer`` with a non-blank answer and
        no disabled user-interaction control tool in the batch, per
        ``_disabled_control_tool_index`` - ``finish`` with that answer's
        exact, unstripped text, the same text ``_handle_control_tool``
        delivers), R2 (no tool calls, plain assistant text - ``finish`` with
        that text) and R3 (anything else - ``fail`` with a fixed reason;
        ``fail`` is a no-op for a stream already closed earlier in this
        response).

        Do not relax R1's first-position condition. A later ``final_answer``
        in a mixed batch can still be delivered
        (``_execute_pending_tool_calls`` keeps walking past e.g. a
        ``send_message`` that expects no response), but such a batch never
        leaves an open stream to close: ``ReActFinalAnswerStreamer``
        permanently disables itself as soon as a non-``final_answer`` tool
        name appears in the response, before any answer content for a
        non-first ``final_answer`` has accumulated, so R0 applies. Relaxing
        the condition would finish streams with candidates whose delivery
        this method has not checked. Do not turn R3 into a ``finish`` to
        avoid the error event it produces - that would report an undelivered
        candidate as the completed answer.
        """

        if not answer_streamer.started:
            return
        if tool_calls and tool_calls[0].get("name") == "final_answer":
            answer = self._final_answer_text(tool_calls[0].get("args"))
            if answer.strip() and (
                self._disabled_control_tool_index(
                    tool_calls,
                    user_interaction_enabled=self.user_interaction_enabled,
                )
                is None
            ):
                await answer_streamer.finish(answer)
                return
        if not tool_calls and assistant_content is not None:
            await answer_streamer.finish(str(assistant_content))
            return
        await answer_streamer.fail(NO_DELIVERABLE_FINAL_ANSWER_REASON)

    def _messages_for_llm(
        self,
        context: Any,
        *,
        has_tools: bool,
        force_final_answer: bool = False,
        tool_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        messages = list(context.get_messages_for_llm())
        if force_final_answer:
            instruction = (
                "Produce the final user-facing answer by calling the final_answer "
                "control tool exactly once using the accumulated conversation and "
                "tool results. Do not call any other tool and do not output "
                "tool-call markup as plain text. Set outcome=completed only when "
                "every requested action or verification succeeded; otherwise set "
                "outcome=partial or outcome=blocked and say what remains. If a "
                "previous ask_user_question narrowed the request to a selected "
                "subset of items or resources, the final answer must cover only "
                "that subset — leave out anything outside it even if an earlier "
                "tool call already returned data about it. "
                f"{grounding_rule(can_call_tools=False)}\n\n"
                f"{final_deliverable_file_reference_instructions(can_lookup=False)}\n\n"
                f"{final_answer_language_rule()}"
            )
        elif has_tools:
            active_tool_names = tool_names or []
            available_tools = ", ".join(active_tool_names) or "(none)"
            can_lookup_output_files = (
                WORKSPACE_OUTPUT_FILES_TOOL_NAME in active_tool_names
            )
            clock_zone = context.clock_zone()
            current_date = (
                context.created_at.astimezone(clock_zone or timezone.utc)
                .date()
                .isoformat()
            )
            clock_zone_label = clock_zone.key if clock_zone is not None else "UTC"
            missing_information_instruction = (
                "If a tool needs missing information from the user, including a "
                "fact-carrying argument value (one that asserts a real-world "
                "fact) the user has not provided, call "
                "ask_user_question; do not ask the question as plain assistant "
                "text and do not fill the value in yourself. When the user answers "
                "an ask_user_question that asked which items or resources to act on "
                "by selecting a subset of the offered options, that selection is the "
                "complete scope for that work: do not include unselected options, "
                "even ones that are already accessible or that fit the original "
                "request. "
                if self.user_interaction_enabled
                else "If missing user information prevents completion, including a "
                "fact-carrying argument value (one that asserts a real-world fact) "
                "the user has not provided, do not ask "
                "the user or attempt an unavailable interaction tool; finish with "
                "outcome=blocked and explain what is missing. "
            )
            instruction = (
                "Use available tools when the user asks you to generate, compute, run, "
                "execute, inspect, read, write, or otherwise produce a concrete result "
                "that a tool can determine. After a successful tool call, base the "
                "final answer on the latest tool result instead of repeating the same "
                "tool work. When the current task is complete, call the final_answer "
                "tool exactly once instead of calling another work tool or returning "
                "plain assistant text. Never put final_answer in the same response "
                "as any other tool call: run the work tools first, then answer on a "
                "later turn from their results. Do not write assistant text in the "
                "same response as a work tool call; call the tool directly. "
                f"{missing_information_instruction}"
                "If the latest user "
                "message explicitly asks you to call a named available tool, call "
                "that tool instead of paraphrasing the request. If a tool "
                "fails, retry with a corrected call when possible; "
                "otherwise explain the failure instead of presenting an unverified "
                "tutorial or example. Treat the latest user message as the controlling "
                "instruction for follow-up requests. If the user corrects a previous "
                "assumption, especially about dates or freshness, revise the answer "
                "instead of restating prior content. "
                f"{grounding_rule()}\n\n"
                "When writing any final user-facing response, including plain "
                "assistant text: "
                f"{final_deliverable_file_reference_instructions(can_lookup=can_lookup_output_files, include_heading=False)}\n\n"
                f"Turn-start date ({clock_zone_label}): {current_date}. "
                "For recent, latest, current, or time-sensitive requests, use this "
                "date when forming search queries and judging source relevance. If the "
                "exact current time matters or the turn may have crossed midnight, call "
                "the get_current_time tool if it is available. "
                "Only call tools that are present in the current tool schema for this "
                "LLM call; "
                "tool names mentioned in memory, previous tasks, plans, or error "
                "messages are unavailable unless they are included in the current "
                "schema. If a selected skill is already present in the system "
                "context, treat its main SKILL.md guidance as already read. Use "
                "skill documentation tools only when you need an additional "
                "referenced file, example, asset, or detail that is not already in "
                "the provided skill context."
                f"\n\nAvailable tool names for this LLM call are exactly: {available_tools}. "
                "Never call a tool name that is not in this list."
            )
        else:
            # Reachable only with tool_choice="none", which no production
            # construction site sets. If that ever changes, this branch needs
            # grounding_rule() too -- it emits a final answer without it today.
            return messages
        completion_instruction = self._completion_evidence_instruction(context)
        if completion_instruction:
            instruction = f"{instruction}\n\n{completion_instruction}"
        if messages and messages[0].get("role") == "system":
            return [
                {
                    **messages[0],
                    "content": f"{messages[0].get('content', '')}\n\n{instruction}",
                },
                *messages[1:],
            ]
        return [{"role": "system", "content": instruction}, *messages]

    async def _retry_tool_protocol_response(
        self,
        *,
        context: Any,
        llm: Any,
        runtime: PatternRuntime,
        iteration: int,
        tool_schemas: list[dict[str, Any]],
        force_final_answer: bool,
        recovery_reason: str | None = None,
        empty_final_answer: bool = False,
    ) -> tuple[Any, ReActFinalAnswerStreamer]:
        tools = (
            [self._final_answer_tool_schema()] if force_final_answer else tool_schemas
        )
        messages = self._messages_for_llm(
            context,
            has_tools=True,
            force_final_answer=force_final_answer,
            tool_names=self._schema_tool_names(tools),
        )
        if recovery_reason == "unavailable_tool_call":
            retry_instruction = (
                "The previous response called a tool that was unavailable in the "
                "narrowed tool schema. Re-decide this turn using the complete "
                "current tool set listed above. Call only a tool present in that "
                "set. If the requested work or artifact has not been successfully "
                "produced, call the appropriate work tool; call final_answer only "
                "when the task is actually complete and its required results exist."
            )
            retry_phase = "unavailable_tool_call_recovery"
        elif recovery_reason == "malformed_tool_arguments":
            retry_instruction = (
                "The previous response returned malformed JSON arguments for a "
                "tool call. Retry this turn by calling exactly one available tool "
                "with one complete JSON object that matches its schema. Do not "
                "truncate, concatenate, manually serialize, or wrap the arguments "
                "in prose. If final_answer is the only available tool, put the "
                "complete user-facing response in its answer field."
            )
            retry_phase = "malformed_tool_arguments_recovery"
        elif recovery_reason == "empty_final_answer":
            # On a forced turn ``tools`` above is final_answer alone, so offering
            # a work tool would instruct the model to do something the schema
            # forbids and waste the one repair attempt.
            retry_instruction = (
                "The previous response called final_answer with an empty answer "
                "field, so the user received no reply at all. "
                + (
                    "final_answer is the only tool available on this turn: call it "
                    "again with the complete user-facing response in its answer "
                    "field."
                    if force_final_answer
                    else "Retry this turn: if the task is complete, call "
                    "final_answer again with the complete user-facing response in "
                    "its answer field; if work remains, call the appropriate "
                    "available work tool instead."
                )
                + " Never call final_answer with an omitted, empty, or "
                "whitespace-only answer."
            )
            retry_phase = "empty_final_answer_recovery"
        else:
            retry_instruction = (
                "The previous response used an invalid tool protocol. Retry the same "
                "turn using native structured tool calls only. Never place one tool "
                "invocation or its arguments inside another tool's arguments. If "
                "work remains, call the appropriate available work tool directly; "
                "call final_answer only when the task is actually complete."
            )
            retry_phase = "tool_protocol_retry"
        if empty_final_answer and recovery_reason != "empty_final_answer":
            # A more fundamental repair owns the instruction, but the empty answer
            # was still detected on this response and would otherwise go
            # unmentioned.
            retry_instruction = (
                f"{retry_instruction} The previous response also called "
                "final_answer with an empty answer field; never call final_answer "
                "with an omitted, empty, or whitespace-only answer."
            )
        messages[0] = {
            **messages[0],
            "content": f"{messages[0].get('content', '')}\n\n{retry_instruction}",
        }
        metadata = {
            "iteration": iteration,
            "phase": retry_phase,
            **resolved_llm_metadata(llm),
        }
        if recovery_reason:
            metadata["recovery_reason"] = recovery_reason
        await runtime.checkpoint(
            retry_phase,
            context=context,
            pattern=self,
            metadata=metadata,
        )
        await runtime.on_llm_start(
            context=context,
            messages=messages,
            tools=tools,
            metadata=metadata,
        )
        answer_streamer = ReActFinalAnswerStreamer(runtime)
        try:
            response = await runtime.run_streaming_llm_call(
                llm,
                messages=messages,
                tools=tools,
                tool_choice="required",
                on_chunk=answer_streamer.handle_chunk,
            )
        except LLMCallInterrupted:
            await answer_streamer.fail(INTERRUPTED_DURING_LLM_STREAM_REASON)
            raise
        except Exception as exc:
            await answer_streamer.fail(str(exc))
            await runtime.on_llm_error(
                context=context,
                error=exc,
                metadata=metadata,
            )
            raise
        normalized = self._normalize_llm_response(response)
        retry_is_invalid = self._response_requires_tool_protocol_retry(
            normalized,
            force_final_answer=force_final_answer,
            reject_mixed_control_calls=(recovery_reason == "unavailable_tool_call"),
        )
        end_metadata = dict(metadata)
        if retry_is_invalid:
            end_metadata.update(
                success=False,
                phase="discarded_invalid_tool_protocol_retry",
            )
        await runtime.on_llm_end(
            context=context,
            response=response,
            metadata=end_metadata,
        )
        return response, answer_streamer

    def _response_requires_tool_protocol_retry(
        self,
        normalized: dict[str, Any],
        *,
        force_final_answer: bool,
        reject_mixed_control_calls: bool = False,
    ) -> bool:
        if get_tool_protocol_error(normalized.get("raw")) is not None:
            return True
        tool_calls = normalized.get("tool_calls") or []
        if (
            reject_mixed_control_calls
            and len(tool_calls) > 1
            and any(
                isinstance(tool_call, dict)
                and tool_call.get("name") in self._control_tool_names()
                for tool_call in tool_calls
            )
        ):
            return True
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            if force_final_answer and tool_call.get("name") != "final_answer":
                return True
        if self._batch_carries_work_tool(tool_calls):
            # A final_answer sharing the batch with a work tool is removed by
            # _strip_final_answer_bundled_with_work_tools before the batch is
            # recorded, so its answer field never reaches the user and cannot
            # be a reason to discard the whole response. Discarding here would
            # take the work calls with it - the exact behavior this guard is
            # not allowed to reintroduce. The empty-answer repair owns
            # responses whose only calls are control tools.
            return False
        return self._empty_final_answer_call(normalized) is not None

    def _empty_final_answer_call(
        self, normalized: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Return the first ``final_answer`` call that carries no answer text.

        Under ``tool_choice="required"`` ``final_answer`` is the only way ReAct
        can reply, so an empty ``answer`` produces a run that completes with
        nothing streamed and nothing persisted. Treating it as a tool-protocol
        violation routes it into the pattern's existing single repair retry
        instead of finalizing silently.
        """

        for tool_call in normalized.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            if tool_call.get("name") != "final_answer":
                continue
            if not self._final_answer_text(tool_call.get("args")).strip():
                return tool_call
        return None

    def _batch_carries_work_tool(self, tool_calls: list[dict[str, Any]]) -> bool:
        """Whether a tool-call batch contains at least one non-control tool.

        The single predicate for "this batch does real work", shared by the
        final_answer strip and the empty-answer guard so the two cannot
        disagree about which batches the strip owns.
        """

        control_tool_names = self._control_tool_names()
        return any(
            isinstance(tool_call, dict)
            and tool_call.get("name") not in control_tool_names
            for tool_call in tool_calls
        )

    def _strip_final_answer_bundled_with_work_tools(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split a batch that mixes ``final_answer`` with real work tools.

        Returns ``(kept, removed)``. A ``final_answer`` sharing a response with
        a work tool was written before that tool ran, so its text cannot
        describe the result it claims to summarize. Dropping it lets the work
        tools execute and the next iteration answer from real results.

        Must run before the batch reaches ``add_assistant_message``. A
        ``final_answer`` recorded in the message history never receives a tool
        result, and ``ExecutionContext._sanitize_tool_message_pairs`` then drops
        the whole assistant block together with the work-tool results the model
        needs on the next turn.

        Control-only batches are returned unchanged. ``send_message`` with
        ``expect_response=False`` continues the turn by design, and a batch
        with no work tool carries no result the answer could be missing.
        """

        if not self._batch_carries_work_tool(tool_calls):
            return tool_calls, []
        kept: list[dict[str, Any]] = []
        removed: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict) and tool_call.get("name") == "final_answer":
                removed.append(tool_call)
            else:
                kept.append(tool_call)
        return kept, removed

    def _tool_names_for_log(self, tool_calls: list[dict[str, Any]]) -> str:
        """Render a batch's tool names for a log line, bounded and escaped.

        Names are model-controlled strings with no length or character limit of
        their own, so they are truncated and ``repr``-escaped before reaching
        the log: a newline inside a name would otherwise split the record.
        Order is preserved and duplicates are kept - a batch calling the same
        tool three times is the fact worth reading. Non-mapping entries are
        rendered as a placeholder instead of raising, matching the
        ``isinstance(tool_call, dict)`` defense in ``_batch_carries_work_tool``
        and ``_strip_final_answer_bundled_with_work_tools``: this method must
        not be the one place in the strip path that crashes the run on a
        malformed batch it is only trying to log. That branch is
        defense-in-depth: ``_normalize_tool_calls`` currently emits only dict
        entries with non-empty names, so no production input reaches it today.
        """

        names = [
            repr(
                (
                    str(tool_call.get("name"))
                    if isinstance(tool_call, dict)
                    else f"<non-mapping tool_call: {type(tool_call).__name__}>"
                )[:STRIP_LOG_MAX_TOOL_NAME_CHARS]
            )
            for tool_call in tool_calls[:STRIP_LOG_MAX_TOOL_NAMES]
        ]
        overflow = len(tool_calls) - len(names)
        rendered = ", ".join(names)
        return f"{rendered} (+{overflow} more)" if overflow > 0 else rendered

    def _final_answer_text(self, args: Any) -> str:
        """Coerce a ``final_answer`` argument payload into its answer text.

        The single coercion for every site that reads an answer out of
        ``final_answer`` args - this check, the finalization it guards, and the
        streamer's candidate lookup - so they cannot disagree on what counts as
        an answer. ``None`` must become the empty string rather than
        ``str(None)``, which would yield the literal ``"None"`` and be sent to
        the user as the answer.
        """

        if not isinstance(args, dict):
            return ""
        answer = args.get("answer")
        if answer is None:
            return ""
        return str(answer)

    def _requires_full_tool_set_recovery(
        self,
        normalized: dict[str, Any],
        *,
        force_final_answer: bool,
    ) -> bool:
        protocol_error = get_tool_protocol_error(normalized.get("raw"))
        if (
            isinstance(protocol_error, dict)
            and protocol_error.get("code") == "unavailable_tool_call"
        ):
            return True
        if not force_final_answer:
            return False
        return any(
            isinstance(tool_call, dict) and tool_call.get("name") != "final_answer"
            for tool_call in normalized.get("tool_calls") or []
        )

    def _final_answer_tool_schema(self) -> dict[str, Any]:
        for schema in self._builtin_tool_schemas():
            function = schema.get("function")
            if isinstance(function, dict) and function.get("name") == "final_answer":
                return schema
        raise RuntimeError("final_answer control tool schema is unavailable")

    def _schema_tool_names(self, tool_schemas: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for schema in tool_schemas:
            function = schema.get("function")
            if isinstance(function, dict) and function.get("name"):
                names.append(str(function["name"]))
        return names

    def _tool_decision_groups_for_tools(self, tools: list[Any]) -> dict[str, str]:
        groups: dict[str, str] = {}
        for tool in tools:
            try:
                tool_name = self._tool_name(tool)
            except ValueError:
                continue
            groups[tool_name] = self._tool_decision_group(tool, tool_name)
        return groups

    def _tool_decision_group(self, tool: Any, tool_name: str) -> str:
        metadata = getattr(tool, "metadata", None)
        explicit_group = self._metadata_text(metadata, "decision_group") or (
            self._metadata_text(tool, "decision_group")
        )
        if explicit_group:
            return explicit_group

        category = getattr(metadata, "category", None) if metadata is not None else None
        category_value = getattr(category, "value", category)
        if category_value is not None:
            category_name = str(category_value).strip()
            if (
                category_name
                and category_name not in UNGROUPED_TOOL_DECISION_CATEGORIES
            ):
                return category_name
        return tool_name

    def _metadata_text(self, obj: Any, field_name: str) -> str:
        if obj is None:
            return ""
        value = getattr(obj, field_name, None)
        if not isinstance(value, str):
            return ""
        return value.strip()

    def _tool_decision_group_for_name(self, tool_name: str) -> str:
        return self._tool_decision_groups_by_name.get(tool_name, tool_name)

    def get_state(self) -> dict[str, Any]:
        """Return JSON-serializable ReAct state for checkpointing."""
        return {
            "reasoning_mode": self.reasoning_mode.value,
            "status": self.status,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "finalize_after_tool_result": self.finalize_after_tool_result,
            "tool_parallel_enabled": self.tool_parallel_enabled,
            "tool_max_concurrency": self.tool_max_concurrency,
            "repeated_tool_decision_after_consecutive_tool_calls": (
                self.repeated_tool_decision_after_consecutive_tool_calls
            ),
            "repeated_tool_decision_after_consecutive_work_tool_calls": (
                self.repeated_tool_decision_after_consecutive_work_tool_calls
            ),
            "force_final_answer_next": self.force_final_answer_next,
            "repeated_tool_decision": self.repeated_tool_decision,
            "waiting_for_user_request": self.waiting_for_user_request,
            "pending_tool_interaction_responses": (
                self.pending_tool_interaction_responses
            ),
            "task_text": self.task_text,
            "memory_input_text": self.memory_input_text,
            "last_response": self.last_response,
            "pending_tool_calls": self.pending_tool_calls,
            "pending_tool_call_content": self.pending_tool_call_content,
            "tool_ledger": {
                key: record.to_dict() for key, record in self.tool_ledger.items()
            },
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore ReAct state from a checkpoint payload."""
        self.reasoning_mode = ReActReasoningMode(
            state.get("reasoning_mode", ReActReasoningMode.TOOL_CALLING.value)
        )
        self.status = str(state.get("status", "idle"))
        self.current_iteration = int(state.get("current_iteration", 0))
        self.max_iterations = int(state.get("max_iterations", self.max_iterations))
        self.finalize_after_tool_result = bool(
            state.get("finalize_after_tool_result", self.finalize_after_tool_result)
        )
        if "tool_parallel_enabled" in state:
            self.tool_parallel_enabled = bool(state["tool_parallel_enabled"])
        if "tool_max_concurrency" in state:
            self.tool_max_concurrency = max(1, int(state["tool_max_concurrency"]))
        if "repeated_tool_decision_after_consecutive_tool_calls" in state:
            raw_threshold = state["repeated_tool_decision_after_consecutive_tool_calls"]
            self.repeated_tool_decision_after_consecutive_tool_calls = (
                int(raw_threshold) if raw_threshold is not None else None
            )
        elif "auto_reroute_after_consecutive_tool_calls" in state:
            raw_threshold = state["auto_reroute_after_consecutive_tool_calls"]
            self.repeated_tool_decision_after_consecutive_tool_calls = (
                int(raw_threshold) if raw_threshold is not None else None
            )

        if "repeated_tool_decision_after_consecutive_work_tool_calls" in state:
            raw_work_threshold = state[
                "repeated_tool_decision_after_consecutive_work_tool_calls"
            ]
            self.repeated_tool_decision_after_consecutive_work_tool_calls = (
                int(raw_work_threshold) if raw_work_threshold is not None else None
            )
        self.force_final_answer_next = bool(state.get("force_final_answer_next", False))
        repeated_tool_decision = state.get("repeated_tool_decision")
        self.repeated_tool_decision = (
            dict(repeated_tool_decision)
            if isinstance(repeated_tool_decision, dict)
            else None
        )
        waiting_request = state.get("waiting_for_user_request")
        self.waiting_for_user_request = (
            dict(waiting_request) if isinstance(waiting_request, dict) else None
        )
        pending_interaction_responses = state.get("pending_tool_interaction_responses")
        self.pending_tool_interaction_responses = [
            {
                "tool_name": str(item.get("tool_name") or ""),
                "tool_call_id": str(item.get("tool_call_id") or ""),
                "interaction_id": str(item.get("interaction_id") or ""),
                "response": str(item.get("response") or ""),
            }
            for item in (
                pending_interaction_responses
                if isinstance(pending_interaction_responses, list)
                else []
            )
            if isinstance(item, dict)
        ]
        stored_task_text = state.get("task_text")
        self.task_text = str(stored_task_text) if stored_task_text else None
        stored_memory_input = state.get("memory_input_text")
        if stored_memory_input:
            self.memory_input_text = str(stored_memory_input)
        self.last_response = state.get("last_response")
        self.pending_tool_calls = list(state.get("pending_tool_calls", []))
        self.pending_tool_call_content = dict(
            state.get("pending_tool_call_content", {})
        )
        self.tool_ledger = {
            key: ToolCallRecord.from_dict(value)
            for key, value in state.get("tool_ledger", {}).items()
        }

    async def _resume_waiting_for_user_if_needed(
        self,
        *,
        context: Any,
        runtime: PatternRuntime,
        tools: list[Any],
    ) -> dict[str, Any] | None:
        if self.status != "waiting_for_user" or not self.waiting_for_user_request:
            return None

        waiting_message_count = int(
            self.waiting_for_user_request.get("message_count", 0)
        )
        if len(getattr(context, "messages", [])) <= waiting_message_count:
            await runtime.checkpoint(
                "waiting_for_user",
                context=context,
                pattern=self,
                metadata={"waiting_for_user_request": self.waiting_for_user_request},
            )
            return {
                "success": False,
                "status": "waiting_for_user",
                "message": self.waiting_for_user_request.get("message", ""),
                "message_type": self.waiting_for_user_request.get(
                    "message_type", "question"
                ),
                "interactions": self.waiting_for_user_request.get("interactions"),
                "context": context,
                "clarification_draft": draft_from_waiting_request(
                    self.waiting_for_user_request,
                    execution_id=getattr(context, "execution_id", None),
                    step_id=None,
                ),
            }

        response = self._mark_latest_user_message_as_waiting_response(
            context=context,
            after_message_count=waiting_message_count,
        )
        self._queue_tool_interaction_responses(
            waiting_request=self.waiting_for_user_request,
            response=response or "",
            tools=tools,
        )
        waiting_task = self.waiting_for_user_request.get("task_text")
        if waiting_task and self.task_text is None:
            self.task_text = str(waiting_task)
        self.waiting_for_user_request = None
        self.status = "thinking"
        await runtime.checkpoint(
            "tool_interaction_response_received",
            context=context,
            pattern=self,
        )
        return None

    def _task_text(self, context: Any) -> str:
        if self.task_text:
            return self.task_text
        self.task_text = latest_user_text(context)
        return self.task_text

    def _memory_text(self, context: Any, *, execution_text: str) -> str:
        if self.memory_input_text:
            return self.memory_input_text
        self.memory_input_text = (
            context.current_user_request_text(prefer_display=True) or execution_text
        )
        return self.memory_input_text

    def seed_memory_input(self, memory_text: str) -> None:
        """Provide DAG-owned provenance without replacing restored state."""
        if self.memory_input_text is None:
            self.memory_input_text = str(memory_text or "") or None

    def _mark_latest_user_message_as_waiting_response(
        self,
        *,
        context: Any,
        after_message_count: int,
    ) -> str | None:
        messages = getattr(context, "messages", [])
        if not isinstance(messages, list):
            return None

        for index in range(len(messages) - 1, after_message_count - 1, -1):
            message = messages[index]
            if getattr(message, "role", None) != "user":
                continue
            metadata = dict(getattr(message, "metadata", {}) or {})
            if metadata.get("response_to_waiting_for_user"):
                return str(getattr(message, "content", "") or "")
            waiting_request = self.waiting_for_user_request or {}
            metadata["response_to_waiting_for_user"] = {
                "tool_name": waiting_request.get("tool_name"),
                "tool_call_id": waiting_request.get("tool_call_id"),
                "question": waiting_request.get("message", ""),
                "message_type": waiting_request.get("message_type", "question"),
                "interactions": waiting_request.get("interactions"),
                "requests": waiting_request.get("requests"),
            }
            messages[index] = replace(message, metadata=metadata)
            return str(getattr(message, "content", "") or "")
        return None

    def _queue_tool_interaction_responses(
        self,
        *,
        waiting_request: Any,
        response: str,
        tools: list[Any],
    ) -> None:
        """Queue replies only for tools that expose the optional resume callback."""

        if (
            not isinstance(waiting_request, dict)
            or waiting_request.get("kind") != "tool_waiting_for_user"
        ):
            return
        raw_requests = waiting_request.get("requests")
        requests = raw_requests if isinstance(raw_requests, list) else [waiting_request]
        for request in requests:
            if not isinstance(request, dict):
                continue
            tool_name = str(request.get("tool_name") or "")
            if not tool_name:
                continue
            try:
                tool = self._find_tool(tool_name, tools)
            except ValueError:
                continue
            if user_interaction_resume_callable(tool) is None:
                # Callback-less tools resume through the normal ReAct replan. The
                # user's answer remains in context with waiting-response metadata.
                continue
            self.pending_tool_interaction_responses.append(
                {
                    "tool_name": tool_name,
                    "tool_call_id": str(request.get("tool_call_id") or ""),
                    "interaction_id": str(
                        request.get("interaction_id")
                        or request.get("tool_call_id")
                        or ""
                    ),
                    "response": response,
                }
            )

    async def _deliver_pending_tool_interaction_responses(
        self,
        *,
        tools: list[Any],
        context: Any,
        runtime: PatternRuntime,
    ) -> None:
        """Deliver checkpointed replies to their exact suspended interactions."""

        while self.pending_tool_interaction_responses:
            pending = self.pending_tool_interaction_responses[0]
            tool_name = pending.get("tool_name", "")
            try:
                tool = self._find_tool(tool_name, tools)
            except ValueError:
                tool = None
            resume = (
                user_interaction_resume_callable(tool) if tool is not None else None
            )
            if resume is None:
                # Legacy checkpoints may contain callback delivery for a tool that
                # no longer exists or never implemented the optional capability.
                # The annotated user message is sufficient for the model to replan.
                self.pending_tool_interaction_responses.pop(0)
                await runtime.checkpoint(
                    "tool_interaction_response_skipped",
                    context=context,
                    pattern=self,
                    metadata={
                        "tool_name": tool_name,
                        "tool_call_id": pending.get("tool_call_id", ""),
                        "interaction_id": pending.get("interaction_id", ""),
                        "reason": "resume_callback_unavailable",
                    },
                )
                continue

            resumed = resume(
                interaction_id=pending.get("interaction_id", ""),
                response=pending.get("response", ""),
            )
            if inspect.isawaitable(resumed):
                await resumed

            # Keep the response retryable until the tool acknowledges delivery.
            self.pending_tool_interaction_responses.pop(0)
            await runtime.checkpoint(
                "tool_interaction_response_delivered",
                context=context,
                pattern=self,
                metadata={
                    "tool_name": tool_name,
                    "tool_call_id": pending.get("tool_call_id", ""),
                    "interaction_id": pending.get("interaction_id", ""),
                },
            )

    def _normalize_llm_response(self, response: Any) -> dict[str, Any]:
        if isinstance(response, str):
            content = unwrap_final_answer_content(response)
            return {
                "content": content,
                "tool_calls": [],
                "done": True,
                "raw": response,
            }

        if not isinstance(response, dict):
            text = unwrap_final_answer_content(str(response))
            return {"content": text, "tool_calls": [], "done": True, "raw": response}

        tool_calls = self._normalize_tool_calls(response.get("tool_calls", []))
        content_value: Any = response.get("content")
        if content_value is None:
            content_value = (
                response.get("answer")
                or response.get("output")
                or response.get("message")
            )
        if isinstance(content_value, str):
            content_value = unwrap_final_answer_content(content_value)

        done = response.get("done")
        if done is None:
            done = not tool_calls

        return {
            "content": content_value,
            "tool_calls": tool_calls,
            "raw_tool_calls": response.get("tool_calls", []),
            "done": bool(done),
            "raw": response,
        }

    def _provider_state_for_context(
        self, normalized_response: dict[str, Any]
    ) -> dict[str, Any]:
        raw = normalized_response.get("raw")
        marker_key = "_xagent_provider_state"
        if isinstance(raw, dict):
            raw_provider_state = raw.get(marker_key)
            if isinstance(raw_provider_state, dict):
                return {marker_key: raw_provider_state}
        if marker_key in normalized_response and isinstance(
            normalized_response[marker_key], dict
        ):
            return {marker_key: normalized_response[marker_key]}
        return {}

    def _normalize_tool_calls(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls):
            # Three provider shapes: a dict with a nested ``function`` payload, a
            # flat dict, and an object with a ``function`` attribute. Each yields
            # the same (id, name, arguments) triple.
            if isinstance(tool_call, dict):
                call_id = tool_call.get("id")
                function_payload = tool_call.get("function")
                if isinstance(function_payload, dict):
                    name = function_payload.get("name")
                    arguments = function_payload.get("arguments", {})
                else:
                    name = tool_call.get("name")
                    arguments = tool_call.get("args", tool_call.get("arguments", {}))
            else:
                function_payload = getattr(tool_call, "function", None)
                if function_payload is None:
                    continue
                call_id = getattr(tool_call, "id", None)
                name = getattr(function_payload, "name", None)
                arguments = getattr(function_payload, "arguments", {})

            normalized.append(
                {
                    "id": call_id or f"tool_call_{index}",
                    "name": name,
                    "args": self._coerce_arguments(arguments, tool_name=name),
                }
            )

        return [call for call in normalized if call.get("name")]

    def _remember_tool_call_content(
        self, tool_calls: list[dict[str, Any]], assistant_content: Any
    ) -> None:
        if not isinstance(assistant_content, str):
            return
        content = assistant_content.strip()
        if not content:
            return

        control_tool_names = self._control_tool_names()
        for tool_call in tool_calls:
            if tool_call.get("name") in control_tool_names:
                continue
            tool_call_id = str(tool_call.get("id") or "")
            if tool_call_id:
                self.pending_tool_call_content[tool_call_id] = content
            return

    def _coerce_arguments(
        self, arguments: Any, *, tool_name: str | None = None
    ) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                if not arguments.strip():
                    # The happy path for a parameterless tool on providers that
                    # send `""` instead of `"{}"` (#1501) - not a malformation.
                    logger.debug(
                        "ReAct tool call %s returned blank arguments; "
                        "treating as no arguments.",
                        tool_name or "<unknown>",
                    )
                else:
                    logger.warning(
                        "ReAct tool call %s returned malformed JSON arguments "
                        "(%d chars); dropping them for control tools, passing "
                        "anything else through as `input`.",
                        tool_name or "<unknown>",
                        len(arguments),
                    )
                return self._fallback_arguments(tool_name, arguments)
            if isinstance(parsed, dict):
                return parsed
            logger.warning(
                "ReAct tool call %s returned non-object JSON arguments (%s).",
                tool_name or "<unknown>",
                type(parsed).__name__,
            )
            return self._fallback_arguments(tool_name, parsed)
        return {}

    def _fallback_arguments(
        self, tool_name: str | None, payload: Any
    ) -> dict[str, Any]:
        """Wrap unusable tool arguments, or drop them for control tools.

        Work tools tolerate an opaque ``input`` passthrough, so they keep a
        non-blank payload. Control tools drop it, because ``input`` is never a
        field any of them declares: carrying it forward only disguises the loss
        as a populated arguments object.

        For ``final_answer`` dropping it is also load-bearing. That tool owns the
        run's only user-visible exit, so a payload smuggled through as ``input``
        would silently strip ``answer`` and finalize the run with nothing to
        show; empty args instead let
        ``_response_requires_tool_protocol_retry`` reject the call and spend the
        pattern's one repair retry on it. ``send_message`` and
        ``ask_user_question`` have no such guard - they degrade to an empty
        message either way, exactly as they did before this branch existed.
        """

        if tool_name in self._control_tool_names():
            return {}
        if isinstance(payload, str) and not payload.strip():
            # A blank payload carries nothing to preserve, so there is no
            # opaque value for the `input` passthrough to forward.
            return {}
        return {"input": payload}

    def _build_tool_schema(self, tool: Any) -> dict[str, Any]:
        name = self._tool_name(tool)
        description = self._compact_tool_description(self._tool_description(tool))
        schema = self._compact_tool_json_schema(self._tool_json_schema(tool))
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema,
            },
        }

    def _compact_tool_description(self, description: str) -> str:
        """Trim redundant whitespace while preserving instructional structure."""
        compacted_lines: list[str] = []
        for line in description.splitlines():
            compacted = " ".join(line.split())
            if compacted:
                compacted_lines.append(compacted)
            elif compacted_lines and compacted_lines[-1]:
                compacted_lines.append("")
        while compacted_lines and not compacted_lines[-1]:
            compacted_lines.pop()
        return "\n".join(compacted_lines)

    def _compact_tool_json_schema(
        self, value: Any, *, named_schema_mapping: bool = False
    ) -> Any:
        """Remove presentation-only Pydantic metadata from provider schemas."""
        if isinstance(value, list):
            return [self._compact_tool_json_schema(item) for item in value]
        if not isinstance(value, dict):
            return value

        compacted: dict[str, Any] = {}
        for key, item in value.items():
            if key == "title" and not named_schema_mapping:
                continue
            if named_schema_mapping:
                compacted[key] = self._compact_tool_json_schema(item)
            elif key == "description" and isinstance(item, str):
                compacted[key] = self._compact_tool_description(item)
            else:
                compacted[key] = self._compact_tool_json_schema(
                    item,
                    named_schema_mapping=key
                    in {"properties", "patternProperties", "$defs", "definitions"},
                )
        return compacted

    def _builtin_tool_schemas(
        self, *, can_lookup_output_files: bool = False
    ) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "description": (
                        "Finish the current ReAct step and send the final answer to "
                        "the user. Set outcome=completed only when every requested "
                        "action and verification succeeded. Use outcome=partial when "
                        "some useful work succeeded but the request remains "
                        "unfinished, or outcome=blocked when no further progress is "
                        "possible without user input or an external state change. "
                        "Never mark an answer completed when it admits work is "
                        "unfinished. Call this tool alone: never place it in the "
                        "same response as any other tool call, and do not call "
                        "additional tools after it. An answer written in the same "
                        "response as a tool call cannot describe that tool's "
                        "result. Set response_language to the target output "
                        "language for this answer. "
                        f"{final_answer_language_rule()}"
                    ),
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "response_language": {
                                "type": "string",
                                "description": REACT_RESPONSE_LANGUAGE_DESCRIPTION,
                            },
                            "answer": {
                                "type": "string",
                                "description": (
                                    "Complete user-facing answer. It must be "
                                    "non-empty and must match response_language. "
                                    f"{final_deliverable_file_reference_instructions(can_lookup=can_lookup_output_files, include_heading=False)} "
                                    f"{final_answer_language_rule()}"
                                ),
                            },
                            "outcome": {
                                "type": "string",
                                "enum": ["completed", "partial", "blocked"],
                                "description": (
                                    "Semantic outcome of the request. completed "
                                    "means the whole request was carried out and "
                                    "verified; partial means only part succeeded; "
                                    "blocked means progress requires the user or an "
                                    "external state change."
                                ),
                            },
                        },
                        "required": ["response_language", "answer", "outcome"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "Send a message to the user, optionally waiting for a response.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                            "message_type": {
                                "type": "string",
                                "enum": [
                                    "info",
                                    "question",
                                    "confirmation",
                                    "progress",
                                    "warning",
                                ],
                            },
                            "expect_response": {"type": "boolean"},
                            "visible": {"type": "boolean"},
                        },
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_user_question",
                    "description": (
                        "Ask the user for structured input and pause execution until "
                        "the user responds. Use this only when execution cannot "
                        "continue without missing user-provided information, such "
                        "as a required file, URL, account, target object, permission, "
                        "a fact-carrying value (one that asserts a real-world fact) "
                        "for a tool argument that the user has not provided, "
                        "or a choice between mutually exclusive actions with "
                        "different side effects. Do not use it to confirm execution "
                        "strategy, whether to search, whether to use memory, whether "
                        "to apply formatting preferences, or whether to proceed with "
                        "a sufficiently specified task; decide those yourself. A task "
                        "is not sufficiently specified if carrying it out would "
                        "require inventing a fact-carrying argument value the user "
                        "has not provided."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                            "interactions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": list(INTERACTION_TYPES),
                                        },
                                        "field": {"type": "string"},
                                        "label": {"type": "string"},
                                        "options": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "label": {"type": "string"},
                                                    "value": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "action_type": {
                                                        "type": "string",
                                                        "enum": [
                                                            "upload",
                                                            "input_url",
                                                            "none",
                                                        ],
                                                    },
                                                },
                                                "required": ["label", "value"],
                                            },
                                        },
                                        "placeholder": {"type": "string"},
                                        "multiline": {"type": "boolean"},
                                        "accept": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "multiple": {"type": "boolean"},
                                    },
                                    "required": ["type", "field", "label"],
                                },
                            },
                        },
                        "required": ["message", "interactions"],
                    },
                },
            },
        ]
        if not self.user_interaction_enabled:
            return [
                schema
                for schema in schemas
                if schema.get("function", {}).get("name")
                not in USER_INTERACTION_CONTROL_TOOL_NAMES
            ]
        return schemas

    def _tool_schemas_with_builtin_controls(
        self,
        tools: list[Any],
    ) -> list[dict[str, Any]]:
        control_tool_names = self._control_tool_names()
        external_tools = [
            self._build_tool_schema(tool)
            for tool in tools
            if self._tool_name(tool) not in control_tool_names
        ]
        can_lookup_output_files = any(
            schema.get("function", {}).get("name") == WORKSPACE_OUTPUT_FILES_TOOL_NAME
            for schema in external_tools
        )
        return [
            *external_tools,
            *self._builtin_tool_schemas(
                can_lookup_output_files=can_lookup_output_files
            ),
        ]

    def _control_tool_names(self) -> set[str]:
        return set(CONTROL_TOOL_NAMES)

    async def _handle_control_tool(
        self,
        tool_call: dict[str, Any],
        context: Any,
        llm: Any,
        runtime: PatternRuntime,
    ) -> dict[str, Any] | None:
        name = tool_call["name"]
        args = tool_call.get("args", {})

        if (
            not self.user_interaction_enabled
            and name in USER_INTERACTION_CONTROL_TOOL_NAMES
        ):
            error = f"Control tool '{name}' is disabled for this execution."
            logger.warning(
                "ReAct rejected disabled user-interaction control tool. "
                "tool=%s tool_call_id=%s",
                name,
                tool_call.get("id"),
            )
            failure = {"success": False, "status": "error", "error": error}
            self._record_tool_call(
                tool_call,
                status="failed",
                result=failure,
                error=error,
            )
            context.add_tool_result(
                tool_name=name,
                result=failure,
                tool_call_id=tool_call.get("id"),
            )
            remaining = [
                pending
                for pending in self.pending_tool_calls
                if pending is not tool_call
            ]
            self.pending_tool_calls = [tool_call]
            self._cancel_tool_calls(
                remaining,
                context,
                reason=f"Discarded because control tool '{name}' is disabled.",
            )
            self.status = "thinking"
            self.force_final_answer_next = True
            return None

        if name == "final_answer":
            answer = self._final_answer_text(args)
            if not answer.strip():
                self._reject_empty_final_answer(tool_call, context)
                return None
            outcome = self._final_answer_outcome(args.get("outcome"))
            self._record_tool_call(
                tool_call,
                status="completed",
                result={"answer": answer, "outcome": outcome},
            )
            self.status = "completed"
            context.add_tool_result(
                tool_name=name,
                result={"answer": answer, "outcome": outcome},
                tool_call_id=tool_call.get("id"),
            )
            # Unconditional: the empty-answer rejection above is the only way
            # into finalization, so an answer here always has text.
            context.add_assistant_message(answer)
            return await self._finalize_outcome(
                context=context,
                runtime=runtime,
                response=answer,
                outcome=outcome,
            )

        if name == "send_message":
            message = str(args.get("message", ""))
            expect_response = bool(args.get("expect_response", False))
            message_type = str(args.get("message_type", "info"))
            visible = bool(args.get("visible", True))
            outbound_message = await runtime.send_message(
                message=message,
                message_type=message_type,
                expect_response=expect_response,
                visible=visible,
            )
            self._record_tool_call(
                tool_call,
                status="completed",
                result={
                    "message": message,
                    "expect_response": expect_response,
                    "visible": visible,
                },
            )
            if expect_response:
                self.status = "waiting_for_user"
                context.add_tool_result(
                    tool_name=name,
                    result={
                        "status": "waiting_for_user",
                        "message": message,
                        "message_type": message_type,
                    },
                    tool_call_id=tool_call.get("id"),
                )
                self.waiting_for_user_request = {
                    "event_id": outbound_message["event_id"],
                    "tool_call_id": tool_call.get("id"),
                    "tool_name": name,
                    "message": message,
                    "message_type": message_type,
                    "task_text": self.task_text,
                    "message_count": len(getattr(context, "messages", [])),
                }
                return {
                    "success": False,
                    "status": self.status,
                    "message": message,
                    "message_type": message_type,
                    "context": context,
                    "clarification_draft": draft_from_waiting_request(
                        self.waiting_for_user_request,
                        execution_id=getattr(context, "execution_id", None),
                        step_id=None,
                    ),
                }
            context.add_tool_result(
                tool_name=name,
                result={"message": message, "status": "sent"},
                tool_call_id=tool_call.get("id"),
            )
            return {
                "success": True,
                "status": "message_sent",
                "output": message,
                "response": message,
                "message": message,
            }

        if name == "ask_user_question":
            message = str(args.get("message", ""))
            interactions = _normalize_ask_user_interactions(
                args.get("interactions", [])
            )
            # Same dedup shape _pause_for_tool_results runs on the multi-tool
            # path (append _2, _3 to a repeated base, first occupant keeps
            # its own name), the only legitimate difference being scope:
            # used_fields here spans only this one call's own interactions,
            # never a sibling tool's, because this branch has no sibling
            # tool to collide with.
            used_fields: set[str] = set()
            deduplicated_interactions: list[dict[str, Any]] = []
            for interaction in interactions:
                item = dict(interaction)
                base_field = str(item.get("field") or "response")
                field = base_field
                suffix = 2
                while field in used_fields:
                    field = f"{base_field}_{suffix}"
                    suffix += 1
                item["field"] = field
                used_fields.add(field)
                deduplicated_interactions.append(item)
            interactions = deduplicated_interactions
            outbound_message = await runtime.send_message(
                message=message,
                message_type="question",
                expect_response=True,
                visible=True,
                metadata={"interactions": interactions},
            )
            self._record_tool_call(
                tool_call,
                status="completed",
                result={
                    "message": message,
                    "expect_response": True,
                    "interactions": interactions,
                },
            )
            self.status = "waiting_for_user"
            context.add_tool_result(
                tool_name=name,
                result={
                    "status": "waiting_for_user",
                    "message": message,
                    "message_type": "question",
                    "interactions": interactions,
                },
                tool_call_id=tool_call.get("id"),
            )
            self.waiting_for_user_request = {
                "event_id": outbound_message["event_id"],
                "tool_call_id": tool_call.get("id"),
                "tool_name": name,
                "message": message,
                "message_type": "question",
                "interactions": interactions,
                "task_text": self.task_text,
                "message_count": len(getattr(context, "messages", [])),
            }
            return {
                "success": False,
                "status": self.status,
                "message": message,
                "message_type": "question",
                "interactions": interactions,
                "context": context,
                "clarification_draft": draft_from_waiting_request(
                    self.waiting_for_user_request,
                    execution_id=getattr(context, "execution_id", None),
                    step_id=None,
                ),
            }

        return None

    def _reject_empty_final_answer(
        self, tool_call: dict[str, Any], context: Any
    ) -> None:
        """Refuse to finalize on an empty ``final_answer`` and re-request one.

        For a batch whose only calls are control tools,
        ``_response_requires_tool_protocol_retry`` still catches this on the
        turn the model produces it, discarding the whole response before it is
        recorded. For a batch that also carries a work tool, the fresh path
        strips the ``final_answer`` call regardless of whether its answer is
        empty, so the work tool runs and this handler never sees that batch
        either. This guard is left covering the one path that reaches the
        handler without passing through response normalization at all — a
        checkpoint resume that restores ``pending_tool_calls`` verbatim.

        Returning ``None`` keeps the run in the loop; ``force_final_answer_next``
        makes the next turn re-request ``final_answer``, and if that turn is empty
        too the normalization guard fails the run instead of looping.

        Two deliberate differences from the fresh-turn path.

        Sibling calls still pending are discarded regardless of order, because
        by the time a resume reaches this point the assistant envelope for the
        batch is already recorded in history (see
        ``_ensure_pending_tool_call_envelope``), so an undiscarded sibling
        would be left queued with no result of its own. This diverges from the
        fresh path, which now runs the work tool instead of discarding it: the
        fresh path can still drop ``final_answer`` before anything is
        recorded, while a resumed batch can only cancel calls that are already
        committed to history.

        And the next turn is forced to ``final_answer`` alone, where the fresh
        path restores the full tool set. That bounds the run: a second empty
        answer ends it as ``invalid_tool_protocol`` rather than looping. It does
        not prevent tool re-execution outright - if the forced turn calls a
        non-``final_answer`` tool, ``_requires_full_tool_set_recovery`` restores
        the full set and clears the flag, so a discarded work tool can be
        re-invoked on that turn.
        """

        logger.warning(
            "ReAct refusing to finalize on final_answer with no answer text; "
            "re-requesting a final answer. tool_call_id=%s arg_keys=%s",
            tool_call.get("id"),
            sorted(self._tool_call_args_dict(tool_call), key=str),
        )
        error = (
            "final_answer was called with an empty answer, so the user received "
            "nothing. Call final_answer again with the complete user-facing "
            "response in its answer field."
        )
        # Carry the classification keys ``tool_result_succeeded`` reads, so this
        # failure is not read back as a success, and record the same shape in the
        # ledger as the cancelled siblings below.
        failure = {"success": False, "status": "error", "error": error}
        self._record_tool_call(tool_call, status="failed", result=failure, error=error)
        context.add_tool_result(
            tool_name="final_answer",
            result=failure,
            tool_call_id=tool_call.get("id"),
        )
        # Leave only the rejected call for the caller to pop, and close out the
        # rest of the batch so no unexecuted call is left without a result.
        # Selected by identity rather than position: the caller happens to pass
        # the head of the queue today, but nothing here depends on that.
        remaining = [
            pending for pending in self.pending_tool_calls if pending is not tool_call
        ]
        self.pending_tool_calls = [tool_call]
        self._cancel_tool_calls(
            remaining,
            context,
            reason=(
                "Discarded because final_answer was called with an empty answer; "
                "the agent will produce a final answer instead."
            ),
        )
        self.status = "thinking"
        self.force_final_answer_next = True

    def _tool_is_concurrency_safe(self, name: str, tools: list[Any]) -> bool:
        """Whether ``name`` may run concurrently with other safe tools.

        The metadata flag is also the tool author's idempotency declaration:
        an interrupted call can be retried on resume when cancellation races
        with an external side effect. Unknown tools and tools without the flag
        are conservatively kept serial.
        """
        try:
            tool = self._find_tool(name, tools)
        except ValueError:
            return False
        metadata = getattr(tool, "metadata", None)
        return bool(getattr(metadata, "concurrency_safe", False))

    def _next_segment(
        self, pending: list[dict[str, Any]], tools: list[Any]
    ) -> tuple[list[dict[str, Any]], str]:
        """Slice the next consecutive segment off the front of ``pending``.

        Returns ``(segment, kind)`` where ``kind`` is one of:
        - ``"control"``: a single control tool (final_answer / send_message /
          ask_user_question), which always owns its segment;
        - ``"serial"``: a single tool executed on its own (a non-safe tool, any
          tool when the parallel flag is off, or a lone safe tool);
        - ``"concurrent"``: two or more consecutive concurrency-safe tools.
        """
        control_tool_names = self._control_tool_names()
        first = pending[0]
        if first["name"] in control_tool_names:
            return [first], "control"
        if not self.tool_parallel_enabled:
            return [first], "serial"
        if not self._tool_is_concurrency_safe(first["name"], tools):
            return [first], "serial"

        segment: list[dict[str, Any]] = []
        for tool_call in pending:
            name = tool_call["name"]
            if name in control_tool_names:
                break
            if not self._tool_is_concurrency_safe(name, tools):
                break
            segment.append(tool_call)
            # Cap a batch at the concurrency width. The loop re-checks the
            # interrupt before each segment, so a mid-turn interrupt is honored
            # after at most one wave instead of after every safe call the model
            # emitted. Tradeoff: a run longer than the width is split into
            # successive gather() barriers rather than one continuously
            # pipelined batch, so under skewed tool latencies a straggler in one
            # wave can briefly idle the next wave's slots. Acceptable for v1;
            # decouple batch size from the semaphore if profiling shows it costs.
            if len(segment) >= self.tool_max_concurrency:
                break
        # A lone safe tool degrades to serial: no gather/Semaphore overhead and
        # byte-for-byte identical behavior to the current serial path.
        if len(segment) == 1:
            return segment, "serial"
        return segment, "concurrent"

    def _backfill_result(
        self, tool_call: dict[str, Any], result: Any, context: Any
    ) -> None:
        """Record one tool result into the context and drop its cached content.

        Shared by the serial and concurrent paths so message-history ordering
        is produced the same way regardless of execution mode.
        """
        context.add_tool_result(
            tool_name=tool_call["name"],
            result=result,
            tool_call_id=tool_call.get("id"),
        )
        self._forget_tool_call_content(tool_call)

    def _discard_pending_tool_plan_after_pause(self, context: Any) -> None:
        """Close unexecuted calls so resume always starts with a fresh LLM plan."""

        discarded_calls = self.pending_tool_calls
        self.pending_tool_calls = []
        self.repeated_tool_decision = None
        self.force_final_answer_next = False
        self._cancel_tool_calls(
            discarded_calls,
            context,
            reason=(
                "Discarded because an earlier tool requires user input; "
                "the agent will replan after the user responds."
            ),
        )

    def _cancel_tool_calls(
        self, tool_calls: list[dict[str, Any]], context: Any, *, reason: str
    ) -> None:
        """Close calls that will never run, one result each.

        Preserves I2 (one result per ``tool_call_id``) for abandoned calls, so a
        discarded batch cannot leave a tool call dangling without a result.
        """

        for tool_call in tool_calls:
            result = {"success": False, "status": "cancelled", "error": reason}
            self._backfill_result(tool_call, result, context)
            self._record_tool_call(
                tool_call,
                status="cancelled",
                result=result,
            )

    async def _pause_for_tool_results(
        self,
        *,
        waiting_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
        context: Any,
        runtime: PatternRuntime,
    ) -> dict[str, Any]:
        """Publish tool-originated questions and checkpoint the suspended run."""

        if not self.user_interaction_enabled:
            tool_names = sorted(
                {
                    str(tool_call.get("name") or "unknown")
                    for tool_call, _ in waiting_pairs
                }
            )
            error = (
                "Tool-requested user interaction is disabled for this execution: "
                + ", ".join(tool_names)
            )
            logger.warning("ReAct rejected tool-requested user interaction: %s", error)
            self.status = "failed"
            self.waiting_for_user_request = None
            return {
                "success": False,
                "status": "failed",
                "error": error,
                "context": context,
            }

        requests: list[dict[str, Any]] = []
        interactions: list[dict[str, Any]] = []
        used_fields: set[str] = set()
        for tool_call, result in waiting_pairs:
            request_interactions = _normalize_ask_user_interactions(
                result.get("interactions", [])
            )
            deduplicated_request_interactions: list[dict[str, Any]] = []
            for interaction in request_interactions:
                item = dict(interaction)
                base_field = str(item.get("field") or "response")
                field = base_field
                suffix = 2
                while field in used_fields:
                    field = f"{base_field}_{suffix}"
                    suffix += 1
                item["field"] = field
                used_fields.add(field)
                interactions.append(item)
                deduplicated_request_interactions.append(item)

            requests.append(
                {
                    "tool_call_id": str(tool_call.get("id") or ""),
                    "tool_name": str(tool_call.get("name") or ""),
                    "interaction_id": str(
                        result.get("interaction_id") or tool_call.get("id") or ""
                    ),
                    "message": str(
                        result.get("message")
                        or result.get("error")
                        or "This tool requires user input before it can continue."
                    ),
                    "message_type": str(result.get("message_type") or "question"),
                    "interactions": deduplicated_request_interactions,
                }
            )

        if len(requests) == 1:
            message = requests[0]["message"]
            message_type = requests[0]["message_type"]
        else:
            message = "Multiple tools need your input before the task can continue:\n\n"
            message += "\n".join(
                f"- {request['tool_name']}: {request['message']}"
                for request in requests
            )
            message_type = "question"

        outbound_message = await runtime.send_message(
            message=message,
            message_type=message_type,
            expect_response=True,
            visible=True,
            metadata={"interactions": interactions},
        )
        self.status = "waiting_for_user"
        self.waiting_for_user_request = {
            "event_id": outbound_message["event_id"],
            "kind": "tool_waiting_for_user",
            "requests": requests,
            "message": message,
            "message_type": message_type,
            "interactions": interactions,
            "task_text": self.task_text,
            "message_count": len(getattr(context, "messages", [])),
        }
        waiting_result = {
            "success": False,
            "status": "waiting_for_user",
            "message": message,
            "message_type": message_type,
            "interactions": interactions,
            "context": context,
            "clarification_draft": draft_from_waiting_request(
                self.waiting_for_user_request,
                execution_id=getattr(context, "execution_id", None),
                step_id=None,
            ),
        }
        await runtime.checkpoint(
            "waiting_for_user",
            context=context,
            pattern=self,
            metadata={
                "waiting_for_user_request": self.waiting_for_user_request,
            },
        )
        return waiting_result

    async def _run_concurrent_batch(
        self,
        batch: list[dict[str, Any]],
        tools: list[Any],
        runtime: PatternRuntime,
        context: Any,
    ) -> list[Any]:
        """Run a segment of concurrency-safe tool calls and back-fill in order.

        Tools run under a Semaphore via ``asyncio.gather`` (results stay aligned
        to ``batch`` regardless of completion order, satisfying I1). Results are
        back-filled serially in the main coroutine after all tools finish, so
        message-history order is deterministic (I1) and every call gets exactly
        one result (I2). ``_execute_tool_safely`` already turns tool exceptions
        into error dicts, so any real exception captured by
        ``return_exceptions=True`` is an infra-callback/unexpected failure and is
        re-raised to halt the turn exactly like the serial path (I5).
        """
        semaphore = asyncio.Semaphore(self.tool_max_concurrency)

        async def _guarded(tool_call: dict[str, Any]) -> Any:
            async with semaphore:
                return await self._execute_tool_safely(tool_call, tools, runtime)

        raw_results = await asyncio.gather(
            *(_guarded(tool_call) for tool_call in batch),
            return_exceptions=True,
        )

        # Tool-level failures are already converted to error dicts inside
        # _execute_tool_safely. Reconcile the successful calls before propagating
        # any real exception so a mixed infra-failure/interrupt batch cannot
        # replay work that already completed. Keep exceptional calls pending by
        # their object identity in this batch: provider-supplied ids are not
        # guaranteed unique, so id-based filtering could accidentally drop a
        # different call.
        infra_error = next(
            (
                result
                for result in raw_results
                if isinstance(result, BaseException)
                and not isinstance(result, ToolCallInterrupted)
            ),
            None,
        )
        interrupted = next(
            (
                result
                for result in raw_results
                if isinstance(result, ToolCallInterrupted)
            ),
            None,
        )
        if infra_error is not None or interrupted is not None:
            completed_call_objects: set[int] = set()
            for tool_call, result in zip(batch, raw_results):
                if isinstance(result, BaseException):
                    continue
                self._backfill_result(tool_call, result, context)
                completed_call_objects.add(id(tool_call))
            self._reorder_ledger_for_batch(batch)
            self.pending_tool_calls = [
                tool_call
                for tool_call in self.pending_tool_calls
                if id(tool_call) not in completed_call_objects
            ]
            # Preserve the serial path's infra-error priority while keeping the
            # ledger, context, and pending queue consistent for diagnostics or
            # an explicit retry.
            if infra_error is not None:
                raise infra_error
            assert interrupted is not None
            raise interrupted

        for tool_call, result in zip(batch, raw_results):
            self._backfill_result(tool_call, result, context)
        self._reorder_ledger_for_batch(batch)
        return list(raw_results)

    def _reorder_ledger_for_batch(self, batch: list[dict[str, Any]]) -> None:
        """Reassert input order for this batch's ledger records (I3).

        Concurrent execution can interleave ``_record_tool_call`` writes, so the
        batch's records may land out of order in the insertion-ordered ledger.
        ``_consecutive_*_count`` walk the ledger in reverse insertion order, so
        we pop this batch's records and re-insert them at the tail in the
        original tool-call order. Records keep their latest (final) state; only
        their relative position is restored.
        """
        ids = [str(tool_call.get("id") or "") for tool_call in batch]
        records = {
            tool_id: self.tool_ledger.pop(tool_id)
            for tool_id in ids
            if tool_id in self.tool_ledger
        }
        for tool_id in ids:
            record = records.get(tool_id)
            if record is not None:
                self.tool_ledger[tool_id] = record

    def _disabled_control_tool_index(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        user_interaction_enabled: bool,
    ) -> int | None:
        """Index of the first disabled user-interaction control tool, if any.

        Shared by ``_execute_pending_tool_calls`` (which cancels every call
        ahead of that index) and ``_close_streamed_answer`` (which must treat
        a batch the same way whether or not its first call has already run).
        Both callers pass their own ``user_interaction_enabled`` state rather
        than this reading ``self`` directly, so the two call sites cannot
        silently drift onto different predicates.
        """

        if user_interaction_enabled:
            return None
        return next(
            (
                index
                for index, pending in enumerate(tool_calls)
                if pending.get("name") in USER_INTERACTION_CONTROL_TOOL_NAMES
            ),
            None,
        )

    async def _execute_pending_tool_calls(
        self,
        *,
        context: Any,
        tools: list[Any],
        llm: Any,
        runtime: PatternRuntime,
    ) -> dict[str, Any] | None:
        if not self.user_interaction_enabled:
            disabled_index = self._disabled_control_tool_index(
                self.pending_tool_calls,
                user_interaction_enabled=self.user_interaction_enabled,
            )
            if disabled_index is not None:
                preceding = self.pending_tool_calls[:disabled_index]
                disabled_name = self.pending_tool_calls[disabled_index].get("name")
                self.pending_tool_calls = self.pending_tool_calls[disabled_index:]
                self._cancel_tool_calls(
                    preceding,
                    context,
                    reason=(
                        "Discarded because the response also called disabled "
                        f"control tool '{disabled_name}'."
                    ),
                )
        successful_tool_result = False
        while self.pending_tool_calls:
            interrupted = await self._interrupt_if_requested(
                runtime=runtime,
                context=context,
                label="before_tool",
            )
            if interrupted is not None:
                return interrupted

            segment, kind = self._next_segment(self.pending_tool_calls, tools)

            if kind == "control":
                tool_call = segment[0]
                control_result = await self._handle_control_tool(
                    tool_call,
                    context,
                    llm,
                    runtime,
                )
                self.pending_tool_calls = self.pending_tool_calls[1:]
                self._forget_tool_call_content(tool_call)
                await runtime.checkpoint(
                    str(control_result.get("status", "control_tool"))
                    if control_result is not None
                    else "control_tool",
                    context=context,
                    pattern=self,
                    metadata={"tool_call": tool_call},
                )
                if control_result is not None:
                    if control_result.get("status") == "completed":
                        self.pending_tool_calls = []
                        return control_result
                    if control_result.get("status") == "waiting_for_user":
                        self._discard_pending_tool_plan_after_pause(context)
                        return control_result
                continue

            if kind == "serial":
                tool_call = segment[0]
                await runtime.checkpoint(
                    "before_tool",
                    context=context,
                    pattern=self,
                    metadata={"tool_call": tool_call},
                )
                result = await self._execute_tool_safely(tool_call, tools, runtime)
                self._backfill_result(tool_call, result, context)
                self.pending_tool_calls = self.pending_tool_calls[1:]
                if tool_result_waits_for_user(result):
                    assert isinstance(result, dict)
                    self._discard_pending_tool_plan_after_pause(context)
                    return await self._pause_for_tool_results(
                        waiting_pairs=[(tool_call, result)],
                        context=context,
                        runtime=runtime,
                    )
                await runtime.checkpoint(
                    "after_tool",
                    context=context,
                    pattern=self,
                    metadata={"tool_call": tool_call},
                )
                results = [result]
            else:  # kind == "concurrent"
                await runtime.checkpoint(
                    "before_tool_batch",
                    context=context,
                    pattern=self,
                    metadata={"tool_calls": segment},
                )
                results = await self._run_concurrent_batch(
                    segment, tools, runtime, context
                )
                self.pending_tool_calls = self.pending_tool_calls[len(segment) :]
                waiting_pairs = [
                    (waiting_call, waiting_result)
                    for waiting_call, waiting_result in zip(segment, results)
                    if tool_result_waits_for_user(waiting_result)
                    and isinstance(waiting_result, dict)
                ]
                if waiting_pairs:
                    self._discard_pending_tool_plan_after_pause(context)
                    return await self._pause_for_tool_results(
                        waiting_pairs=waiting_pairs,
                        context=context,
                        runtime=runtime,
                    )
                await runtime.checkpoint(
                    "after_tool_batch",
                    context=context,
                    pattern=self,
                    metadata={"tool_calls": segment},
                )
                # Catch interrupts that arrive after the batch completed but
                # before the next segment begins. Interrupts during the batch
                # cancel its runtime-owned tool tasks immediately.
                interrupted = await self._interrupt_if_requested(
                    runtime=runtime,
                    context=context,
                    label="after_tool_batch",
                )
                if interrupted is not None:
                    return interrupted

            # Evaluate repeated-tool-decision once per segment, on its last call.
            requested_decision = await self._request_repeated_tool_decision_if_needed(
                tool_call=segment[-1],
                context=context,
                runtime=runtime,
            )
            if any(self._tool_result_success(result) for result in results):
                successful_tool_result = True
            if requested_decision:
                successful_tool_result = False

        if (
            self.finalize_after_tool_result
            and successful_tool_result
            and not self.repeated_tool_decision
        ):
            self.force_final_answer_next = True
        return None

    async def _request_repeated_tool_decision_if_needed(
        self,
        *,
        tool_call: dict[str, Any],
        context: Any,
        runtime: PatternRuntime,
    ) -> bool:
        if self.repeated_tool_decision is not None:
            return False

        metadata = self._repeated_tool_call_metadata(tool_call)
        if metadata is None:
            return False

        self.repeated_tool_decision = metadata
        await runtime.checkpoint(
            REPEATED_TOOL_DECISION_REQUESTED_STATUS,
            context=context,
            pattern=self,
            metadata=metadata,
        )
        return True

    async def _run_repeated_tool_decision(
        self,
        *,
        context: Any,
        llm: Any,
        runtime: PatternRuntime,
    ) -> dict[str, Any] | None:
        metadata = dict(self.repeated_tool_decision or {})
        if not metadata:
            return None

        messages = self._messages_for_repeated_tool_decision(context, metadata)
        call_llm = await prepare_llm_for_context(
            llm=llm,
            messages=messages,
            context=context,
        )
        decision_tools = [self._react_decision_tool_schema()]
        llm_metadata = {
            "phase": REPEATED_TOOL_DECISION_REQUESTED_STATUS,
            **resolved_llm_metadata(call_llm),
            **metadata,
        }
        await runtime.on_llm_start(
            context=context,
            messages=messages,
            tools=decision_tools,
            metadata=llm_metadata,
        )
        try:
            response = await runtime.run_streaming_llm_call(
                call_llm,
                messages=messages,
                tools=decision_tools,
                tool_choice="required",
                thinking={"type": "disabled", "enable": False},
            )
        except LLMCallInterrupted:
            raise
        except Exception as exc:
            await runtime.on_llm_error(
                context=context,
                error=exc,
                metadata=llm_metadata,
            )
            raise

        await runtime.on_llm_end(
            context=context,
            response=response,
            metadata=llm_metadata,
        )
        self.last_response = response
        self.repeated_tool_decision = None

        decision = self._parse_react_decision(response)
        if decision is None:
            await runtime.checkpoint(
                "repeated_tool_decision_invalid",
                context=context,
                pattern=self,
                metadata={"response": response, **metadata},
            )
            return None

        if decision["action"] == REACT_DECISION_FINAL_ANSWER:
            self.force_final_answer_next = True
            await runtime.checkpoint(
                "repeated_tool_decision_final_requested",
                context=context,
                pattern=self,
                metadata={**metadata, "decision": decision},
            )
            context.add_system_message(
                "Repeated tool decision completion guidance:\n"
                "The repeated-tool decision selected final_answer, so the next "
                "normal ReAct step must produce the final user-facing answer from "
                "the accumulated conversation and tool results. Do not call more "
                "tools in that final step. Do not send a progress update or promise "
                "future work as the final answer; if the accumulated results are "
                "insufficient or show the task is incomplete, say that directly.",
                metadata={
                    "source": "repeated_tool_decision",
                    **metadata,
                },
            )
            return None

        await runtime.checkpoint(
            "repeated_tool_decision_continue",
            context=context,
            pattern=self,
            metadata={**metadata, "decision": decision},
        )
        missing_verification = decision.get("missing_verification", "").strip()
        if missing_verification:
            context.add_system_message(
                "Repeated tool decision continuation guidance:\n"
                "The previous repeated-tool decision chose to continue. The next "
                "work-tool call should retrieve or verify this specific missing "
                f"information: {missing_verification}",
                metadata={
                    "source": "repeated_tool_decision",
                    "missing_verification": missing_verification,
                },
            )
        return None

    def _repeated_tool_call_metadata(
        self,
        tool_call: dict[str, Any],
    ) -> dict[str, Any] | None:
        tool_name = str(tool_call.get("name") or "")
        if not tool_name or tool_name in self._control_tool_names():
            return None

        same_tool_threshold = self.repeated_tool_decision_after_consecutive_tool_calls
        if same_tool_threshold is not None and same_tool_threshold > 0:
            tool_group = self._tool_decision_group_for_name(tool_name)
            same_tool_count = self._consecutive_successful_tool_group_count(tool_group)
            if same_tool_count >= same_tool_threshold:
                return {
                    "trigger": "same_tool_successes",
                    "tool_name": tool_group,
                    "latest_tool_name": tool_name,
                    "consecutive_tool_calls": same_tool_count,
                    "threshold": same_tool_threshold,
                }

        work_tool_threshold = (
            self.repeated_tool_decision_after_consecutive_work_tool_calls
        )
        if work_tool_threshold is None or work_tool_threshold <= 0:
            return None

        work_tool_count = self._consecutive_work_tool_call_count()
        if work_tool_count < work_tool_threshold:
            return None
        return {
            "trigger": "work_tool_attempts",
            "tool_name": tool_name,
            "latest_tool_name": tool_name,
            "consecutive_tool_calls": work_tool_count,
            "threshold": work_tool_threshold,
        }

    def _messages_for_repeated_tool_decision(
        self,
        context: Any,
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        messages = list(context.get_messages_for_llm())
        tool_name = str(metadata.get("tool_name") or "the tool")
        count = int(metadata.get("consecutive_tool_calls") or 0)
        trigger = metadata.get("trigger")
        if trigger == "work_tool_attempts":
            latest_tool_name = str(metadata.get("latest_tool_name") or tool_name)
            count_text = (
                f"{count} consecutive work-tool calls without a final answer"
                if count > 0
                else "repeated work-tool calls without a final answer"
            )
            call_context = (
                f"{count_text}; the latest work tool was {latest_tool_name}. "
                "Some attempts may have failed; count them as work already spent."
            )
        else:
            latest_tool_name = str(metadata.get("latest_tool_name") or tool_name)
            count_text = (
                f"{count} consecutive successful calls"
                if count > 0
                else "repeated successful calls"
            )
            if latest_tool_name != tool_name:
                call_context = (
                    f"{count_text} in the {tool_name} tool group; "
                    f"the latest tool was {latest_tool_name}."
                )
            else:
                call_context = f"{count_text} to {tool_name}."
        current_request = truncate_prompt_preview(
            latest_user_text(context) or "",
            limit=400,
        )
        request_anchor = (
            "Latest user request text:\n"
            f"{current_request or '(unavailable)'}\n\n"
            "Use this as the controlling request when deciding whether the "
            "accumulated tool results have completed the user's requested work."
        )
        prompt = (
            f"You must call {REACT_DECISION_TOOL_NAME} exactly once. Decide whether "
            "the current ReAct run should finish or make another work-tool call. "
            f"{request_anchor} "
            f"You have just made {call_context} action must be "
            f"{REACT_DECISION_FINAL_ANSWER} or {REACT_DECISION_TOOL_CALL}. Choose "
            f"{REACT_DECISION_FINAL_ANSWER} when the conversation and accumulated "
            "tool results are sufficient to answer the latest user request. A "
            "final answer means the latest user request is already completed; if "
            "the next user-facing answer would describe a future tool action or "
            "say work is still in progress, choose tool_call instead. Choose "
            f"{REACT_DECISION_TOOL_CALL} only when a specific missing fact, source, "
            "verification, or work step remains; the next normal ReAct turn will "
            "choose and call the actual work tool from the full available tool set. "
            "In the tool arguments, write reason and missing_verification before "
            "action so you assess completion before committing to the decision. "
            "Use an empty missing_verification only when no work remains. "
            "Audit every explicit deliverable in the latest request against a "
            "successful tool result before choosing final_answer. Model-only or "
            "private observations are evidence for reasoning, not user-visible "
            "deliverables. A requested computer screenshot is complete only when "
            "the computer result marks it as a user-visible artifact and returns "
            "its file reference. For other file or media tools, a successful "
            "artifact, file_ref, or markdown_link is sufficient evidence. "
            "Do not call work tools in this decision. Do not put user-facing final "
            "answer text in this decision; the next normal ReAct step will produce "
            "the final answer if you choose final_answer. Treat the completed-call "
            "count in this instruction as authoritative; do not count the user's "
            "requested number as already completed. If the latest user request "
            "explicitly requires more completed work-tool calls or results than "
            f"the current context contains, choose {REACT_DECISION_TOOL_CALL}."
        )
        return [*messages, {"role": "user", "content": prompt}]

    def _react_decision_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": REACT_DECISION_TOOL_NAME,
                "description": (
                    "Decide whether ReAct should finish in the next normal final "
                    "answer step or continue to another work-tool call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": (
                                "Assess whether every requested work result already "
                                "exists. Write this assessment before choosing action."
                            ),
                        },
                        "missing_verification": {
                            "type": "string",
                            "description": (
                                "When action is tool_call, the specific missing "
                                "fact, source, verification, or work step that "
                                "requires another tool call. Empty only when no work "
                                "remains."
                            ),
                        },
                        "action": {
                            "type": "string",
                            "enum": [
                                REACT_DECISION_FINAL_ANSWER,
                                REACT_DECISION_TOOL_CALL,
                            ],
                            "description": (
                                "Choose final_answer only when the preceding "
                                "assessment found no missing work; otherwise choose "
                                "tool_call."
                            ),
                        },
                    },
                    "required": ["reason", "missing_verification", "action"],
                    "additionalProperties": False,
                },
            },
        }

    def _parse_react_decision(self, response: Any) -> dict[str, str] | None:
        normalized = self._normalize_llm_response(response)
        for tool_call in normalized.get("tool_calls", []):
            if tool_call.get("name") != REACT_DECISION_TOOL_NAME:
                continue
            args = tool_call.get("args")
            if not isinstance(args, dict):
                return None
            action = str(args.get("action") or "").strip()
            if action not in {
                REACT_DECISION_FINAL_ANSWER,
                REACT_DECISION_TOOL_CALL,
            }:
                return None
            missing_verification = str(args.get("missing_verification") or "")
            if action == REACT_DECISION_FINAL_ANSWER and missing_verification.strip():
                action = REACT_DECISION_TOOL_CALL
            return {
                "action": action,
                "reason": str(args.get("reason") or ""),
                "missing_verification": missing_verification,
            }
        return None

    def _consecutive_successful_tool_group_count(self, tool_group: str) -> int:
        count = 0
        control_tool_names = self._control_tool_names()
        for record in reversed(list(self.tool_ledger.values())):
            if record.tool_name in control_tool_names:
                continue
            if self._tool_decision_group_for_name(record.tool_name) != tool_group:
                break
            if record.status != "completed" or not self._tool_result_success(
                record.result
            ):
                break
            count += 1
        return count

    def _consecutive_work_tool_call_count(self) -> int:
        count = 0
        control_tool_names = self._control_tool_names()
        for record in reversed(list(self.tool_ledger.values())):
            if record.tool_name in control_tool_names:
                continue
            if record.status not in {"completed", "failed"}:
                continue
            count += 1
        return count

    async def _finalize_success(
        self,
        *,
        context: Any,
        runtime: PatternRuntime,
        response: Any,
    ) -> dict[str, Any]:
        return await self._finalize_outcome(
            context=context,
            runtime=runtime,
            response=response,
            outcome="completed",
        )

    @staticmethod
    def _final_answer_outcome(value: Any) -> str:
        outcome = str(value or "completed").strip().lower()
        if outcome not in {"completed", "partial", "blocked"}:
            return "blocked"
        return outcome

    async def _finalize_outcome(
        self,
        *,
        context: Any,
        runtime: PatternRuntime,
        response: Any,
        outcome: str,
    ) -> dict[str, Any]:
        """Finish the run while preserving its semantic completion outcome."""

        self.pending_tool_calls = []
        self.waiting_for_user_request = None
        self.pending_tool_interaction_responses = []
        self.force_final_answer_next = False
        self.status = "completed"
        await runtime.checkpoint("final", context=context, pattern=self)
        result = PatternResult(
            success=True,
            output=response,
            metadata={
                "response": response,
                "status": self.status,
                "completion_outcome": outcome,
            },
        ).to_dict()
        return result

    def _ensure_pending_tool_call_envelope(self, context: Any) -> None:
        if not self.pending_tool_calls:
            return
        messages = [
            message
            for message in getattr(context, "messages", [])
            if not getattr(message, "hidden", False)
        ]
        index = len(messages) - 1
        while index >= 0 and messages[index].role == "tool":
            index -= 1
        if index >= 0 and messages[index].role == "assistant":
            tool_calls = messages[index].tool_calls or []
            existing_ids = {
                str(tool_call.get("id"))
                for tool_call in tool_calls
                if isinstance(tool_call, dict) and tool_call.get("id")
            }
            pending_ids = {
                str(tool_call.get("id"))
                for tool_call in self.pending_tool_calls
                if tool_call.get("id")
            }
            if pending_ids and pending_ids.issubset(existing_ids):
                return
            if not pending_ids and len(tool_calls) >= len(self.pending_tool_calls):
                return

        context.add_assistant_message(
            "",
            tool_calls=[
                self._tool_call_for_context(tool_call)
                for tool_call in self.pending_tool_calls
            ],
        )

    def _tool_call_for_context(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": tool_call.get("id"),
            "type": "function",
            "function": {
                "name": tool_call.get("name"),
                "arguments": json.dumps(
                    tool_call.get("args", {}),
                    ensure_ascii=False,
                    default=str,
                ),
            },
        }

    def _tool_result_success(self, result: Any) -> bool:
        return tool_result_succeeded(result)

    def _latest_tool_result_success(self, context: Any) -> bool:
        for message in reversed(getattr(context, "messages", [])):
            if getattr(message, "role", None) != "tool":
                continue
            metadata = getattr(message, "metadata", {}) or {}
            return self._tool_result_success(metadata.get("raw_result"))
        return False

    def _completion_evidence_instruction(self, context: Any) -> str:
        for message in reversed(getattr(context, "messages", [])):
            metadata = getattr(message, "metadata", {}) or {}
            evidence = metadata.get("dag_completion_evidence")
            if not isinstance(evidence, str):
                continue
            evidence = evidence.strip()
            if evidence:
                return (
                    "Step completion evidence: "
                    f"{evidence} "
                    "When the latest tool result or response satisfies this evidence "
                    "and the termination condition, call final_answer for this step. "
                    "Do not repeat the same work for a nicer variant unless the result "
                    "failed or the user explicitly requested a revision."
                )
        return ""

    async def _interrupt_if_requested(
        self,
        *,
        runtime: PatternRuntime,
        context: Any,
        label: str,
    ) -> dict[str, Any] | None:
        if not await runtime.should_interrupt():
            return None

        self.status = "interrupted"
        await runtime.checkpoint(
            "interrupted",
            context=context,
            pattern=self,
            metadata={"safe_point": label, "reason": runtime.interrupt_reason},
        )
        return PatternResult(
            success=False,
            error="ReActPattern interrupted.",
            metadata={
                "status": self.status,
                "interrupt_reason": runtime.interrupt_reason,
            },
        ).to_dict()

    async def _execute_tool_safely(
        self,
        tool_call: dict[str, Any],
        tools: list[Any],
        runtime: PatternRuntime,
    ) -> Any:
        # Stamp a stable id on the *original* dict before the _with_* transforms
        # (which may return a copy). _record_tool_call only computes a fallback
        # key locally; without writing it back, the key drifts between the
        # running/completed writes as the ledger grows, and the still-id-less
        # dict that _backfill_result / _reorder_ledger_for_batch read desyncs
        # from the ledger (I2/I3). No await runs before the first record below,
        # so concurrent batch members get distinct fallback ids.
        if not tool_call.get("id"):
            tool_call["id"] = f"tool_call_{len(self.tool_ledger)}"
        tool_call = self._with_tool_call_content(tool_call)
        tool_call = self._with_runtime_step(tool_call, runtime)
        tool_call = self._with_runtime_turn_id(tool_call, runtime)
        tool_call = self._with_trace_safe_tool_args(tool_call, tools)
        self._record_tool_call(tool_call, status="running")
        recorded_terminal = False
        try:
            await runtime.on_tool_start(tool_call=tool_call)
            try:
                result = await runtime.run_tool_call(
                    lambda: self._execute_tool(tool_call, tools)
                )
            except ToolCallInterrupted as exc:
                await runtime.on_tool_cancelled(
                    tool_call=tool_call,
                    reason=str(exc),
                )
                self._record_tool_call(
                    tool_call,
                    status="interrupted",
                    error=str(exc),
                )
                recorded_terminal = True
                raise
            except Exception as exc:  # noqa: BLE001
                error_result = {
                    "success": False,
                    "error": str(exc),
                    "tool_name": tool_call["name"],
                }
                await runtime.on_tool_error(
                    tool_call=tool_call, error=exc, result=error_result
                )
                self._record_tool_call(
                    tool_call,
                    status="failed",
                    result=error_result,
                    error=str(exc),
                )
                recorded_terminal = True
                return error_result

            if tool_result_waits_for_user(result):
                self._record_tool_call(
                    tool_call,
                    status="waiting_for_user",
                    result=result,
                )
                recorded_terminal = True
                await runtime.on_tool_end(tool_call=tool_call, result=result)
                return result

            if not self._tool_result_success(result):
                error_message = str(
                    result.get("error") or result.get("message") or result
                )
                await runtime.on_tool_error(
                    tool_call=tool_call,
                    error=RuntimeError(error_message),
                    result=result,
                )
                self._record_tool_call(
                    tool_call,
                    status="failed",
                    result=result,
                    error=error_message,
                )
                recorded_terminal = True
                return result

            self._record_tool_call(tool_call, status="completed", result=result)
            recorded_terminal = True
            await runtime.on_tool_end(tool_call=tool_call, result=result)
            return result
        finally:
            # An infra callback (on_tool_start) can raise before any terminal
            # record is written. Never leave the ledger stuck at "running": the
            # consecutive-count walks skip non-terminal records, which would
            # undercount repeated-tool-decision triggers. The exception still
            # propagates (serial path) or is captured by the batch gather.
            if not recorded_terminal:
                self._record_tool_call(
                    tool_call,
                    status="failed",
                    error="tool execution aborted before completion",
                )

    def _with_trace_safe_tool_args(
        self, tool_call: dict[str, Any], tools: list[Any]
    ) -> dict[str, Any]:
        try:
            tool = self._find_tool(tool_call["name"], tools)
        except Exception:  # noqa: BLE001
            return tool_call

        sanitizer = getattr(tool, "sanitize_tool_args_for_trace", None)
        if not callable(sanitizer):
            return tool_call

        raw_args = tool_call.get("args")
        if raw_args is None:
            args: dict[str, Any] = {}
            original_args: dict[str, Any] = {}
        elif isinstance(raw_args, dict):
            args = copy.deepcopy(raw_args)
            original_args = copy.deepcopy(raw_args)
        else:
            return tool_call
        sanitized = sanitizer(args)
        if not isinstance(sanitized, dict) or sanitized == original_args:
            return tool_call
        return {**tool_call, "args": sanitized}

    def _with_runtime_step(
        self, tool_call: dict[str, Any], runtime: PatternRuntime
    ) -> dict[str, Any]:
        if tool_call.get("step_id") or tool_call.get("dag_step_id"):
            return tool_call

        step_id = getattr(runtime, "active_react_step_id", None)
        if not step_id:
            return tool_call

        return {
            **tool_call,
            "step_id": str(step_id),
            "dag_step_id": str(step_id),
        }

    def _with_runtime_turn_id(
        self, tool_call: dict[str, Any], runtime: PatternRuntime
    ) -> dict[str, Any]:
        """Stamp the current turn's id onto ``tool_call`` for on_tool_* to read.

        ``runtime.active_turn_id`` carries the durable turn_id: on the real
        ``PatternRuntime`` it's set in ``on_pattern_start`` from the
        triggering user message's metadata; on ``_DAGStepRuntime`` (dag.py)
        it's a property resolving the same value from the DAG's root
        context. Either way this lets ``PatternRuntime.on_tool_start`` /
        ``on_tool_end`` / ``on_tool_error`` attribute the resulting trace
        event to its turn by join, not by timestamp adjacency.
        """
        if tool_call.get("turn_id"):
            return tool_call

        turn_id = getattr(runtime, "active_turn_id", None)
        if not turn_id:
            return tool_call

        return {**tool_call, "turn_id": str(turn_id)}

    def _with_tool_call_content(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        tool_call_id = str(tool_call.get("id") or "")
        content = self.pending_tool_call_content.get(tool_call_id)
        if not content:
            return tool_call
        return {
            **tool_call,
            "assistant_content": content,
        }

    def _forget_tool_call_content(self, tool_call: dict[str, Any]) -> None:
        tool_call_id = str(tool_call.get("id") or "")
        if tool_call_id:
            self.pending_tool_call_content.pop(tool_call_id, None)

    def _record_tool_call(
        self,
        tool_call: dict[str, Any],
        *,
        status: str,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        tool_call_id = str(tool_call.get("id") or f"tool_call_{len(self.tool_ledger)}")
        args = self._tool_call_args_dict(tool_call)
        args_hash = self._args_hash(args)
        self.tool_ledger[tool_call_id] = ToolCallRecord(
            tool_call_id=tool_call_id,
            tool_name=str(tool_call["name"]),
            args=args,
            args_hash=args_hash,
            status=status,
            result=result,
            error=error,
        )

    def _args_hash(self, args: dict[str, Any]) -> str:
        try:
            canonical = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            canonical = str(args)
        # Digest instead of the raw JSON: the hash is persisted in every
        # ledger record, so large args would otherwise be stored twice.
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _tool_call_args_dict(
        self, tool_call: dict[str, Any], *, require_mapping: bool = False
    ) -> dict[str, Any]:
        raw_args = tool_call.get("args")
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return dict(raw_args)
        if require_mapping:
            raise ValueError("Tool call args must be a JSON object.")
        return {}

    def _tool_name(self, tool: Any) -> str:
        metadata = getattr(tool, "metadata", None)
        if metadata is not None and getattr(metadata, "name", None):
            return str(metadata.name)
        if getattr(tool, "name", None):
            return str(tool.name)
        if getattr(tool, "__name__", None):
            return str(tool.__name__)
        raise ValueError(f"Tool {tool!r} is missing a name.")

    def _tool_description(self, tool: Any) -> str:
        metadata = getattr(tool, "metadata", None)
        if metadata is not None and getattr(metadata, "description", None):
            return str(metadata.description)
        return (
            str(getattr(tool, "description", ""))
            or str(getattr(tool, "__doc__", "")).strip()
            or self._tool_name(tool)
        )

    def _tool_json_schema(self, tool: Any) -> dict[str, Any]:
        args_type = getattr(tool, "args_type", None)
        if callable(args_type):
            schema_type = args_type()
            if hasattr(schema_type, "model_json_schema"):
                return cast(dict[str, Any], schema_type.model_json_schema())
            if hasattr(schema_type, "schema"):
                return cast(dict[str, Any], schema_type.schema())
        for schema_attr in ("args_schema", "tool_call_schema"):
            schema_type = getattr(tool, schema_attr, None)
            if schema_type is None:
                continue
            if hasattr(schema_type, "model_json_schema"):
                return cast(dict[str, Any], schema_type.model_json_schema())
            if hasattr(schema_type, "schema"):
                return cast(dict[str, Any], schema_type.schema())
        args = getattr(tool, "args", None)
        if isinstance(args, dict) and args:
            return {"type": "object", "properties": args}
        if inspect.isfunction(tool):
            return self._signature_json_schema(tool)
        return {"type": "object", "properties": {}}

    def _signature_json_schema(self, fn: Any) -> dict[str, Any]:
        signature = inspect.signature(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, parameter in signature.parameters.items():
            if name in {"self", "cls"}:
                continue
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            properties[name] = self._annotation_json_schema(parameter.annotation)
            if parameter.default is inspect.Parameter.empty:
                required.append(name)

        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def _annotation_json_schema(self, annotation: Any) -> dict[str, Any]:
        if annotation is inspect.Parameter.empty:
            return {}
        if annotation is str or annotation == "str":
            return {"type": "string"}
        if annotation is int or annotation == "int":
            return {"type": "integer"}
        if annotation is float or annotation == "float":
            return {"type": "number"}
        if annotation is bool or annotation == "bool":
            return {"type": "boolean"}
        if annotation is dict or annotation == "dict":
            return {"type": "object"}
        if annotation is list or annotation == "list":
            return {"type": "array"}
        return {}

    async def _execute_tool(self, tool_call: dict[str, Any], tools: list[Any]) -> Any:
        tool = self._find_tool(tool_call["name"], tools)
        args = self._tool_args_for_execution(tool_call, tool)

        execute = getattr(tool, "execute", None)
        if callable(execute):
            return await self._invoke_callable(execute, **args)

        run_json_async = getattr(tool, "run_json_async", None)
        if callable(run_json_async):
            return await run_json_async(args)

        ainvoke = getattr(tool, "ainvoke", None)
        if callable(ainvoke):
            return await ainvoke(args)

        call = getattr(tool, "__call__", None)
        if callable(call):
            return await self._invoke_callable(call, **args)

        raise ValueError(
            f"Tool {tool_call['name']} does not expose a supported executor."
        )

    def _tool_args_for_execution(
        self, tool_call: dict[str, Any], tool: Any
    ) -> dict[str, Any]:
        args = self._tool_call_args_dict(tool_call, require_mapping=True)
        tool_name = self._tool_name(tool)
        if not (tool_name.startswith("browser_") or tool_name == "computer"):
            return args

        step_id = tool_call.get("dag_step_id") or tool_call.get("step_id")
        if step_id and not args.get("session_id"):
            args.setdefault("_xagent_step_id", str(step_id))
        return args

    def _find_tool(self, name: str, tools: list[Any]) -> Any:
        for tool in tools:
            if self._tool_name(tool) == name:
                return tool
        # Semantic Agent tool names may change when an Agent is renamed. Keep
        # historical ``agent_<id>`` calls and prior semantic names executable
        # without exposing duplicate aliases to the model's tool schema.
        for tool in tools:
            matches_name = getattr(tool, "matches_name", None)
            if callable(matches_name) and matches_name(name):
                return tool
        raise ValueError(f"Tool not found: {name}")

    async def _invoke_callable(self, fn: Any, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(fn):
            return await fn(**kwargs)
        result = await asyncio.to_thread(fn, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
