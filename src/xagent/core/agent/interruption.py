"""Why a run stopped short of a terminal answer.

A run that ends in an exception or an unsuccessful result is either finished
for good (the task itself cannot succeed: a context window it does not fit,
bad credentials, a bad request) or merely interrupted (the database went away,
the LLM provider kept refusing, the process died, or the model's output was
unusable even after in-run repair). Only the second kind is worth resuming
from its checkpoint, and this module is where core names it.

Unusable model output counts as an interruption because resuming does not
replay the failed provider request: the run continues from its last
checkpoint and samples again, which can succeed where an identical retry
would not. The settling layer must bound how often that happens (below).

Classification is advisory metadata. Nothing in core acts on it; callers that
settle a run decide what an interruption reason means for the task.

Where the reason travels: the runner annotates only failures a pattern
contains (an exception it catches, or an unsuccessful result) under
``interruption_reason``, and ``AgentExecutionAdapter`` forwards that key as a
top-level field of its result, which is what ``AgentService`` callers receive.

Obligations of the layer that settles a run (none of them is enforced here):

- Failures the runner deliberately lets escape, notably
  ``CheckpointPersistenceError``, are classified there, by calling
  :func:`classify_run_failure` on the escaping exception; that is the
  producer of ``PERSISTENCE_FAILURE`` for checkpoint writes.
- Resumes must be capped there. ``MODEL_OUTPUT_INVALID`` in particular can
  recur on every run of a deterministically bad prompt, so without a cap
  (a limit on resumes that make no progress, and an absolute limit per run)
  treating it as resumable would loop forever.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from enum import Enum
from typing import Any

from ..model.chat.error import matches_context_length_error, retry_on
from ..model.chat.exceptions import (
    LLMEmptyContentError,
    LLMInvalidResponseError,
    LLMToolProtocolError,
)
from ..retry.policy import is_capacity_error
from .checkpoint import CheckpointPersistenceError

INTERRUPTION_REASON_KEY = "interruption_reason"


class InterruptionReason(str, Enum):
    """Why a run was interrupted rather than finished.

    Only the system-caused reasons (``LEASE_EXPIRED``, ``PERSISTENCE_FAILURE``,
    ``LLM_UNAVAILABLE``, ``MODEL_OUTPUT_INVALID`` and the reserved
    ``SHUTDOWN``) describe a run that may be resumed without the user. The
    rest are recorded so a paused task can say why it is paused.
    """

    # Process crash, kill -9, lost heartbeat or failed settlement, recovered
    # once the lease TTL expired.
    LEASE_EXPIRED = "lease_expired"
    # A checkpoint or execution-event write failed, or the database
    # connection itself did.
    PERSISTENCE_FAILURE = "persistence_failure"
    # A transient provider failure outlived the LLM call's own retry budget.
    LLM_UNAVAILABLE = "llm_unavailable"
    # The model kept producing an unusable response (invalid tool protocol,
    # empty or unparsable content) after in-run repair.
    MODEL_OUTPUT_INVALID = "model_output_invalid"
    # Reserved for graceful shutdown hand-off; nothing produces it yet.
    SHUTDOWN = "shutdown"
    # The user paused the task, or it was PAUSE_REQUESTED when the run died.
    USER_PAUSE = "user_pause"
    # The outcome of delivering input to the run is unknown.
    INPUT_OUTCOME_UNKNOWN = "input_outcome_unknown"
    # A tool attempt has no confirmed outcome; terminal, never auto-resumed.
    UNKNOWN_TOOL_EFFECT = "unknown_tool_effect"
    # No usable checkpoint to resume from; terminal.
    NOT_RECOVERABLE = "not_recoverable"


_MODEL_OUTPUT_INVALID_ERRORS = (
    LLMToolProtocolError,
    LLMEmptyContentError,
    LLMInvalidResponseError,
)


def _cause_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield ``error`` and its explicit ``__cause__`` chain, cycle-safe.

    This walk deliberately does not follow ``__context__``, for the reason
    ``xagent.core.retry.policy._causes`` gives: it is set implicitly for any
    exception raised while another was being handled, so it can link an
    unrelated cleanup failure and misclassify the run.

    Composed rules can still look further: ``retry_on`` (rule 5) starts with
    ``is_context_length_error``, which follows ``__context__`` too. That can
    only turn ``LLM_UNAVAILABLE`` into a terminal ``None``, never the reverse.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__


def _is_database_unavailable(error: BaseException) -> bool:
    """Whether ``error`` is a SQLAlchemy connectivity failure.

    Imported lazily so core keeps working, and simply never matches, where
    SQLAlchemy is not importable. ``OperationalError`` and ``InterfaceError``
    are the DBAPI's connection-level classes; any other ``DBAPIError`` counts
    only when SQLAlchemy flagged the connection as invalidated. The pool's
    ``TimeoutError`` (checkout timeout) is a connectivity failure too.
    """
    try:
        from sqlalchemy import exc as sa_exc
    except ImportError:  # pragma: no cover - sqlalchemy is a core dependency
        return False
    if isinstance(error, (sa_exc.OperationalError, sa_exc.InterfaceError)):
        return True
    if isinstance(error, sa_exc.DBAPIError) and error.connection_invalidated:
        return True
    return isinstance(error, sa_exc.TimeoutError)


def is_database_unavailable(error: BaseException) -> bool:
    """Whether ``error`` or its explicit ``__cause__`` chain is a database
    connectivity failure -- the kind that a later attempt can get past.

    Settling code uses it to tell a transient read failure (worth deferring)
    from a fault that would reproduce on every attempt.
    """
    return any(_is_database_unavailable(link) for link in _cause_chain(error))


def _is_llm_unavailable(error: BaseException) -> bool:
    if isinstance(error, Exception) and retry_on(error):
        return True
    return is_capacity_error(error)


def _is_provider_refusal(error: BaseException) -> bool:
    """Whether the provider refused the account rather than the moment.

    An exhausted quota or balance and a rejected credential often arrive as
    a 429 or a wrapped ``RuntimeError`` that ``retry_on`` accepts, but asking
    again later will not help. The provider-call boundary already names
    them; this reuses its classification instead of a second set of markers.
    Imported lazily so this module does not load every provider adapter.
    Its chain walk also follows ``__context__``, which can only make a
    failure terminal here, never resumable.
    """
    from ..model.chat.basic.call_boundary import (
        CREDENTIAL_REJECTED,
        PROVIDER_QUOTA,
        classify_provider_failure,
    )

    return classify_provider_failure(error) in {PROVIDER_QUOTA, CREDENTIAL_REJECTED}


def _guarded_provider_failure(
    chain: list[BaseException],
) -> tuple[bool, InterruptionReason | None]:
    """Classify a ``ProviderCallError`` in ``chain`` by its fixed code.

    A model wrapped by ``guard_llm_calls`` replaces every provider exception
    with a ``ProviderCallError`` raised outside the handler, so it carries no
    cause for ``retry_on`` and no provider text for the refusal markers --
    the same blindness ``runtime._budget_cannot_help`` works around. Its
    ``code`` and ``transient`` flag are exact and text-free, so they decide
    instead. Returns ``(found, reason)``; imported lazily like
    :func:`_is_provider_refusal`.
    """
    from ..model.chat.basic.call_boundary import (
        CREDENTIAL_REJECTED,
        PROVIDER_QUOTA,
        PROVIDER_UNAVAILABLE,
        RATE_LIMITED,
        TIMEOUT,
        ProviderCallError,
    )

    for error in chain:
        if not isinstance(error, ProviderCallError):
            continue
        if error.code in {PROVIDER_QUOTA, CREDENTIAL_REJECTED}:
            return True, None
        if error.transient or error.code in {
            RATE_LIMITED,
            TIMEOUT,
            PROVIDER_UNAVAILABLE,
        }:
            return True, InterruptionReason.LLM_UNAVAILABLE
        return True, None
    return False, None


def classify_run_failure(exc: BaseException) -> InterruptionReason | None:
    """Classify an exception that ended a run, or ``None`` if it is terminal.

    Each rule is tried against the whole ``__cause__`` chain before the next
    rule runs, so precedence is by rule rather than by chain position:

    1. A context-window failure anywhere in the chain is intrinsic to the
       task and terminal, whatever wraps it.
    2. Checkpoint/execution-event persistence failures.
    3. SQLAlchemy connectivity failures.
    4. Unusable model output, including the ``LLMToolProtocolError`` codes
       ``retry_on`` refuses (``malformed_tool_arguments``,
       ``unavailable_tool_call``): that veto is about replaying the identical
       provider request, whereas a resumed run re-generates from its
       checkpoint. Checked before rule 5 because these classes
       subclass ``LLMRetryableError``, which ``retry_on`` accepts, and a
       wrapper's ``retry_on`` looks one ``__cause__`` deep.
    5. A guarded model's ``ProviderCallError``, by its fixed code: an
       exhausted quota or a rejected credential is terminal; a rate limit,
       timeout, unavailable provider or a ``transient`` flag is
       ``LLM_UNAVAILABLE``; any other code is terminal.
    6. Transient provider failures (``retry_on``) and capacity refusals,
       unless the provider-call boundary classifies the failure as an
       exhausted quota or a rejected credential, which are terminal. That
       text-based check is for unguarded models, whose raw provider errors
       still carry the provider's code and message.

    Anything else (configuration, authentication, bad requests, tool errors,
    iteration limits) is terminal and returns ``None``.
    """
    chain = list(_cause_chain(exc))
    if any(matches_context_length_error(error) for error in chain):
        return None
    if any(isinstance(error, CheckpointPersistenceError) for error in chain):
        return InterruptionReason.PERSISTENCE_FAILURE
    if any(_is_database_unavailable(error) for error in chain):
        return InterruptionReason.PERSISTENCE_FAILURE
    if any(isinstance(error, _MODEL_OUTPUT_INVALID_ERRORS) for error in chain):
        return InterruptionReason.MODEL_OUTPUT_INVALID
    found, reason = _guarded_provider_failure(chain)
    if found:
        return reason
    if any(_is_llm_unavailable(error) for error in chain):
        if _is_provider_refusal(exc):
            return None
        return InterruptionReason.LLM_UNAVAILABLE
    return None


def classify_run_result(result: Any) -> InterruptionReason | None:
    """Classify an unsuccessful run result, or ``None`` if it is terminal.

    A ReAct run that gave up on the model's tool protocol returns
    ``status="invalid_tool_protocol"`` instead of raising. Otherwise the
    runner's own ``interruption_reason`` is trusted when it names a known
    reason.
    """
    if not isinstance(result, Mapping):
        return None
    # Deliberately the whole status, not only the repair-exhausted empty
    # answer: it also covers provider protocol errors, mixed control calls and
    # a non-final_answer tool on a forced turn (see ``_child_never_answered``
    # in tools/adapters/vibe/agent_tool.py). All of them are model output a
    # fresh sample can fix, and callers cap how often they resume for it.
    if result.get("status") == "invalid_tool_protocol":
        return InterruptionReason.MODEL_OUTPUT_INVALID
    reason = result.get(INTERRUPTION_REASON_KEY)
    if isinstance(reason, InterruptionReason):
        return reason
    if isinstance(reason, str):
        try:
            return InterruptionReason(reason)
        except ValueError:
            return None
    return None


def interruption_reason_value(reason: InterruptionReason | None) -> str | None:
    """The JSON-safe form stored on run results."""
    return reason.value if reason is not None else None
