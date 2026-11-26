"""Commands that hand a user message to a run that is already executing.

The command executor runs in the task process that holds the task lease. It
only reads in-memory state and returns in milliseconds, because commands of
one task execute one at a time and a stop request must not queue behind it.
The message waits in the running execution's mailbox, in memory only; the
delivery ledger is touched later, by ``DeliveryLedgerAdmission``, when the
execution reaches a safe point and asks to claim the message.

Who the sender is and whether they may still spend quota is decided by the
registered ``LiveInputHost``; this module never inspects identities.

A host may also implement the claim hooks of ``LiveInputClaimHooks``: it then
stages rows of its own in the claim transaction, removes them in the
withdrawal transaction, and settles them once the outcome is decided. A host
implements all three hooks or none.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from ...core.agent.live_input import (
    LiveInputClaim,
    LiveInputOffer,
    LiveInputResult,
    result_without_injection,
)
from ...core.runtime_performance import increment_counter
from . import task_execution as task_execution_service
from .agent_service_manager import get_agent_manager
from .chat_history_service import (
    DELIVERY_COMPLETED,
    DELIVERY_FAILED,
    DELIVERY_OUTCOME_UNKNOWN,
    claim_user_message_delivery_no_commit,
    inspect_user_message_delivery,
    mark_user_message_delivery_sync,
    withdraw_pending_user_message_delivery_no_commit,
)
from .db_runtime import run_db_io_cancellation_safe
from .task_command_transport import LIVE_INPUT_COMMAND_PREFIX, ClaimedTaskCommand
from .task_events import publish_task_event
from .turn_settlement import settlement_turn_for_run

logger = logging.getLogger(__name__)

LIVE_INPUT_COMMAND_SCOPE = "live_input"


def live_input_command_id(run_id: str, turn_id: str, attempt: int) -> str:
    """Command id for the ``attempt``-th delivery try of one message in one run."""

    digest = hashlib.sha256(f"{run_id}\x1f{turn_id}\x1f{attempt}".encode()).hexdigest()
    return f"{LIVE_INPUT_COMMAND_PREFIX}{digest[:48]}"


@dataclass(frozen=True)
class LiveInputPayload:
    """The validated payload of a live input command."""

    task_id: int
    turn_id: str
    message: str
    fields: dict[str, Any] = field(default_factory=dict)


def parse_live_input_payload(command: ClaimedTaskCommand) -> LiveInputPayload | None:
    """Parse ``command``'s payload, or ``None`` when it is not well formed."""

    payload = command.payload
    message = payload.get("message")
    turn_id = payload.get("turn_id")
    if not isinstance(message, str) or not isinstance(turn_id, str):
        return None
    if not message.strip() or not turn_id.strip():
        return None
    return LiveInputPayload(
        task_id=command.task_id,
        turn_id=turn_id.strip(),
        message=message,
        fields={k: v for k, v in payload.items() if k not in {"message", "turn_id"}},
    )


class LiveInputHost(Protocol):
    """What the embedding application supplies for live input.

    A ``queued`` command result is held in process memory only; see
    ``execute_live_input_command`` for what the host has to reconcile.
    """

    def authorize(self, db: Session, payload: LiveInputPayload) -> int | None:
        """Revalidate the sender; return the task owner's user id, or ``None``."""

    def record_applied(self, payload: LiveInputPayload) -> None:
        """Meter one message that was written into the run (best effort)."""


class LiveInputClaimHooks(Protocol):
    """What a host adds to record side effects together with the claim.

    Not part of ``LiveInputHost``: a host implements all three methods or none
    of them, and ``_claim_hooks`` is the only place that tells which.
    """

    def stage_claim(
        self,
        db: Session,
        payload: LiveInputPayload,
        *,
        settlement_turn_id: str | None,
    ) -> None:
        """Stage the host's rows for a message being claimed, in the claim's
        transaction, before its commit. ``settlement_turn_id`` is the turn id
        the run will report when it settles, or None when that is not known
        yet; it is a label only and not a basis for authorization. Raising
        fails the claim and leaves nothing behind. The hook must not commit or
        roll back."""

    def discard_claim(self, db: Session, payload: LiveInputPayload) -> None:
        """Remove what ``stage_claim`` staged, in the transaction that deletes
        the message's still-pending delivery row, only when that delete
        removed the row. Raising rolls the withdrawal back."""

    def settle_claim(self, payload: LiveInputPayload) -> None:
        """Called once a claimed message's outcome is decided and its delivery
        row stays: applied, refused or conflict. The host opens its own
        session. Best effort: an error is logged and changes nothing."""


_CLAIM_HOOK_NAMES = ("stage_claim", "discard_claim", "settle_claim")


def _claim_hooks(host: LiveInputHost) -> LiveInputClaimHooks | None:
    """The host's claim hooks, or None when it implements none of them.

    Raises ``ValueError`` when it implements only some: such a host would
    leave rows it can no longer withdraw or release.
    """
    present = [callable(getattr(host, name, None)) for name in _CLAIM_HOOK_NAMES]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "A live input host must implement stage_claim, discard_claim and "
            "settle_claim, or none of them"
        )
    return host  # type: ignore[return-value]


_host_lock = threading.Lock()
_host: LiveInputHost | None = None


def register_live_input_host(host: LiveInputHost) -> None:
    """Register the process-wide host; the same object again is a no-op."""

    global _host
    with _host_lock:
        if _host is host:
            return
        if _host is not None:
            raise ValueError("A live input host is already registered")
        _claim_hooks(host)
        _host = host


def unregister_live_input_host() -> LiveInputHost | None:
    global _host
    with _host_lock:
        host, _host = _host, None
        return host


def registered_live_input_host() -> LiveInputHost | None:
    with _host_lock:
        return _host


def live_input_returned_frame(
    task_id: int, turn_id: str, reason: str
) -> dict[str, Any]:
    return {
        "type": "live_input_returned",
        "task_id": task_id,
        "turn_id": turn_id,
        "reason": reason,
    }


async def _publish_returned(task_id: int, turn_id: str, reason: str) -> None:
    try:
        await publish_task_event(
            live_input_returned_frame(task_id, turn_id, reason), task_id
        )
    except Exception:
        logger.warning(
            "live input notice %s failed for task %s", reason, task_id, exc_info=True
        )


_background: set[asyncio.Task[None]] = set()


def _publish_in_background(task_id: int, turn_id: str, reason: str) -> None:
    task = asyncio.ensure_future(_publish_returned(task_id, turn_id, reason))
    _background.add(task)
    task.add_done_callback(_background.discard)


@dataclass
class _ClaimOutcome:
    claim: LiveInputClaim
    claimed: bool = False
    withdrawn: bool = False
    refused: bool = False


class DeliveryLedgerAdmission:
    """Records a live message in the delivery ledger around its injection.

    Its ledger writes are keyed by task and turn id and are not fenced by the
    task lease, so after a lost lease they can still land while the runner
    holds the run's cancellation back (``LIVE_INPUT_CANCEL_GRACE_SECONDS``).
    The injection's checkpoint is written under the run's task lease and is
    refused once the lease has moved to another runner.

    Rows the host stages with its claim hooks are written in the claim
    transaction and withdrawn in the withdrawal transaction, like the ledger
    row itself.
    """

    def __init__(
        self,
        payload: LiveInputPayload,
        host: LiveInputHost,
        *,
        settlement_turn_id: str | None = None,
    ) -> None:
        self._payload = payload
        self._host = host
        # The turn id the run reports when it settles, handed to the host's
        # claim hook; None when the run does not know it yet.
        self._settlement_turn_id = settlement_turn_id
        # True once this admission has staged a row of its own for the
        # commit. Only then may it change the state of the row under its turn
        # id; before that, any row there belongs to another delivery.
        self._wrote = False
        self._claimed = False
        self._withdrawn = False
        self._refused = False
        # The claim's outcome, recorded by its worker thread before the thread
        # returns. ``claim()`` can be cancelled after the thread has finished,
        # and the outcome is then known here although the caller never got it.
        self._claim_outcome: _ClaimOutcome | None = None
        # Whether ``claim()`` handed its outcome to the caller. Until it has,
        # the caller cannot have injected the message.
        self._claim_returned = False

    def _claim_sync(self) -> _ClaimOutcome:
        try:
            outcome = self._claim_in_session()
        except Exception:
            if self._wrote:
                raise
            # Nothing of this admission was written, so the message is not
            # claimed, whatever failed. The failures a claim expects are
            # handled where they occur; this one is unexpected, so it is
            # logged as an error and counted.
            logger.error(
                "live input claim failed before anything was written", exc_info=True
            )
            increment_counter("xagent.live_input.claim_errors")
            outcome = _ClaimOutcome(LiveInputClaim.UNAVAILABLE)
        self._claim_outcome = outcome
        return outcome

    def _claim_in_session(self) -> _ClaimOutcome:
        from ..models.database import get_session_local

        payload = self._payload
        with get_session_local()() as db:
            try:
                owner = self._host.authorize(db, payload)
            except SQLAlchemyError:
                logger.warning("live input authorization read failed", exc_info=True)
                return _ClaimOutcome(LiveInputClaim.UNAVAILABLE)
            if owner is None:
                return _ClaimOutcome(LiveInputClaim.UNAVAILABLE, refused=True)
            try:
                claim = claim_user_message_delivery_no_commit(
                    db,
                    payload.task_id,
                    owner,
                    payload.message,
                    attachments=None,
                    turn_id=payload.turn_id,
                )
            except IntegrityError:
                db.rollback()
                winner = inspect_user_message_delivery(
                    db,
                    payload.task_id,
                    payload.message,
                    attachments=None,
                    turn_id=payload.turn_id,
                )
                if winner is None:
                    return _ClaimOutcome(LiveInputClaim.UNAVAILABLE)
                return _ClaimOutcome(LiveInputClaim.TAKEN)
            except Exception:
                # The lock, the lookup or the insert failed before anything of
                # ours was committed. A row under this turn id, if any, belongs
                # to another delivery and must be left alone.
                logger.warning("live input claim failed before write", exc_info=True)
                db.rollback()
                return _ClaimOutcome(LiveInputClaim.UNAVAILABLE)
            if not claim.claimed:
                db.rollback()
                return _ClaimOutcome(LiveInputClaim.TAKEN)
            try:
                hooks = _claim_hooks(self._host)
                if hooks is not None:
                    hooks.stage_claim(
                        db, payload, settlement_turn_id=self._settlement_turn_id
                    )
                    # The session does not autoflush: write the host's rows
                    # here, so their failure fails the claim before it commits.
                    db.flush()
            except Exception:
                logger.warning("live input host claim staging failed", exc_info=True)
                increment_counter("xagent.live_input.claim_hook_errors")
                db.rollback()
                return _ClaimOutcome(LiveInputClaim.UNAVAILABLE)
            self._wrote = True
            try:
                db.commit()
            except Exception:
                logger.warning("live input claim commit failed", exc_info=True)
                try:
                    db.rollback()
                except Exception:
                    logger.warning("live input claim rollback failed", exc_info=True)
                return self._after_failed_commit()
        return _ClaimOutcome(LiveInputClaim.CLAIMED, claimed=True)

    def _after_failed_commit(self) -> _ClaimOutcome:
        payload = self._payload
        try:
            withdrawn = _withdraw_claimed_sync(payload, self._host)
            if not withdrawn:
                from ..models.database import get_session_local

                with get_session_local()() as db:
                    withdrawn = (
                        inspect_user_message_delivery(
                            db,
                            payload.task_id,
                            payload.message,
                            attachments=None,
                            turn_id=payload.turn_id,
                        )
                        is None
                    )
        except Exception:
            logger.warning("live input claim withdrawal failed", exc_info=True)
            return _ClaimOutcome(LiveInputClaim.UNKNOWN)
        if withdrawn:
            # The commit may have succeeded before it failed, so the claim is
            # treated as made and then taken back.
            return _ClaimOutcome(
                LiveInputClaim.UNAVAILABLE, claimed=True, withdrawn=True
            )
        return _ClaimOutcome(LiveInputClaim.UNKNOWN)

    async def claim(self) -> LiveInputClaim:
        outcome = await run_db_io_cancellation_safe(self._claim_sync)
        self._adopt(outcome)
        self._claim_returned = True
        return outcome.claim

    def _adopt(self, outcome: _ClaimOutcome) -> None:
        self._claimed = outcome.claimed
        self._withdrawn = outcome.withdrawn
        self._refused = outcome.refused

    async def _mark(self, status: str) -> str | None:
        if not self._wrote:
            return None
        payload = self._payload
        try:
            transition = await run_db_io_cancellation_safe(
                lambda: mark_user_message_delivery_sync(
                    payload.task_id, payload.turn_id, status
                )
            )
        except Exception:
            logger.warning(
                "live input delivery marker %s failed for task %s",
                status,
                payload.task_id,
                exc_info=True,
            )
            return None
        if transition.outcome != "updated":
            logger.info(
                "live input delivery marker %s left task %s row at %s",
                status,
                payload.task_id,
                transition.status,
            )
        return transition.outcome

    async def _withdraw(self) -> bool:
        payload = self._payload
        try:
            return await run_db_io_cancellation_safe(
                lambda: _withdraw_claimed_sync(payload, self._host)
            )
        except Exception:
            logger.warning("live input withdrawal failed", exc_info=True)
            return False

    async def _settle(self) -> None:
        """Tell the host this claimed message's outcome is decided.

        Only for a message whose row stays; a failure changes nothing.
        """
        if not self._wrote:
            return
        payload = self._payload
        try:
            hooks = _claim_hooks(self._host)
            if hooks is None:
                return
            await run_db_io_cancellation_safe(lambda: hooks.settle_claim(payload))
        except Exception:
            logger.warning("live input host settle failed", exc_info=True)
            increment_counter("xagent.live_input.settle_hook_errors")

    async def _record_applied(self) -> None:
        payload = self._payload
        try:
            await run_db_io_cancellation_safe(
                lambda: self._host.record_applied(payload)
            )
        except Exception:
            logger.warning("live input metering failed", exc_info=True)

    def _notify(self, reason: str) -> None:
        _publish_in_background(self._payload.task_id, self._payload.turn_id, reason)

    async def finish(self, result: LiveInputResult) -> None:
        if result is LiveInputResult.UNKNOWN:
            result = self._result_for_unreturned_claim()
        if result is LiveInputResult.APPLIED:
            # APPLIED is reported only after the message is in the run's
            # checkpoint, so the row records that as ``completed``. Other
            # writers can close a live input row as ``dispatched``, which does
            # not say whether the message was written: the orphan sweep in
            # lease recovery, the stop fallbacks, and the message command path
            # when a same-id message finds the row already accepted
            # (``settle_accepted_outcome_unknown``) or takes it over while it
            # is still pending. ``dispatched -> completed`` is allowed, so this
            # mark still moves such a row to ``completed``. If the mark raises,
            # the row keeps its prior status; if another writer already moved
            # it to ``outcome_unknown`` or ``failed``, the mark is a conflict
            # and that status is kept.
            await self._mark(DELIVERY_COMPLETED)
            await self._settle()
            await self._record_applied()
        elif result is LiveInputResult.RETURNED:
            if not self._claimed:
                self._notify("unavailable" if self._refused else "run_ended")
            elif self._withdrawn or await self._withdraw():
                self._notify("not_applied")
            else:
                await self._finish_unknown()
        elif result is LiveInputResult.TAKEN:
            self._notify("taken")
        elif result is LiveInputResult.REFUSED:
            await self._mark(DELIVERY_FAILED)
            await self._settle()
            self._notify("refused")
        elif result is LiveInputResult.CONFLICT:
            await self._mark(DELIVERY_FAILED)
            await self._settle()
            self._notify("conflict")
        else:
            await self._finish_unknown()

    async def _finish_unknown(self) -> None:
        await self._mark(DELIVERY_OUTCOME_UNKNOWN)
        self._notify("outcome_unknown")

    def _result_for_unreturned_claim(self) -> LiveInputResult:
        """``UNKNOWN``, unless the claim finished but its caller never saw it.

        That happens when ``claim()`` is cancelled while its worker thread
        runs: the thread still finishes and records its outcome, and a caller
        that never received a claim cannot have injected the message. The
        message is then finished as one that was not injected after that
        claim.
        """
        outcome = self._claim_outcome
        if self._claim_returned or outcome is None:
            return LiveInputResult.UNKNOWN
        self._adopt(outcome)
        return result_without_injection(outcome.claim)


def _withdraw_claimed_sync(payload: LiveInputPayload, host: LiveInputHost) -> bool:
    """Withdraw the claimed message's pending row and the host's rows together.

    One transaction: the host's ``discard_claim`` runs only when the row was
    removed, and a failure in it rolls the withdrawal back. Returns whether
    the row was removed.
    """
    from ..models.database import get_session_local

    with get_session_local()() as db:
        removed = withdraw_pending_user_message_delivery_no_commit(
            db, payload.task_id, payload.turn_id
        )
        if removed:
            hooks = _claim_hooks(host)
            if hooks is not None:
                hooks.discard_claim(db, payload)
        db.commit()
        return removed


def _done(result: str) -> dict[str, Any]:
    return {"result": result}


async def execute_live_input_command(
    command: ClaimedTaskCommand,
) -> dict[str, Any] | None:
    """Offer one claimed live input command to the run in this process.

    Reads memory only. Notices to the browser are published in the background
    so a slow event bus cannot hold up the commands queued behind this one.

    ``queued`` means only that the message is waiting in this process's
    memory. If the process exits before the run reaches its safe point,
    nothing records the message and no ``live_input_returned`` is published;
    a command that fails before it reaches this executor publishes none
    either. A host must therefore settle a message it has no notice for once
    the run named by the command's ``target_run_id`` is no longer executing,
    matching the command by its id and run id.
    """

    payload = parse_live_input_payload(command)
    if payload is None:
        logger.error("Malformed live input command %s", command.command_id)
        return _done("malformed")
    host = registered_live_input_host()
    if host is None:
        _publish_in_background(payload.task_id, payload.turn_id, "unavailable")
        return _done("no_host")

    service = get_agent_manager().agent_for_run(payload.task_id, command.target_run_id)
    if service is not None:
        offer = service.offer_live_user_message(
            str(payload.task_id),
            execution_message=payload.message,
            display_message=payload.message,
            turn_id=payload.turn_id,
            admission=DeliveryLedgerAdmission(
                payload,
                host,
                settlement_turn_id=settlement_turn_for_run(
                    payload.task_id, command.target_run_id
                ),
            ),
        )
        if offer in (LiveInputOffer.QUEUED, LiveInputOffer.ALREADY_QUEUED):
            return _done("queued")
        if offer is LiveInputOffer.ALREADY_ACCEPTED:
            return _done("already_applied")
        if offer is LiveInputOffer.CONFLICT:
            _publish_in_background(payload.task_id, payload.turn_id, "conflict")
            return _done("conflict")
        if offer is LiveInputOffer.UNSUPPORTED:
            _publish_in_background(payload.task_id, payload.turn_id, "run_ended")
            return _done("unsupported")

    if payload.task_id in task_execution_service.background_task_manager.running_tasks:
        increment_counter("xagent.live_input.not_started")
        _publish_in_background(payload.task_id, payload.turn_id, "not_started")
        return _done("not_started")
    _publish_in_background(payload.task_id, payload.turn_id, "run_ended")
    return _done("run_ended")
