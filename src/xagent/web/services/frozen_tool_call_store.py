"""The host half of the write gate: freeze a call, then run exactly that call.

Installs the two hooks ``mcp_adapter`` consults (``write_gate``), keeping
every database dependency on this side of the seam -- the adapter itself
runs inside the sandbox for npx/uvx connectors, where sqlalchemy is not
installed.

The invariant this module exists to hold: between the pause and the
decision, the row is the authority on what executes. Nothing here ever asks
the model what the arguments were.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, cast

from sqlalchemy import update

from ...core.tools.adapters.vibe.write_gate import (
    GatedCall,
    GateDecision,
    set_write_gate_hook,
    set_write_gate_resume_hook,
)
from ..models.database import get_session_local
from ..models.frozen_tool_call import FrozenToolCall
from .task_lease_service import current_task_lease

logger = logging.getLogger(__name__)

# How long an unanswered approval stays runnable. Long enough that a question
# asked at the end of a working day can still be answered the next morning,
# short enough that a forgotten approval does not execute against a world
# that has moved on.
DEFAULT_APPROVAL_TTL = timedelta(hours=24)

# What the gate treats as "the server told us this is read-only". Everything
# else -- destructive, or undeclared -- is a write. Undeclared is the common
# case: annotations are optional and most connectors omit them, so treating
# silence as safe would leave the gate closed only for servers polite enough
# to admit what they do.
_READ_ONLY = "read_only"

# A policy decides which writes need approval. ``None`` means the gate is
# installed but nothing is gated, which is what keeps this module inert until
# a deployment opts in.
WritePolicy = Callable[[GatedCall], bool]


def _requires_approval(call: GatedCall, policy: WritePolicy | None) -> bool:
    """Whether ``call`` needs a human before it runs."""
    if call.write_hint == _READ_ONLY:
        return False
    return bool(policy(call)) if policy is not None else False


def install_write_gate(policy: WritePolicy | None = None) -> None:
    """Install both halves of the gate for this process.

    ``policy=None`` installs the hooks without gating anything, so an
    approval already pending from an earlier deployment can still be
    resolved while new calls run ungated.
    """

    def gate(call: GatedCall) -> GateDecision | None:
        if not _requires_approval(call, policy):
            return None
        lease = current_task_lease()
        if lease is None:
            # No task to scope the row to. Ungated rather than blocked: see
            # ``consult_write_gate`` -- this seam makes an approved call
            # faithful, it is not what stops a dangerous one.
            logger.warning(
                "Write gate could not resolve the running task for %s; "
                "executing ungated",
                call.tool_name,
            )
            return None
        interaction_id = f"ftc_{uuid.uuid4().hex}"
        now = datetime.now(UTC)
        session_local = get_session_local()
        with session_local() as db:
            db.add(
                FrozenToolCall(
                    interaction_id=interaction_id,
                    task_id=lease.task_id,
                    tool_name=call.tool_name,
                    server_name=call.server_name,
                    write_hint=call.write_hint,
                    # dict() so a later mutation of the caller's mapping
                    # cannot reach what was frozen.
                    arguments=dict(call.arguments),
                    status="pending",
                    created_at=now,
                    expires_at=now + DEFAULT_APPROVAL_TTL,
                )
            )
            db.commit()
        return GateDecision(
            approval_required=True,
            interaction_id=interaction_id,
            message=f"Approve running {call.tool_name}?",
        )

    async def resume(
        *,
        interaction_id: str,
        approved: bool,
        executor: Callable[[Mapping[str, Any]], Any],
    ) -> Any:
        session_local = get_session_local()
        now = datetime.now(UTC)
        with session_local() as db:
            # Claim the row before running anything: the UPDATE ... WHERE
            # status = 'pending' is what makes a double-click execute once.
            # A second answer finds no pending row and is reported as
            # already settled rather than running the payload again.
            row = db.get(FrozenToolCall, interaction_id)
            if row is None:
                return _settled_result("This approval is no longer available.")
            arguments = dict(row.arguments or {})
            # Read through cast(): the model declares its columns with bare
            # ``Column(...)`` rather than ``Mapped[...]``, so an instance
            # attribute is typed as the Column descriptor rather than the
            # value it holds.
            expired = _as_utc(cast(datetime, row.expires_at)) <= now
            target = "executed" if (approved and not expired) else "voided"
            result = db.execute(
                update(FrozenToolCall)
                .where(
                    FrozenToolCall.interaction_id == interaction_id,
                    FrozenToolCall.status == "pending",
                )
                .values(status=target, settled_at=now)
            )
            # ``rowcount`` is what makes this claim exclusive; it is defined
            # for an UPDATE on both backends, but the generic ``Result``
            # protocol does not declare it.
            claimed = int(cast(Any, result).rowcount or 0)
            db.commit()
        if not claimed:
            return _settled_result("This approval has already been answered.")
        if expired:
            return _settled_result("This approval expired before it was answered.")
        if not approved:
            return {"success": False, "status": "cancelled", "cancelled": True}
        # Only here, and only with what was frozen.
        return await executor(arguments)

    set_write_gate_hook(gate)
    set_write_gate_resume_hook(resume)


def uninstall_write_gate() -> None:
    """Remove both hooks, restoring ungated execution."""
    set_write_gate_hook(None)
    set_write_gate_resume_hook(None)


def _as_utc(value: datetime) -> datetime:
    """Read a stored timestamp as UTC.

    SQLite hands back naive datetimes even for ``DateTime(timezone=True)``,
    so an expiry comparison against an aware ``now`` would raise there. The
    stored value is always UTC; this only restores the tzinfo the backend
    dropped.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _settled_result(message: str) -> dict[str, Any]:
    return {"success": False, "status": "error", "error": message}
