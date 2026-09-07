"""The seam a host uses to require approval before an MCP write executes.

A gated call is not executed. The hook is handed the call exactly as it was
about to run -- tool name and arguments -- and answers with either "run it"
or a pause carrying the frozen payload's identity. On approval the same
arguments are executed verbatim, because they were never handed back to the
model to be written a second time.

**Why a host-injected hook rather than a direct call into the web layer.**
``MCPToolAdapter.run_json_async`` is reconstructed and executed *inside the
sandbox* for every npx/uvx MCP tool (``sandboxed_tool/tool_runner.py``),
where sqlalchemy is not installed. Anything that reached for a database from
the adapter would turn every sandboxed tool call into a
``ModuleNotFoundError``. The host installs this hook only in the process
that can serve it; in the sandbox nothing is installed and every call runs
exactly as it does today.

That also *is* the off switch. No registration means no gate -- there is no
policy to consult, no row to write, and no behavior change at all.

**Not a trust boundary.** The hook decides using, among other things, a
server's own ``readOnlyHint``/``destructiveHint`` annotations, which the MCP
spec says a client must not trust from an untrusted server. What this seam
guarantees is narrower and worth stating exactly: *if* a call is gated, the
arguments that eventually execute are the ones that were shown, byte for
byte. It does not guarantee that every dangerous call gets gated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class GatedCall:
    """One MCP call presented to the gate before it runs."""

    tool_name: str
    """The tool's runtime name, as the model called it."""

    server_name: str
    """Normalized identity of the MCP server the tool came from."""

    arguments: Mapping[str, Any]
    """The arguments the call would have executed with, already normalized."""

    write_hint: str
    """The server's own write declaration: an ``MCPWriteHint`` value.

    Carried as a plain string so this module stays independent of the
    adapter's enum. A hook must treat everything except ``"read_only"`` as a
    write: ``"undeclared"`` is the common case, not a promise of safety.
    """


@dataclass(frozen=True)
class GateDecision:
    """What the host decided about one gated call.

    ``interaction_id`` is the identity the frozen payload was stored under
    and the identity the resume callback will be handed back. It is the
    hook's to mint: the adapter neither generates nor interprets it, it only
    carries it into the pause so the two halves meet.
    """

    approval_required: bool
    interaction_id: str = ""
    message: str = ""


WriteGateHook = Callable[[GatedCall], Optional[GateDecision]]

# Given the interaction's identity, whether the user approved, and a callable
# that runs one frozen argument set, the host loads the frozen payload,
# settles the row exactly once, and returns the tool result. The executor is
# passed in rather than imported because only the adapter knows how to place
# a call on its own connection.
WriteGateResumeHook = Callable[..., Any]

_HOOK: WriteGateHook | None = None
_RESUME_HOOK: WriteGateResumeHook | None = None

# The only response that runs a frozen call. Compared after stripping and
# case-folding, and nothing else is treated as consent -- an unrecognized
# answer voids the row rather than being guessed at.
_APPROVAL_GRANT = "approve"


def _approval_response_is_grant(response: Any) -> bool:
    """Whether ``response`` is the explicit approval value.

    Only an exact ``"approve"`` grants. A free-text reply that happens to
    contain the word does not: the pause offers two option values, and
    anything else reaching here means the answer did not come from those
    buttons.
    """
    return isinstance(response, str) and response.strip().lower() == _APPROVAL_GRANT


def set_write_gate_hook(hook: WriteGateHook | None) -> None:
    """Install (or clear) the process-wide approval hook.

    Idempotent and last-writer-wins, matching ``set_connector_runtime_resolver``.
    Passing ``None`` restores ungated execution.
    """
    global _HOOK
    _HOOK = hook


def get_write_gate_hook() -> WriteGateHook | None:
    """Return the installed hook, or ``None`` when nothing gates writes."""
    return _HOOK


def set_write_gate_resume_hook(hook: WriteGateResumeHook | None) -> None:
    """Install (or clear) the hook that settles an approved or rejected call."""
    global _RESUME_HOOK
    _RESUME_HOOK = hook


def get_write_gate_resume_hook() -> WriteGateResumeHook | None:
    """Return the installed resume hook, or ``None``."""
    return _RESUME_HOOK


def consult_write_gate(call: GatedCall) -> GateDecision | None:
    """Ask the installed hook about ``call``; ``None`` means "just run it".

    A hook that raises is treated as no decision and the call proceeds. That
    direction is deliberate and is the opposite of what a security boundary
    would do, for the reason in the module docstring: this seam makes an
    approved call faithful, it is not what keeps a dangerous call from
    running. Failing closed here would let a transient database error strand
    every connector call in a workspace behind an approval nobody can grant,
    which trades a bounded loss of gating for an unbounded loss of function.
    The host owns the decision to fail closed on its own side, where it can
    tell a policy miss from an outage.
    """
    hook = _HOOK
    if hook is None:
        return None
    try:
        return hook(call)
    except Exception:  # noqa: BLE001 - see the docstring
        import logging

        logging.getLogger(__name__).warning(
            "Write gate hook failed for %s; executing ungated",
            call.tool_name,
            exc_info=True,
        )
        return None
