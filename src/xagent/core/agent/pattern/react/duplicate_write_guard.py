"""Same-turn duplicate-write guard for the ReAct tool-execution path.

Defends against a model re-issuing a write-category tool call with
byte-identical arguments after that call already succeeded in the same turn
(xorbitsai/xagent#2217): the repeat is not executed; the model receives a
structured envelope carrying the prior result instead.

Scope is deliberately narrow:

* Same turn only, enforced by the ledger record's guard turn identity. An
  ordinary completion uses its original turn id. A resumed success uses the
  settlement turn that received the result, so a model cannot immediately
  repeat the approved write after resume. A later user turn remains a new
  authorization boundary and may issue the same arguments again. A call with
  no guard turn id is never guarded: suppression must not outlive a turn, so
  an unknowable turn fails open.
* Identical execution arguments only, compared via the ledger's canonical
  args hash.
* Only tools that *explicitly* declare a non-idempotent write:

  - an MCP tool whose wire annotations classify as a non-idempotent write
    (``classify_non_idempotent_write`` in the MCP adapter: an exact
    ``idempotentHint: false``, or an exact ``destructiveHint: true`` with no
    idempotency promise, never a read-only tool), carried here as
    ``ToolMetadata.mcp_non_idempotent_write``; or
  - an internal tool that sets ``non_idempotent = True`` on itself, carried
    as ``ToolMetadata.non_idempotent``.

  Undeclared tools are exempt: an unannotated MCP tool may be a legitimate
  identical-args poll loop, and suppressing it would hand the model stale
  data. Deduplication therefore fails OPEN — the mirror image of a
  confirmation-style consumer, for which ``MCPWriteHint``'s docstring
  prescribes treating everything except an explicit read-only claim as a
  write.

Enrollment is read from ``tool.metadata`` rather than the concrete tool
object: wrappers such as the sandbox tool wrapper forward only
``.metadata``, so a declaration read off the adapter directly would vanish
for every sandboxed MCP server. A bare ``non_idempotent = True`` attribute
on the tool object is honored as well for tools that do not build their
metadata through ``AbstractBaseTool``.
"""

from __future__ import annotations

from typing import Any

# Marker key stamped on every suppression envelope. Doubles as the signal
# that a ledger record is an envelope rather than a genuine execution, so a
# later duplicate always attaches the original result, never a suppression
# of a suppression.
DUPLICATE_WRITE_SUPPRESSED_KEY = "duplicate_write_suppressed"


def tool_requires_duplicate_write_guard(tool: Any) -> bool:
    """Whether ``tool`` explicitly declares itself a non-idempotent write.

    True for exactly three opt-in declarations — the tool object's own
    ``non_idempotent is True`` marker, the same marker carried on its
    metadata, or an MCP wire declaration carried as
    ``metadata.mcp_non_idempotent_write is True``. ``is True`` throughout so
    a truthy non-boolean never enrolls a tool by accident; everything
    undeclared stays exempt (see the module docstring for why deduplication
    fails open).
    """
    if getattr(tool, "non_idempotent", None) is True:
        return True

    metadata = getattr(tool, "metadata", None)
    if metadata is None:
        return False
    if getattr(metadata, "non_idempotent", None) is True:
        return True
    return getattr(metadata, "mcp_non_idempotent_write", None) is True


def build_suppression_envelope(
    *,
    tool_name: str,
    prior_tool_call_id: str,
    prior_result: Any,
    prior_succeeded: bool = True,
) -> dict[str, Any]:
    """Build the model-facing envelope for a suppressed duplicate write.

    A successful prior call produces a successful envelope because the effect
    already exists. A denied or dispatch-unknown settlement stays unsuccessful
    while still preventing the write from being executed again.
    """
    outcome = "already succeeded" if prior_succeeded else "was denied or is uncertain"
    return {
        "success": prior_succeeded,
        DUPLICATE_WRITE_SUPPRESSED_KEY: True,
        "tool_name": tool_name,
        "suppressed_duplicate_of": prior_tool_call_id,
        "result": prior_result,
        "message": (
            f"Duplicate write suppressed: this exact {tool_name} call "
            f"{outcome} earlier in this turn "
            f"(tool call {prior_tool_call_id}); it was not executed again. "
            "The previous call's result is attached under 'result' — use it "
            "instead of retrying. Repeating this write with identical "
            "arguments is only possible in a later turn."
        ),
    }
