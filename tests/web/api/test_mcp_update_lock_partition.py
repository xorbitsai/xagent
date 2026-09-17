"""Guards the partition ``update_mcp_server`` uses to decide which row a
``PUT /api/mcp/servers/{id}`` request writes -- and therefore which row it
locks (see the ``writes_definition_row`` comment in
``xagent/web/api/mcp.py``).

That decision is ``bool(server_data.model_fields_set - {"is_active",
"user_env"})``: those two fields write only the caller's own
``UserMCPServer`` link row, and every other field on ``MCPServerUpdate``
writes the shared ``MCPServer`` definition row. This module has no access
to that expression directly -- it lives as a local variable inside the
route function, not a standalone helper -- so it re-derives the partition
against the model's own declared field set instead, and fails loudly if
that set ever drifts out from under it: a new field added to
``MCPServerUpdate`` with no corresponding decision about which row it
writes would otherwise silently join the "writes the definition row"
side by default (any field not in the two-item exclusion set does), and
nothing else in this repo would notice.
"""

from __future__ import annotations

from xagent.web.api.mcp import MCPServerUpdate

# The complete, literal field set as of this test's writing. Kept as a
# literal (not derived from the model) so a field added to the model without
# a matching update here fails this assertion instead of silently passing.
_EXPECTED_FIELDS = {
    "name",
    "transport",
    "description",
    "config",
    "is_active",
    "user_env",
    "runtime_input_schema",
    "runtime_bindings",
    "allow_delegated_authorization",
}

# The two fields that write only the caller's own UserMCPServer link row.
# Mirrors the literal set in the ``writes_definition_row`` expression in
# ``xagent/web/api/mcp.py`` -- kept as a second literal, not imported from
# there, for the same reason: this test exists to catch that expression and
# this docstring's picture of it drifting apart un-noticed.
_LINK_ROW_ONLY_FIELDS = {"is_active", "user_env"}


def _writes_definition_row(payload: MCPServerUpdate) -> bool:
    """The exact expression ``update_mcp_server`` evaluates, reproduced here
    so this module can exercise it without a live request/session/route
    call -- see the module docstring for why this can't just import it."""
    return bool(payload.model_fields_set - _LINK_ROW_ONLY_FIELDS)


def test_mcp_server_update_field_set_matches_the_lock_partition() -> None:
    """If this fails, ``MCPServerUpdate`` gained or lost a field and the
    partition above (and the matching literals in
    ``xagent/web/api/mcp.py``'s ``writes_definition_row`` comment and this
    module's ``_LINK_ROW_ONLY_FIELDS``) needs a deliberate decision about
    which row the new field belongs to -- not a silent default."""
    assert set(MCPServerUpdate.model_fields) == _EXPECTED_FIELDS, (
        "MCPServerUpdate's fields no longer match the set this test and "
        "the route's writes_definition_row comment were written against "
        f"(saw {set(MCPServerUpdate.model_fields)!r}); decide which row "
        "the new/removed field belongs to before updating this literal"
    )
    assert _LINK_ROW_ONLY_FIELDS <= _EXPECTED_FIELDS, (
        "the link-row-only fields must themselves be a subset of the "
        "model's fields -- this failing means the two literals in this "
        "file have already drifted apart"
    )


def test_an_is_active_only_payload_does_not_write_the_definition_row() -> None:
    assert _writes_definition_row(MCPServerUpdate(is_active=True)) is False


def test_a_user_env_only_payload_does_not_write_the_definition_row() -> None:
    assert _writes_definition_row(MCPServerUpdate(user_env={"K": "v"})) is False


def test_a_combined_is_active_and_user_env_payload_does_not_write_the_definition_row() -> (
    None
):
    assert (
        _writes_definition_row(MCPServerUpdate(is_active=True, user_env={"K": "v"}))
        is False
    )


def test_a_description_only_payload_writes_the_definition_row() -> None:
    assert _writes_definition_row(MCPServerUpdate(description="new")) is True


def test_a_mixed_is_active_and_description_payload_writes_the_definition_row() -> None:
    """A payload naming one link-row field and one definition-row field
    must fall on the definition-row (locking) side -- the partition is
    "only the two link-row fields, and nothing else" skips the lock, not
    "the two link-row fields, even alongside other fields"."""
    assert (
        _writes_definition_row(MCPServerUpdate(is_active=True, description="new"))
        is True
    )


def test_an_explicitly_null_definition_row_field_still_writes_the_definition_row() -> (
    None
):
    """``model_fields_set`` tracks which fields were named in the payload,
    not which ones carry a non-``None`` value: an explicit
    ``runtime_input_schema: null`` is still a decision to write that field
    (clearing it), distinct from omitting it (leave it alone) -- so it
    must still land on the locking side even though its value is falsy."""
    assert _writes_definition_row(MCPServerUpdate(runtime_input_schema=None)) is True
