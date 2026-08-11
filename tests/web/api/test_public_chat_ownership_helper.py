"""Unit coverage for ``task_is_owned_by_public_principal``, the shared
ownership predicate extracted from ``public_chat_access.py``'s four
widget/share entry points (see that predicate's own docstring for the exact
conjunction and the two deliberate tightenings over the pre-existing code).

Pure-Python unit tests against unpersisted ``Task`` instances -- the
predicate never issues SQL, so no database fixture is needed here. Phase 2
of this delivery is what proves the three wired entry points still behave
identically after routing through this predicate (via the existing,
unmodified ``public_chat_access`` test suites).
"""

from __future__ import annotations

import pytest

from xagent.web.models.task import Task
from xagent.web.services.task_interaction_service import (
    InteractionPrincipal,
    public_chat_identity_matches,
    task_is_owned_by_public_principal,
)


def _task(**overrides: object) -> Task:
    defaults: dict[str, object] = {
        "id": 1,
        "user_id": 7,
        "channel_id": None,
        "agent_id": 5,
        "agent_config": {"auth_mode": "widget", "guest_id": "g-1"},
    }
    defaults.update(overrides)
    return Task(**defaults)


def _widget_agent_principal(**overrides: object) -> InteractionPrincipal:
    defaults: dict[str, object] = {
        "kind": "guest",
        "user_id": 7,
        "is_admin": False,
        "channel_id": None,
        "auth_mode": "widget",
        "widget_agent_id": 5,
        "guest_id": "g-1",
    }
    defaults.update(overrides)
    return InteractionPrincipal(**defaults)


def _widget_workforce_principal(**overrides: object) -> InteractionPrincipal:
    defaults: dict[str, object] = {
        "kind": "guest",
        "user_id": 7,
        "is_admin": False,
        "channel_id": None,
        "auth_mode": "widget",
        "widget_workforce_id": 9,
        "guest_id": "g-1",
    }
    defaults.update(overrides)
    return InteractionPrincipal(**defaults)


def _share_agent_principal(**overrides: object) -> InteractionPrincipal:
    defaults: dict[str, object] = {
        "kind": "guest",
        "user_id": 7,
        "is_admin": False,
        "channel_id": None,
        "auth_mode": "share",
        "share_agent_id": 5,
        "guest_id": "g-1",
    }
    defaults.update(overrides)
    return InteractionPrincipal(**defaults)


def _share_workforce_principal(**overrides: object) -> InteractionPrincipal:
    defaults: dict[str, object] = {
        "kind": "guest",
        "user_id": 7,
        "is_admin": False,
        "channel_id": None,
        "auth_mode": "share",
        "share_workforce_id": 9,
        "guest_id": "g-1",
    }
    defaults.update(overrides)
    return InteractionPrincipal(**defaults)


# ---------------------------------------------------------------------------
# Four directions, each a passing case
# ---------------------------------------------------------------------------


def test_widget_agent_direction_passes_when_the_conjunction_holds() -> None:
    task = _task(agent_id=5, agent_config={"auth_mode": "widget", "guest_id": "g-1"})
    assert task_is_owned_by_public_principal(task, _widget_agent_principal()) is True


def test_widget_workforce_direction_passes_when_the_conjunction_holds() -> None:
    task = _task(
        agent_id=None,
        agent_config={
            "auth_mode": "widget",
            "guest_id": "g-1",
            "widget_workforce_id": 9,
        },
    )
    assert (
        task_is_owned_by_public_principal(task, _widget_workforce_principal()) is True
    )


def test_share_agent_direction_passes_when_the_conjunction_holds() -> None:
    task = _task(
        agent_id=5,
        agent_config={"auth_mode": "share", "guest_id": "g-1", "share_agent_id": 5},
    )
    assert task_is_owned_by_public_principal(task, _share_agent_principal()) is True


def test_share_workforce_direction_passes_when_the_conjunction_holds() -> None:
    task = _task(
        agent_id=None,
        agent_config={
            "auth_mode": "share",
            "guest_id": "g-1",
            "share_workforce_id": 9,
        },
    )
    assert task_is_owned_by_public_principal(task, _share_workforce_principal()) is True


# ---------------------------------------------------------------------------
# A2/A3/A4: rejections carried over from the design's failure-matrix rows
# ---------------------------------------------------------------------------


def test_same_guest_id_but_task_bound_to_a_different_workforce_is_rejected() -> None:
    """A2: a widget guest whose guest_id matches, but the task belongs to a
    different widget_workforce_id, must not be treated as this guest's
    task."""

    task = _task(
        agent_id=None,
        agent_config={
            "auth_mode": "widget",
            "guest_id": "g-1",
            "widget_workforce_id": 999,
        },
    )
    assert (
        task_is_owned_by_public_principal(task, _widget_workforce_principal()) is False
    )


def test_predicate_alone_does_not_enforce_task_owner_by_design() -> None:
    """A3 (same guest_id, another owner -> reject) holds end-to-end, but not
    from this predicate alone: ownership by ``Task.user_id`` stays a
    caller-side SQL filter in every one of the three wired entry points
    (see ``task_is_owned_by_public_principal``'s docstring), so a ``task``
    this predicate is handed with every *other* conjunct satisfied passes
    regardless of ``task.user_id`` -- the production A3 rejection comes from
    the caller's query never loading such a task in the first place (pinned
    by ``test_share_guest_isolation.py`` and ``test_workforce_widget.py``,
    which exercise the real query path end to end). This test documents
    that division of labor rather than re-asserting A3 at a layer that does
    not own it."""

    task = _task(user_id=999, agent_id=5)
    principal = _widget_agent_principal()
    assert task_is_owned_by_public_principal(task, principal) is True


def test_widget_principal_against_a_share_task_with_matching_guest_id_is_rejected() -> (
    None
):
    """A4: a widget principal and a share task whose guest_id happen to be
    equal, but auth_mode disagrees. The same inputs fed to the untouched
    get_task_for_public_context on the current tree return the task instead
    of rejecting it -- this test asserts the helper's tightening over that
    existing behavior, not existing behavior itself."""

    share_task = _task(
        agent_id=5,
        agent_config={"auth_mode": "share", "guest_id": "g-1", "share_agent_id": 5},
    )
    widget_principal = _widget_agent_principal(guest_id="g-1")
    assert task_is_owned_by_public_principal(share_task, widget_principal) is False


# ---------------------------------------------------------------------------
# Entity binding must not short-circuit when the task-side value is missing
# ---------------------------------------------------------------------------


def test_widget_agent_binding_rejects_when_task_has_no_agent() -> None:
    task = _task(agent_id=None)
    assert task_is_owned_by_public_principal(task, _widget_agent_principal()) is False


def test_widget_workforce_binding_rejects_when_config_key_is_missing() -> None:
    task = _task(agent_id=None, agent_config={"auth_mode": "widget", "guest_id": "g-1"})
    assert (
        task_is_owned_by_public_principal(task, _widget_workforce_principal()) is False
    )


def test_share_agent_binding_rejects_when_config_key_is_missing_even_if_row_level_matches() -> (
    None
):
    task = _task(agent_id=5, agent_config={"auth_mode": "share", "guest_id": "g-1"})
    assert task_is_owned_by_public_principal(task, _share_agent_principal()) is False


def test_share_workforce_binding_rejects_when_config_key_is_missing() -> None:
    task = _task(agent_id=None, agent_config={"auth_mode": "share", "guest_id": "g-1"})
    assert (
        task_is_owned_by_public_principal(task, _share_workforce_principal()) is False
    )


# ---------------------------------------------------------------------------
# Misuse and other conjuncts
# ---------------------------------------------------------------------------


def test_principal_addressing_zero_directions_raises() -> None:
    principal = InteractionPrincipal(
        kind="guest",
        user_id=7,
        is_admin=False,
        channel_id=None,
        auth_mode="widget",
        guest_id="g-1",
    )
    with pytest.raises(ValueError):
        task_is_owned_by_public_principal(_task(), principal)


def test_principal_addressing_two_directions_raises() -> None:
    principal = InteractionPrincipal(
        kind="guest",
        user_id=7,
        is_admin=False,
        channel_id=None,
        auth_mode="widget",
        widget_agent_id=5,
        widget_workforce_id=9,
        guest_id="g-1",
    )
    with pytest.raises(ValueError):
        task_is_owned_by_public_principal(_task(), principal)


def test_channel_bound_task_is_rejected() -> None:
    task = _task(channel_id=42)
    assert task_is_owned_by_public_principal(task, _widget_agent_principal()) is False


def test_non_dict_agent_config_is_rejected() -> None:
    task = _task(agent_config=None)
    assert task_is_owned_by_public_principal(task, _widget_agent_principal()) is False


def test_empty_guest_id_on_principal_is_rejected() -> None:
    principal = _widget_agent_principal(guest_id="")
    assert task_is_owned_by_public_principal(_task(), principal) is False


def test_mismatched_guest_id_is_rejected() -> None:
    principal = _widget_agent_principal(guest_id="someone-else")
    assert task_is_owned_by_public_principal(_task(), principal) is False


def test_identity_string_for_guest_principal() -> None:
    principal = _widget_agent_principal(guest_id="abc")
    assert principal.identity_string() == "guest:abc"


def test_identity_string_for_user_principal() -> None:
    principal = InteractionPrincipal(
        kind="user",
        user_id=42,
        is_admin=False,
        channel_id=None,
        auth_mode=None,
    )
    assert principal.identity_string() == "user:42"


# ---------------------------------------------------------------------------
# public_chat_identity_matches: the message-selection slice used by Phase 2
# ---------------------------------------------------------------------------


def test_identity_matches_when_guest_id_agrees() -> None:
    task = _task(agent_config={"guest_id": "g-1"})
    assert public_chat_identity_matches(task, _widget_agent_principal()) is True


def test_identity_does_not_match_on_guest_id_mismatch() -> None:
    task = _task(agent_config={"guest_id": "someone-else"})
    assert public_chat_identity_matches(task, _widget_agent_principal()) is False


def test_identity_does_not_match_when_principal_guest_id_is_empty() -> None:
    task = _task(agent_config={"guest_id": ""})
    principal = _widget_agent_principal(guest_id="")
    assert public_chat_identity_matches(task, principal) is False


def test_identity_does_not_match_when_agent_config_is_not_a_dict() -> None:
    task = _task(agent_config=None)
    assert public_chat_identity_matches(task, _widget_agent_principal()) is False


def test_identity_match_does_not_read_task_user_id() -> None:
    """Deliberate: a stub task lacking ``user_id`` entirely (as some
    pre-existing public_chat_access tests construct) must not raise --
    ownership by user stays the caller's SQL-filter responsibility, never a
    post-load attribute read here."""

    from types import SimpleNamespace

    stub = SimpleNamespace(agent_config={"guest_id": "g-1"})
    assert public_chat_identity_matches(stub, _widget_agent_principal()) is True
