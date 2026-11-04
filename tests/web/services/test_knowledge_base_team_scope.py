"""Unit tests for the knowledge-base team-scope seam."""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.tools.core.knowledge_base_scope import KnowledgeBaseScopeError
from xagent.web.services import knowledge_base_team_scope as kb_scope

T1 = 11
T2 = 22


@pytest.fixture(autouse=True)
def _isolated_hooks():
    with kb_scope.snapshot_knowledge_base_team_hooks():
        kb_scope.set_knowledge_base_team_hooks()
        yield


def _valid_access(
    name: str = "handbook", storage_user_id: int = 7
) -> kb_scope.KnowledgeBaseAccess:
    return kb_scope.KnowledgeBaseAccess(
        name=name,
        storage_user_id=storage_user_id,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )


# ---------------------------------------------------------------------------
# The team-keyed Protocol's keyword-only marker: a user-keyed implementation
# bound into the team-keyed slot raises TypeError instead of resolving
# against the wrong tenant.
# ---------------------------------------------------------------------------


def test_user_keyed_function_in_team_slot_raises_type_error():
    """The reverse swap is caught: ``team_knowledge_bases`` always calls
    the team-keyed slot with ``team_id=`` as a keyword, so a user-keyed
    implementation whose second parameter is positional-or-keyword under a
    different name raises ``TypeError`` at call time instead of resolving
    against the wrong tenant.
    """

    def user_keyed_shaped(db: Any, user_id: int) -> list:
        return []

    kb_scope.set_knowledge_base_team_hooks(team_visibility=user_keyed_shaped)

    with pytest.raises(TypeError):
        kb_scope.team_knowledge_bases(None, team_id=T1)


# ---------------------------------------------------------------------------
# The validator's seven rejection classes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer,description",
    [
        (None, "non-iterable"),
        ([{"name": "x", "storage_user_id": 1}], "wrong element type"),
        (
            [
                kb_scope.KnowledgeBaseAccess(
                    name="",
                    storage_user_id=1,
                    team_owned=True,
                    can_edit=False,
                    can_delete=False,
                )
            ],
            "empty name",
        ),
        (
            [
                kb_scope.KnowledgeBaseAccess(
                    name="x",
                    storage_user_id="1",
                    team_owned=True,
                    can_edit=False,
                    can_delete=False,  # type: ignore[arg-type]
                )
            ],
            "non-int storage_user_id",
        ),
        (
            [
                kb_scope.KnowledgeBaseAccess(
                    name="x",
                    storage_user_id=True,
                    team_owned=True,
                    can_edit=False,
                    can_delete=False,  # type: ignore[arg-type]
                )
            ],
            "bool storage_user_id",
        ),
        (
            [
                kb_scope.KnowledgeBaseAccess(
                    name="x", storage_user_id=1, team_owned=True
                )
            ],
            "can_edit/can_delete omitted, so the dataclass defaults (True) apply",
        ),
        (
            [
                kb_scope.KnowledgeBaseAccess(
                    name="x",
                    storage_user_id=1,
                    team_owned=False,
                    can_edit=False,
                    can_delete=False,
                )
            ],
            "team_owned not True",
        ),
    ],
)
def test_kb_team_hook_shape_violation_raises(answer, description) -> None:
    def _hook(db: Any, *, team_id: int):
        return answer

    kb_scope.set_knowledge_base_team_hooks(team_visibility=_hook)

    with pytest.raises(ValueError):
        kb_scope.team_knowledge_bases(None, team_id=T1)


def test_valid_answer_round_trips():
    kb_scope.set_knowledge_base_team_hooks(
        team_visibility=lambda db, *, team_id: [_valid_access()]
    )
    result = kb_scope.team_knowledge_bases(None, team_id=T1)
    assert result == [_valid_access()]
    for entry in result:
        assert entry.can_edit is False
        assert entry.can_delete is False
        assert entry.team_owned is True


def test_no_input_for_which_a_malformed_answer_returns_empty_instead_of_raising():
    """There exists no malformed-answer input for which
    ``resolve_team_knowledge_bases_or_raise`` returns ``[]`` after an
    internal failure -- it always raises the typed error instead."""

    kb_scope.set_knowledge_base_team_hooks(
        team_visibility=lambda db, *, team_id: [
            kb_scope.KnowledgeBaseAccess(name="", storage_user_id=1)
        ]
    )
    with pytest.raises(KnowledgeBaseScopeError):
        kb_scope.resolve_team_knowledge_bases_or_raise(None, team_id=T1, log_subject=7)


def test_resolve_or_raise_passes_a_typed_error_through_unchanged():
    def _raising(db: Any, *, team_id: int):
        raise KnowledgeBaseScopeError("already-typed", "boom")

    kb_scope.set_knowledge_base_team_hooks(team_visibility=_raising)

    with pytest.raises(KnowledgeBaseScopeError) as excinfo:
        kb_scope.resolve_team_knowledge_bases_or_raise(None, team_id=T1, log_subject=7)
    assert excinfo.value.code == "already-typed"


def test_resolve_or_raise_wraps_arbitrary_hook_failure():
    def _raising(db: Any, *, team_id: int):
        raise RuntimeError("db unavailable")

    kb_scope.set_knowledge_base_team_hooks(team_visibility=_raising)

    with pytest.raises(KnowledgeBaseScopeError) as excinfo:
        kb_scope.resolve_team_knowledge_bases_or_raise(None, team_id=T1, log_subject=7)
    assert excinfo.value.status_code == 503


# ---------------------------------------------------------------------------
# No hook installed -> zero calls, predicate reflects presence.
# ---------------------------------------------------------------------------


def test_team_knowledge_bases_empty_without_hook_installed():
    assert kb_scope.team_knowledge_base_hook_installed() is False
    assert kb_scope.team_knowledge_bases(None, team_id=T1) == []


def test_team_knowledge_base_hook_installed_reflects_presence():
    assert kb_scope.team_knowledge_base_hook_installed() is False
    kb_scope.set_knowledge_base_team_hooks(team_visibility=lambda db, *, team_id: [])
    assert kb_scope.team_knowledge_base_hook_installed() is True


# ---------------------------------------------------------------------------
# The snapshot/restore primitive restores every slot.
# ---------------------------------------------------------------------------


_SLOT_INSTALLERS = {
    "visibility": lambda: kb_scope.set_knowledge_base_team_hooks(
        visibility=lambda db, user_id: []
    ),
    "team_visibility": lambda: kb_scope.set_knowledge_base_team_hooks(
        team_visibility=lambda db, *, team_id: []
    ),
}


@pytest.mark.parametrize("slot", sorted(_SLOT_INSTALLERS))
def test_kb_hooks_snapshot_restores_all_slots(slot: str) -> None:
    kb_scope.set_knowledge_base_team_hooks()  # every slot starts cleared

    with kb_scope.snapshot_knowledge_base_team_hooks():
        _SLOT_INSTALLERS[slot]()
        assert kb_scope.team_knowledge_base_hook_installed() == (
            slot == "team_visibility"
        )

    # Every slot -- not only the one this iteration touched -- is back to
    # its pre-entry (cleared) state.
    assert kb_scope.team_knowledge_base_hook_installed() is False
    assert kb_scope.has_knowledge_base_visibility_hook() is False


def test_kb_hooks_snapshot_restores_object_identity() -> None:
    sentinel_visibility = lambda db, user_id: []  # noqa: E731
    kb_scope.set_knowledge_base_team_hooks(visibility=sentinel_visibility)

    with kb_scope.snapshot_knowledge_base_team_hooks():
        kb_scope.set_knowledge_base_team_hooks(
            visibility=lambda db, user_id: [_valid_access()]
        )
        assert kb_scope._visibility_hook is not sentinel_visibility

    assert kb_scope._visibility_hook is sentinel_visibility
    kb_scope.set_knowledge_base_team_hooks()
