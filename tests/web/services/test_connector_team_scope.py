"""Unit tests for the ``connector_team_scope`` seam itself, plus the checks
that need a real database: agent-team-keyed visibility (never the runner's
own membership), the legacy-hook-only fallback contract, and the
custom-API twins of the MCP-focused checks -- team-keyed connector
visibility covers both connector kinds, even though most of the
surrounding tests in this file only exercise MCP.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.requests import Request

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.models import Base, MCPServer, Task, User, UserMCPServer
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import (
    _ROOT_TXN_END_COUNT_KEY,
    release_db_connection_if_clean,
)
from xagent.web.models.task import TaskStatus
from xagent.web.services import agent_team_scope, connector_team_scope
from xagent.web.services.connector_runtime import _load_visible_runtime_connectors
from xagent.web.services.connector_team_scope import (
    ConnectorHookSessionBoundaryError,
    connector_hook_session_boundary_error_handler,
)
from xagent.web.tools.config import WebToolConfig, _load_custom_api_runtime_view_sync

T1 = 101
T2 = 102


# ---------------------------------------------------------------------------
# Seam unit tests -- no DB required.
# ---------------------------------------------------------------------------


@contextmanager
def _reset_hooks_scope() -> Iterator[None]:
    # Snapshot-and-restore, not clear-everything: this module's own
    # ``set_connector_team_hooks`` docstring says it clears every slot it
    # is not given, so calling it bare to "reset" would drop whatever the
    # process had installed before this file ran. ``snapshot_connector_team_hooks``
    # is what the newer suites in this repo use; older suites elsewhere in
    # tests/ still reset by clearing, which is a separate cleanup and not
    # this file's to make. Pulled out of the fixture below so a test can
    # exercise this scope directly (see
    # ``test_the_reset_scope_restores_a_pre_installed_hook_rather_than_clearing_it``),
    # since the fixture itself wraps the whole test body and cannot be
    # asserted on from inside one.
    try:
        with connector_team_scope.snapshot_connector_team_hooks():
            yield
    finally:
        agent_team_scope.set_agent_team_scope_hook(None)


@pytest.fixture(autouse=True)
def _reset_hooks() -> Iterator[None]:
    with _reset_hooks_scope():
        yield


def test_the_reset_scope_restores_a_pre_installed_hook_rather_than_clearing_it():
    """This file's autouse reset must restore what the process had, not
    clear everything: a bare ``set_connector_team_hooks()`` drops any hook
    installed before this file ran (its own docstring says so), which is
    what the newer suites in this repo use ``snapshot_connector_team_hooks``
    to avoid. Asserted directly against the extracted scope rather than
    from inside a fixture-wrapped test, since the fixture wraps the whole
    test body and so cannot observe its own effect on itself."""
    # No manual cleanup needed here: this whole test body already runs
    # inside the autouse fixture's own ``_reset_hooks_scope()``, which
    # restores whatever was installed before this test to whatever it was
    # before, once this test returns -- a bare ``set_connector_team_hooks()``
    # here would be exactly the clear-everything pattern this fix removes.
    sentinel = lambda *_a, **_k: {}  # noqa: E731
    connector_team_scope.set_connector_team_hooks(access=sentinel)
    with _reset_hooks_scope():
        connector_team_scope.set_connector_team_hooks(access=lambda *_a, **_k: {})
    assert connector_team_scope._connector_access_hook is sentinel


def test_the_reset_scope_releases_the_agent_team_hook_when_the_body_raises():
    """The scope is also used as a plain ``with`` block, not only through
    the autouse fixture, and on that path an exception inside the block
    must not leave the agent-team hook installed for whatever runs next.
    Written against the extracted scope directly, since that is the path
    where the release is not otherwise guaranteed."""
    agent_team_scope.set_agent_team_scope_hook(lambda db, user_id: None)
    with pytest.raises(RuntimeError):
        with _reset_hooks_scope():
            raise RuntimeError("boom inside the block")
    assert agent_team_scope._agent_team_scope_hook is None


def test_team_connector_ids_empty_without_hook_installed():
    assert connector_team_scope.team_connector_hook_installed() is False
    assert connector_team_scope.team_connector_ids(None, team_id=5) == {
        "mcp": set(),
        "custom_api": set(),
    }


def test_team_connector_hook_installed_reflects_presence():
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: {"mcp": set(), "custom_api": set()}
    )
    assert connector_team_scope.team_connector_hook_installed() is True
    # Load-bearing, not teardown: this line is what the assertion below is
    # actually exercising -- that clearing the hook flips the reported
    # presence back to False. The autouse fixture's own snapshot restore
    # still runs after this test regardless, so nothing here is relied on
    # for cleanup.
    connector_team_scope.set_connector_team_hooks()
    assert connector_team_scope.team_connector_hook_installed() is False


def test_team_connector_ids_resolves_none_team_without_calling_hook():
    calls = []

    def _hook(db, *, team_id):
        calls.append(team_id)
        return {"mcp": {1}, "custom_api": set()}

    connector_team_scope.set_connector_team_hooks(team_visibility=_hook)
    assert connector_team_scope.team_connector_ids(None, team_id=None) == {
        "mcp": set(),
        "custom_api": set(),
    }
    assert calls == []


def test_team_hook_invocation_contract():
    """The hook is called exactly once, by keyword, and never for a None team."""
    calls: list[tuple[str, object]] = []

    def _record(db, *, team_id):
        calls.append(("kw", team_id))
        return {"mcp": set(), "custom_api": set()}

    connector_team_scope.set_connector_team_hooks(team_visibility=_record)
    assert connector_team_scope.team_connector_ids(None, team_id=None) == {
        "mcp": set(),
        "custom_api": set(),
    }
    assert calls == []
    connector_team_scope.team_connector_ids(None, team_id=T1)
    assert calls == [("kw", T1)]


def test_team_hook_positional_only_callable_raises():
    """Positive control: a positional-only hook must not type-check
    silently -- the keyword call is what stands between a swapped install and
    an unrelated team's connectors on every run."""

    def _positional_only(db, team_id, /):
        return {"mcp": set(), "custom_api": set()}

    connector_team_scope.set_connector_team_hooks(team_visibility=_positional_only)
    with pytest.raises(TypeError):
        connector_team_scope.team_connector_ids(None, team_id=T1)


# ---------------------------------------------------------------------------
# ConnectorAccess and the access hook slot.
# ---------------------------------------------------------------------------


def test_connector_access_defaults_are_both_false():
    access = connector_team_scope.ConnectorAccess()
    assert access.team_owned is False
    assert access.can_edit is False


def test_resolve_connector_access_returns_an_empty_map_without_a_hook_installed():
    for refs in ([("mcp", 1), ("custom_api", 1), ("mcp", 999)], [("mcp", 1)]):
        assert connector_team_scope.resolve_connector_access(None, 7, refs) == {}


def test_resolve_connector_access_reads_no_ref_at_all_without_a_hook_installed():
    """With no hook installed this seam does no work whatsoever -- it does
    not even look at the refs. Deciding that before normalizing is what
    keeps the "standalone xagent is byte-for-byte unchanged" property
    true: a ref shape only the installing application would ever produce
    must not be able to raise in a deployment that has no application
    installed."""
    assert connector_team_scope._connector_access_hook is None
    assert (
        connector_team_scope.resolve_connector_access(
            None, 7, [("mcp", "not-an-int"), ("custom_api", None), ("mcp",)]
        )
        == {}
    )


def test_resolve_connector_access_asks_no_hook_when_no_ref_needs_one():
    """An installed hook is never called when there is nothing to ask about
    -- an empty ``refs`` collection short-circuits before the hook, the
    same way no hook installed does."""
    calls: list[object] = []

    def _hook(db, user_id, refs):
        calls.append(refs)
        return {}

    connector_team_scope.set_connector_team_hooks(access=_hook)
    assert connector_team_scope.resolve_connector_access(None, 7, []) == {}
    assert calls == []


def test_resolve_connector_access_calls_the_hook_once_with_the_requested_refs():
    calls = []

    def _hook(db, user_id, refs):
        calls.append((db, user_id, refs))
        return {
            ("mcp", 11): connector_team_scope.ConnectorAccess(
                team_owned=True, can_edit=True
            )
        }

    connector_team_scope.set_connector_team_hooks(access=_hook)
    result = connector_team_scope.resolve_connector_access(None, 7, [("mcp", 11)])
    assert result == {
        ("mcp", 11): connector_team_scope.ConnectorAccess(
            team_owned=True, can_edit=True
        )
    }
    assert len(calls) == 1
    called_db, called_user_id, called_refs = calls[0]
    assert (called_db, called_user_id) == (None, 7)
    assert called_refs == frozenset({("mcp", 11)})


def test_resolve_connector_access_a_ref_missing_from_the_answer_means_not_linked():
    """Leaving a ref out of the answer is the only way to say "the caller's
    team does not link this connector" -- distinct from a rejected
    malformed verdict for that same ref."""
    connector_team_scope.set_connector_team_hooks(access=lambda *a: {})
    assert connector_team_scope.resolve_connector_access(None, 7, [("mcp", 11)]) == {}


# ---------------------------------------------------------------------------
# Validation of the access hook's answer shape at the boundary.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malformed_answer",
    [
        "dict-of-fields",
        "connector-delete-decision",
        "tuple",
        "truthy-object-with-right-attrs",
        "none",
        "list",
    ],
)
def test_resolve_connector_access_rejects_a_non_dict_answer(malformed_answer):
    # Built inside the test body, not the parametrize list: a couple of
    # these shapes are instances of types this module defines, and
    # constructing them at collection time would make the whole file
    # uncollectable while those types don't exist yet.
    answer = {
        "dict-of-fields": {"team_owned": True, "can_edit": True},
        "connector-delete-decision": connector_team_scope.ConnectorDeleteDecision(
            team_owned=True, authorized=True
        ),
        "tuple": (True, True),
        "truthy-object-with-right-attrs": SimpleNamespace(
            team_owned=True, can_edit=True
        ),
        "none": None,
        "list": [connector_team_scope.ConnectorAccess(team_owned=True, can_edit=True)],
    }[malformed_answer]

    connector_team_scope.set_connector_team_hooks(access=lambda *a: answer)
    with pytest.raises(ValueError):
        connector_team_scope.resolve_connector_access(None, 7, [("mcp", 11)])


def test_resolve_connector_access_rejects_a_verdict_for_a_connector_nobody_asked_about():
    """A verdict keyed on a ref outside the requested set means the hook
    answered a different question than the one it was asked -- silently
    dropping it would hide that the hook and the caller have gone out of
    sync, so this must fail loudly instead."""
    connector_team_scope.set_connector_team_hooks(
        access=lambda *a: {
            ("mcp", 999): connector_team_scope.ConnectorAccess(
                team_owned=True, can_edit=True
            )
        }
    )
    with pytest.raises(ValueError):
        connector_team_scope.resolve_connector_access(None, 7, [("mcp", 11)])


@pytest.mark.parametrize(
    "connector_type,requested_id,alias_id",
    [
        ("mcp", 1, True),
        ("mcp", 1, 1.0),
        ("mcp", 1, Decimal("1")),
        ("custom_api", 2, 2.0),
        ("custom_api", 1, True),
    ],
    ids=["mcp-bool", "mcp-float", "mcp-decimal", "custom-api-float", "custom-api-bool"],
)
def test_resolve_connector_access_rejects_a_key_whose_id_is_only_equal_to_an_int(
    connector_type, requested_id, alias_id
):
    """``True == 1``, ``1.0 == 1`` and ``Decimal("1") == 1`` in Python, so a
    key carrying any of those in place of the requested connector id would
    pass an ``in``-based membership check against ``requested`` -- and be
    stored as a grant for the connector it merely aliases, not the one it
    actually is. The exact-type check must reject it before membership is
    ever checked."""
    connector_team_scope.set_connector_team_hooks(
        access=lambda *a: {
            (connector_type, alias_id): connector_team_scope.ConnectorAccess(
                team_owned=True, can_edit=True
            )
        }
    )
    with pytest.raises(ValueError, match="not an int"):
        connector_team_scope.resolve_connector_access(
            None, 7, [(connector_type, requested_id)]
        )


@pytest.mark.parametrize(
    "bad_key,requested,expected_message",
    [
        ("mcp", [("mcp", 1)], r"not a \(connector_type, connector_id\) pair"),
        (
            ("mcp", 1, "x"),
            [("mcp", 1)],
            r"not a \(connector_type, connector_id\) pair",
        ),
        ((1, 1), [(1, 1)], "connector type that is not a str"),
    ],
    ids=["not-a-tuple", "wrong-length", "connector-type-not-a-str"],
)
def test_resolve_connector_access_rejects_a_structurally_malformed_answer_key(
    bad_key, requested, expected_message
):
    """A key the seam cannot even read as a ref is rejected before it is
    matched against what was asked. The three shapes differ only in how
    they fail to be a ``(connector_type, connector_id)`` pair -- not a
    tuple at all, a tuple of the wrong length, or a pair whose type half
    is not a ``str`` -- so they share one body and differ only in the key
    and the message it must produce."""
    connector_team_scope.set_connector_team_hooks(
        access=lambda *a: {
            bad_key: connector_team_scope.ConnectorAccess(
                team_owned=True, can_edit=True
            )
        }
    )
    with pytest.raises(ValueError, match=expected_message):
        connector_team_scope.resolve_connector_access(None, 7, requested)


@pytest.mark.parametrize(
    "wrong_value",
    ["dict", "duck-typed", "delete-decision", "none", "true"],
)
def test_resolve_connector_access_rejects_a_verdict_value_that_is_not_a_connector_access(
    wrong_value,
):
    """The key was asked about and the key's shape is fine -- what is
    wrong is the value. A duck-typed object carrying ``team_owned=True``
    and ``can_edit=True`` would satisfy every attribute check below it, so
    the type check is the only thing that stops a hook from answering with
    something that merely resembles a verdict. Built in the body, not the
    parametrize list, because two of these are instances of types this
    module defines."""
    value = {
        "dict": {"team_owned": True, "can_edit": True},
        "duck-typed": SimpleNamespace(team_owned=True, can_edit=True),
        "delete-decision": connector_team_scope.ConnectorDeleteDecision(
            team_owned=True, authorized=True
        ),
        "none": None,
        "true": True,
    }[wrong_value]

    connector_team_scope.set_connector_team_hooks(
        access=lambda *_a: {("mcp", 11): value}
    )
    with pytest.raises(ValueError, match="expected ConnectorAccess values"):
        connector_team_scope.resolve_connector_access(None, 7, [("mcp", 11)])


@pytest.mark.parametrize(
    "bad_team_owned",
    [False, "yes", 1],
    ids=["false", "truthy-string", "truthy-int"],
)
def test_resolve_connector_access_rejects_a_team_owned_that_is_not_true(
    bad_team_owned,
):
    """``team_owned`` must be exactly ``True`` on every verdict that
    reaches a caller -- "not linked" is expressed by leaving the ref out
    of the answer, never by a verdict carrying a falsy or merely-truthy
    ``team_owned``."""
    verdict = connector_team_scope.ConnectorAccess(
        team_owned=bad_team_owned, can_edit=True
    )
    connector_team_scope.set_connector_team_hooks(
        access=lambda *a: {("mcp", 11): verdict}
    )
    with pytest.raises(ValueError):
        connector_team_scope.resolve_connector_access(None, 7, [("mcp", 11)])


def test_resolve_connector_access_rejects_a_bare_connector_access_default():
    """``ConnectorAccess()`` -- the dataclass's own all-``False`` default --
    is rejected the same way: constructing a bare instance must never
    become a legitimate "not linked" answer."""
    connector_team_scope.set_connector_team_hooks(
        access=lambda *a: {("mcp", 11): connector_team_scope.ConnectorAccess()}
    )
    with pytest.raises(ValueError):
        connector_team_scope.resolve_connector_access(None, 7, [("mcp", 11)])


@pytest.mark.parametrize(
    "bad_can_edit",
    ["false", 1, 0],
    ids=["string", "truthy-int", "falsy-int"],
)
def test_resolve_connector_access_rejects_a_can_edit_that_is_not_exactly_bool(
    bad_can_edit,
):
    """``bool`` is a subclass of ``int`` in Python, so ``1``/``0`` would
    pass a truthiness check -- this seam requires an exact ``True``/
    ``False`` instead, since a truthy value is never a legitimate grant."""
    verdict = connector_team_scope.ConnectorAccess(
        team_owned=True, can_edit=bad_can_edit
    )
    connector_team_scope.set_connector_team_hooks(
        access=lambda *a: {("mcp", 11): verdict}
    )
    with pytest.raises(ValueError):
        connector_team_scope.resolve_connector_access(None, 7, [("mcp", 11)])


def test_resolve_connector_access_accepts_linked_but_not_editable():
    """A linked-but-not-editable answer is legal on its own -- the seam does
    not require can_edit to be True just because team_owned is."""
    answer = connector_team_scope.ConnectorAccess(team_owned=True, can_edit=False)
    connector_team_scope.set_connector_team_hooks(
        access=lambda *a: {("mcp", 11): answer}
    )
    assert connector_team_scope.resolve_connector_access(None, 7, [("mcp", 11)]) == {
        ("mcp", 11): answer
    }


# ---------------------------------------------------------------------------
# The typed-failure wrapper.
# ---------------------------------------------------------------------------


def test_resolve_connector_access_or_raise_converts_value_error_to_503():
    def _hook(db, user_id, refs):
        raise ValueError("hook returned garbage")

    connector_team_scope.set_connector_team_hooks(access=_hook)
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        connector_team_scope.resolve_connector_access_or_raise(None, 7, [("mcp", 11)])
    assert excinfo.value.status_code == 503


def test_resolve_connector_access_or_raise_converts_even_with_uncomparable_refs():
    """The failure arm logs the refs it resolved with, and those refs are only
    type-checked statically. A collection whose members do not compare against
    each other must still leave through the typed 503 rather than through a
    TypeError raised by the logging call itself."""

    def _hook(db, user_id, refs):
        raise RuntimeError("hook exploded")

    connector_team_scope.set_connector_team_hooks(access=_hook)
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        connector_team_scope.resolve_connector_access_or_raise(
            None, 7, [("mcp", 11), (5, 12)]
        )
    assert excinfo.value.status_code == 503


def test_resolve_connector_access_or_raise_passes_through_planted_error():
    planted = ConnectorRuntimeError(
        "planted_code", "planted", details={"reason": "planted_reason"}
    )

    def _hook(db, user_id, refs):
        raise planted

    connector_team_scope.set_connector_team_hooks(access=_hook)
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        connector_team_scope.resolve_connector_access_or_raise(None, 7, [("mcp", 11)])
    assert excinfo.value is planted


def test_resolve_connector_access_or_raise_converts_malformed_answer_too():
    """The validator's ValueError for a malformed answer goes through the
    same conversion as any other hook-side failure."""
    connector_team_scope.set_connector_team_hooks(
        access=lambda *a: {
            ("mcp", 11): connector_team_scope.ConnectorAccess(
                team_owned=False, can_edit=True
            )
        }
    )
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        connector_team_scope.resolve_connector_access_or_raise(None, 7, [("mcp", 11)])
    assert excinfo.value.status_code == 503


MALFORMED_REFS = [
    ("mcp", "abc"),
    ("mcp", None),
    ("mcp",),
    ("mcp", 1, 2),
    # An id that WOULD have converted to an int is malformed too, not something
    # to convert quietly. Converting asked the hook about ``("mcp", 11)`` while
    # the caller still held ``("mcp", "11")``, so the answer came back under a
    # key the caller could not look up, and the single-ref wrapper read the
    # miss as "the team does not link it".
    ("mcp", "11"),
    # ``isinstance(True, int)`` is ``True`` in Python and ``("mcp", True)``
    # hashes and compares equal to ``("mcp", 1)``, so tolerating it resolved
    # connector 1's access under another connector's name.
    ("mcp", True),
]


@pytest.mark.parametrize("malformed_ref", MALFORMED_REFS)
def test_resolve_connector_access_or_raise_lets_a_malformed_ref_raise_raw(
    malformed_ref,
):
    """A ref whose id is not already an ``int`` is a defect in the calling
    route, not an outage of the installing application, so it stays the
    ``ValueError``/``TypeError`` it is instead of being converted into the
    seam's retryable 503. A hook is installed here on purpose: it is what
    gives the trailing ``assert calls == []`` its meaning, by showing the
    raise lands before the hook is ever reached."""
    calls: list[object] = []

    def _hook(db, user_id, refs):
        calls.append(refs)
        return {}

    connector_team_scope.set_connector_team_hooks(access=_hook)
    with pytest.raises((ValueError, TypeError)) as excinfo:
        connector_team_scope.resolve_connector_access_or_raise(None, 7, [malformed_ref])
    assert not isinstance(excinfo.value, ConnectorRuntimeError)
    assert calls == []


@pytest.mark.parametrize("malformed_ref", MALFORMED_REFS)
def test_resolve_one_connector_access_or_raise_lets_a_malformed_ref_raise_raw(
    malformed_ref,
):
    """The single-ref wrapper inherits the batch resolver's boundary: it
    wraps the ref and calls the batch form, so a malformed ref surfaces raw
    there too rather than as the seam's 503."""
    calls: list[object] = []

    def _hook(db, user_id, refs):
        calls.append(refs)
        return {}

    connector_team_scope.set_connector_team_hooks(access=_hook)
    with pytest.raises((ValueError, TypeError)) as excinfo:
        connector_team_scope.resolve_one_connector_access_or_raise(
            None, 7, malformed_ref
        )
    assert not isinstance(excinfo.value, ConnectorRuntimeError)
    assert calls == []


@pytest.mark.parametrize("malformed_ref", MALFORMED_REFS)
def test_resolve_connector_access_or_raise_reads_no_ref_at_all_without_a_hook_installed(
    malformed_ref,
):
    """The sibling of
    ``test_resolve_connector_access_reads_no_ref_at_all_without_a_hook_installed``
    for the raising entry point. The two checks above install a hook on
    purpose, so they pin the malformed-ref boundary only for a deployment that
    has an application installed; without this one, nothing pins the other
    deployment, and this entry point raised there while
    ``resolve_connector_access`` returned ``{}`` for the very same ref."""
    assert connector_team_scope._connector_access_hook is None
    assert (
        connector_team_scope.resolve_connector_access_or_raise(None, 7, [malformed_ref])
        == {}
    )


def test_resolve_connector_access_or_raise_still_converts_with_well_formed_refs():
    """The opposite direction of the two checks above, so that widening the
    raw-error path cannot go unnoticed: with the refs well formed, a hook
    that raises the very same ``ValueError`` still becomes the seam's one
    typed 503, carrying its reason."""

    def _hook(db, user_id, refs):
        raise ValueError("hook blew up")

    connector_team_scope.set_connector_team_hooks(access=_hook)
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        connector_team_scope.resolve_connector_access_or_raise(None, 7, [("mcp", 11)])
    assert excinfo.value.status_code == 503
    assert excinfo.value.details["reason"] == "connector_access_resolution_failed"


def test_resolve_one_connector_access_or_raise_returns_what_the_batch_form_resolved():
    """The single-ref wrapper agrees with the batch form it wraps.

    It unwraps with ``.get(ref)``, which is only correct while every answer
    comes back keyed on the ref the caller passed. That holds because
    ``_normalize_connector_refs`` rejects an id that is not already an ``int``
    instead of rewriting it -- back when it coerced, ``("mcp", "11")`` was
    asked about as ``("mcp", 11)`` and this lookup missed, reporting ``None``
    for a connector the hook had granted. The malformed-ref cases above pin
    the rejecting half; this pins that a legitimate ref still round-trips.
    """
    grant = connector_team_scope.ConnectorAccess(team_owned=True, can_edit=True)
    connector_team_scope.set_connector_team_hooks(
        access=lambda db, user_id, requested: {r: grant for r in requested}
    )

    batch = connector_team_scope.resolve_connector_access_or_raise(
        None, 7, [("mcp", 11)]
    )
    one = connector_team_scope.resolve_one_connector_access_or_raise(
        None, 7, ("mcp", 11)
    )

    assert batch == {("mcp", 11): grant}
    assert one == grant, f"batch resolved {batch!r} but the wrapper said {one!r}"


def test_resolve_one_connector_access_or_raise_still_reports_an_unlinked_connector():
    """The other direction, so the check above cannot be satisfied by a
    wrapper that returns a grant for everything: a hook that links nothing
    still unwraps to ``None``."""
    connector_team_scope.set_connector_team_hooks(access=lambda *a: {})
    assert (
        connector_team_scope.resolve_one_connector_access_or_raise(None, 7, ("mcp", 11))
        is None
    )


@pytest.mark.parametrize(
    "entry_point",
    ["resolve_connector_access", "resolve_connector_access_or_raise"],
)
def test_each_entry_point_normalizes_the_refs_exactly_once(entry_point, monkeypatch):
    """Both public entry points share one normalization pass.

    The raising entry point has to normalize above its ``try``, so it cannot
    simply hand ``refs`` down; it hands the normalized set to
    ``_resolve_normalized_connector_access`` instead, which is the same
    function the non-raising entry point calls. Nothing else in the suite can
    see this: a second normalization is idempotent over an already-normalized,
    already-deduplicated set, so every observable result stays identical and
    only the call count changes.
    """
    calls = []
    original = connector_team_scope._normalize_connector_refs

    def _counting(refs):
        calls.append(list(refs))
        return original(refs)

    monkeypatch.setattr(connector_team_scope, "_normalize_connector_refs", _counting)
    connector_team_scope.set_connector_team_hooks(
        access=lambda db, user_id, requested: {
            ref: connector_team_scope.ConnectorAccess(team_owned=True, can_edit=True)
            for ref in requested
        }
    )

    # A duplicated ref rather than a text id: the seam rejects a text id
    # outright now, and this test is about the number of passes, not about
    # what a pass does.
    resolved = getattr(connector_team_scope, entry_point)(
        None, 7, [("mcp", 5), ("mcp", 5)]
    )

    assert len(calls) == 1, f"normalized {len(calls)} times, received {calls}"
    assert resolved == {
        ("mcp", 5): connector_team_scope.ConnectorAccess(team_owned=True, can_edit=True)
    }


# ---------------------------------------------------------------------------
# snapshot_connector_team_hooks and its discovery-based coverage test.
# ---------------------------------------------------------------------------


def _connector_hook_slot_names() -> list[str]:
    return [name for name in vars(connector_team_scope) if name.endswith("_hook")]


def test_connector_hook_slot_names_are_discoverable():
    # Sanity check the enumeration itself finds all five known slots, so
    # the coverage test below is not vacuously true.
    names = _connector_hook_slot_names()
    assert names.count("_connector_deleted_hook") == 1
    assert names.count("_connector_renamed_hook") == 1
    assert names.count("_connector_visibility_hook") == 1
    assert names.count("_team_connector_visibility_hook") == 1
    assert names.count("_connector_access_hook") == 1
    assert len(names) == 5


def test_snapshot_connector_team_hooks_restores_every_slot_by_identity():
    names = _connector_hook_slot_names()
    originals = {name: getattr(connector_team_scope, name) for name in names}

    with connector_team_scope.snapshot_connector_team_hooks():
        for name in names:
            setattr(connector_team_scope, name, lambda *a, **k: None)
        for name in names:
            assert getattr(connector_team_scope, name) is not originals[name]

    for name in names:
        assert getattr(connector_team_scope, name) is originals[name]


def test_snapshot_connector_team_hooks_restores_on_exception():
    names = _connector_hook_slot_names()
    originals = {name: getattr(connector_team_scope, name) for name in names}

    with pytest.raises(RuntimeError):
        with connector_team_scope.snapshot_connector_team_hooks():
            for name in names:
                setattr(connector_team_scope, name, lambda *a, **k: None)
            raise RuntimeError("boom inside the block")

    for name in names:
        assert getattr(connector_team_scope, name) is originals[name]


# ---------------------------------------------------------------------------
# DB-backed fixtures for the checks below.
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _create_user(db: Session, username: str) -> User:
    user = User(username=username, password_hash="hash")
    db.add(user)
    db.flush()
    return user


def _poisoning_hook_by_orm_flush(colliding_user_id: int):
    """A hook that leaves a failed ORM flush on the shared session and then
    raises. A failed flush marks the session's transaction inactive on
    every backend, so any later statement raises ``PendingRollbackError``
    until something rolls back -- which is exactly what the seam's hook
    door must do before the exception leaves the module."""

    def hook(db, *_args, **_kwargs):
        db.add(
            User(id=colliding_user_id, username="flush-poison-dup", password_hash="x")
        )
        db.flush()

    return hook


@pytest.mark.parametrize(
    "slot,invoke",
    [
        (
            "visibility",
            lambda db: connector_team_scope.visible_team_connector_ids(db, 1),
        ),
        (
            "deleted",
            lambda db: connector_team_scope.delete_team_connector(db, 1, "mcp", 1),
        ),
        (
            "renamed",
            lambda db: connector_team_scope.rename_team_connector(
                db, 1, "mcp", 1, "old", "new"
            ),
        ),
    ],
    ids=["visibility-hook", "deleted-hook", "renamed-hook"],
)
def test_every_hook_door_restores_the_session_when_the_hook_fails(
    db_session, slot, invoke
):
    """The session restore lives on the single invocation door rather than
    on the ``*_or_raise`` wrappers, so every slot has it -- including a
    slot added to this module later. These three are the slots whose
    answers this seam does not validate. The two this parametrization
    leaves out are the two whose answers it does validate; they are
    covered by the sister test below, where the hook does not raise at
    all."""
    existing = _create_user(db_session, "already-here")
    db_session.commit()

    with connector_team_scope.snapshot_connector_team_hooks():
        connector_team_scope.set_connector_team_hooks(
            **{slot: _poisoning_hook_by_orm_flush(int(existing.id))}
        )
        # Narrow on purpose: the hook fails by colliding on a primary key,
        # so what leaves the door is the driver's own IntegrityError. A bare
        # ``Exception`` here would also swallow a TypeError from a mis-built
        # ``invoke`` and let a broken setup pass as a passing test.
        with pytest.raises(SQLAlchemyError):
            invoke(db_session)

    # Without the restore this raises PendingRollbackError instead.
    assert db_session.query(User).count() == 1


def _swallowing_poisoning_hook_answering(colliding_user_id: int, answer: object):
    """A hook that leaves a failed ORM flush on the shared session,
    swallows that failure itself, and then answers with a shape the seam's
    own validator rejects.

    The sister of ``_poisoning_hook_by_orm_flush`` above: there the hook
    lets its failure propagate, so the door's ``except`` fires on the hook
    call. Here nothing propagates out of the hook at all -- the door's
    ``except`` fires on the validator's rejection instead, which is the
    other half the restore has to cover.
    """

    def hook(db, *_args, **_kwargs):
        try:
            db.add(
                User(
                    id=colliding_user_id,
                    username="swallowed-poison-dup",
                    password_hash="x",
                )
            )
            db.flush()
        except Exception:
            pass
        return answer

    return hook


@pytest.mark.parametrize(
    "slot,answer,invoke",
    [
        (
            "team_visibility",
            {"mcp": "not-a-set", "custom_api": set()},
            lambda db: connector_team_scope.resolve_team_connector_ids_or_raise(
                db, team_id=T1, log_subject=None
            ),
        ),
        (
            "access",
            {"not-a-ref": object()},
            lambda db: connector_team_scope.resolve_connector_access_or_raise(
                db, 1, [("mcp", 11)]
            ),
        ),
    ],
    ids=["team-visibility-hook", "access-hook"],
)
def test_a_hook_that_swallows_its_failure_and_answers_malformed_restores_too(
    db_session, slot, answer, invoke
):
    """The two slots whose answers this seam validates are the two where
    it can notice a hook that poisoned the shared session without ever
    raising: the hook runs a statement that fails, catches that itself,
    and returns an answer the validator then rejects. A hook can do the
    same on the other three slots, where nothing checks the answer and so
    nothing raises -- see the door's docstring on the shape that stays
    uncovered. The rejection is the seam's own exception, not
    the hook's, so the restore has to sit where it sees both -- inside the
    door, around the validation as well as around the call."""
    existing = _create_user(db_session, "already-here")
    db_session.commit()

    with connector_team_scope.snapshot_connector_team_hooks():
        connector_team_scope.set_connector_team_hooks(
            **{slot: _swallowing_poisoning_hook_answering(int(existing.id), answer)}
        )
        with pytest.raises(ConnectorRuntimeError) as excinfo:
            invoke(db_session)
        assert excinfo.value.status_code == 503
        # The cause is the seam's own rejection of the answer, not the DB
        # failure the hook swallowed: the two failure modes stay apart.
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert not isinstance(excinfo.value.__cause__, SQLAlchemyError)

    # No rollback of our own before this line: the query is the statement
    # that proves the door restored the session, and its count proves the
    # poisoning insert never landed.
    assert db_session.query(User).count() == 1


def _create_mcp(db: Session, name: str, *, owner: User | None = None) -> MCPServer:
    server = MCPServer(
        name=name,
        description=f"{name} description",
        managed="external",
        transport="streamable_http",
        url="https://example.com/mcp",
    )
    db.add(server)
    db.flush()
    if owner is not None:
        db.add(
            UserMCPServer(
                user_id=owner.id,
                mcpserver_id=server.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
        db.flush()
    return server


def _create_custom_api(
    db: Session, name: str, *, owner: User | None = None
) -> CustomApi:
    api = CustomApi(
        name=name,
        description=f"{name} description",
        url="https://example.com/api",
        method="GET",
    )
    db.add(api)
    db.flush()
    if owner is not None:
        db.add(
            UserCustomApi(
                user_id=owner.id,
                custom_api_id=api.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
        db.flush()
    return api


@pytest.fixture()
def seed(db_session: Session):
    c = _create_user(db_session, "run-owner")
    active_own = _create_mcp(db_session, "active-own", owner=c)
    team_s = _create_mcp(db_session, "team-s")
    team_x = _create_mcp(db_session, "team-x")
    capi_own = _create_custom_api(db_session, "capi-own", owner=c)
    a_capi = _create_custom_api(db_session, "a-capi")
    return SimpleNamespace(
        c=c,
        active_own=active_own,
        team_s=team_s,
        team_x=team_x,
        capi_own=capi_own,
        a_capi=a_capi,
    )


def _team_hook(seed):
    def _hook(db, *, team_id):
        if team_id == T1:
            return {"mcp": {int(seed.team_s.id)}, "custom_api": {int(seed.a_capi.id)}}
        if team_id == T2:
            return {"mcp": {int(seed.team_x.id)}, "custom_api": set()}
        return {"mcp": set(), "custom_api": set()}

    return _hook


# ---------------------------------------------------------------------------
# visible_mcp_server_clause is fail-closed on its own terms, not only
# because of how its one production caller happens to be gated today.
# ---------------------------------------------------------------------------


def test_visible_mcp_server_clause_matches_nothing_for_none_owner_even_with_team_ids(
    db_session, seed
):
    """With ``owner_user_id=None``, the clause reduces to the personal arm
    regardless of ``team_mcp_ids`` -- it never matches a team-owned row even
    when one is named, rather than relying on a caller to have already
    checked identity first."""
    from xagent.web.services.connector_team_scope import visible_mcp_server_clause

    clause = visible_mcp_server_clause(None, {int(seed.team_s.id)})
    matches = db_session.query(MCPServer).filter(clause).all()
    assert matches == []


# ---------------------------------------------------------------------------
# Keyed on the agent's team; the run owner's own membership is irrelevant.
# Parameterised over three run-owner states, each encoded by the
# agent-team-scope hook -- a negative control for a runner-keyed
# implementation: if visibility were keyed on the runner instead of the
# governing agent, this would flip for the T2 and no-team parameters.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_team", [T2, T1, None])
async def test_scope_keys_on_agent_team_not_runner(db_session, seed, owner_team):
    connector_team_scope.set_connector_team_hooks(team_visibility=_team_hook(seed))
    if owner_team is not None:
        agent_team_scope.set_agent_team_scope_hook(
            lambda db, user_id, _team=owner_team: agent_team_scope.AgentTeamScope(
                team_id=_team, is_team_admin=False
            )
        )
    cfg = WebToolConfig(
        db=db_session,
        request=None,
        user_id=int(seed.c.id),
        connector_team_id=T1,
        include_mcp_tools=True,
    )
    configs = await cfg._load_mcp_server_configs()
    assert {c["name"] for c in configs} == {
        seed.active_own.name,
        seed.team_s.name,
    }


# ---------------------------------------------------------------------------
# A checkout that adopts this revision without installing the new team hook
# is unchanged on both read points, for both connector kinds.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_visibility_hook_alone_is_unchanged(db_session, seed):
    connector_team_scope.set_connector_team_hooks(
        # Also matches ``T1``: a hypothetical implementation that put the
        # fallback inside the shared helper instead of at this one read
        # point would call this hook with the *team* id misread as a user
        # id. Real user ids and team ids are unrelated dense integers, so
        # that collision isn't guaranteed by construction -- matching T1
        # here makes the assertion below deterministic rather than leaving
        # it to an incidental id coincidence.
        visibility=lambda db, user_id: (
            {"mcp": {int(seed.team_s.id)}, "custom_api": {int(seed.a_capi.id)}}
            if user_id in (int(seed.c.id), T1)
            else {"mcp": set(), "custom_api": set()}
        )
    )
    assert connector_team_scope.team_connector_hook_installed() is False

    # The tool loader consults no hook today and must not widen.
    cfg = WebToolConfig(
        db=db_session,
        request=None,
        user_id=int(seed.c.id),
        connector_team_id=T1,
        include_mcp_tools=True,
    )
    configs = await cfg._load_mcp_server_configs()
    assert {c["name"] for c in configs} == {seed.active_own.name}

    # The runtime-connector loader keeps exactly today's answer via the
    # fallback, for both connector kinds.
    visible = _load_visible_runtime_connectors(
        db_session, user_id=int(seed.c.id), agent_team_id=T1
    )
    mcp_ids = {r.connector_id for r in visible if r.connector_type == "mcp"}
    capi_ids = {r.connector_id for r in visible if r.connector_type == "custom_api"}
    assert mcp_ids == {int(seed.active_own.id), int(seed.team_s.id)}
    assert capi_ids == {int(seed.capi_own.id), int(seed.a_capi.id)}


# ---------------------------------------------------------------------------
# A personal agent (no governing team) resolves no team custom API -- the
# custom-API twin of the MCP-side check with the same shape.
# ---------------------------------------------------------------------------


def test_personal_agent_gets_no_team_custom_api(db_session, seed):
    connector_team_scope.set_connector_team_hooks(team_visibility=_team_hook(seed))
    visible = _load_visible_runtime_connectors(
        db_session, user_id=int(seed.c.id), agent_team_id=None
    )
    capi_ids = {r.connector_id for r in visible if r.connector_type == "custom_api"}
    assert capi_ids == {int(seed.capi_own.id)}


# ---------------------------------------------------------------------------
# The fallback selects on hook presence, never on an empty answer: an
# installed hook legitimately answering "this team owns nothing" must not
# be silently overridden by the legacy runner-keyed hook.
# ---------------------------------------------------------------------------


def test_installed_hook_returning_empty_does_not_fall_back(db_session, seed):
    connector_team_scope.set_connector_team_hooks(
        visibility=lambda db, user_id: (
            {"mcp": {int(seed.team_s.id)}, "custom_api": {int(seed.a_capi.id)}}
            if user_id == int(seed.c.id)
            else {"mcp": set(), "custom_api": set()}
        ),
        team_visibility=lambda db, *, team_id: {"mcp": set(), "custom_api": set()},
    )
    visible = _load_visible_runtime_connectors(
        db_session, user_id=int(seed.c.id), agent_team_id=T1
    )
    mcp_ids = {r.connector_id for r in visible if r.connector_type == "mcp"}
    capi_ids = {r.connector_id for r in visible if r.connector_type == "custom_api"}
    assert mcp_ids == {int(seed.active_own.id)}
    assert capi_ids == {int(seed.capi_own.id)}


# ---------------------------------------------------------------------------
# Installing a team-keyed hook fully supersedes the legacy user-keyed
# overlay for every resolution, including a run with no governing agent.
# This is deliberate, not an oversight: falling back to the legacy overlay
# whenever there is no governing agent would re-introduce runner-keyed
# visibility for exactly the population this design excludes -- most
# visibly, a personal agent would inherit its own owner's team connectors,
# which the personal-agent checks above exist to forbid. It is also the
# actual configuration a deployment that installs both the legacy and the
# new hook together runs with on every agent-less resolution, not a
# hypothetical corner case.
# ---------------------------------------------------------------------------


def test_installed_hook_with_no_governing_agent_supersedes_legacy_overlay(
    db_session, seed
):
    connector_team_scope.set_connector_team_hooks(
        visibility=lambda db, user_id: (
            {"mcp": {int(seed.team_s.id)}, "custom_api": {int(seed.a_capi.id)}}
            if user_id == int(seed.c.id)
            else {"mcp": set(), "custom_api": set()}
        ),
        team_visibility=_team_hook(seed),
    )
    visible = _load_visible_runtime_connectors(
        db_session, user_id=int(seed.c.id), agent_team_id=None
    )
    mcp_ids = {r.connector_id for r in visible if r.connector_type == "mcp"}
    capi_ids = {r.connector_id for r in visible if r.connector_type == "custom_api"}
    # Personal-only on both connector kinds: seed.team_s / seed.a_capi
    # (the legacy hook's answer) do NOT appear, even though the legacy
    # hook alone would have granted them.
    assert mcp_ids == {int(seed.active_own.id)}
    assert capi_ids == {int(seed.capi_own.id)}


# ---------------------------------------------------------------------------
# The new-hook branch unions both connector kinds: a team hook's
# "custom_api" grant is consumed at this seam exactly like its "mcp" grant.
# Custom API is now team-keyed on both sides of this seam -- the tool-build
# loaders (WebToolConfig's custom-API paths) are team-keyed too, so a
# team-owned custom API entering a task's runtime selection snapshot always
# has a personal-or-team-satisfied runtime-view resolution and a tool
# loader able to build it. This is the same shape as the legacy branch
# above (test_legacy_visibility_hook_alone_is_unchanged), which already
# grants team custom APIs through the legacy user-keyed hook -- the two
# branches now agree on custom API instead of diverging.
# ---------------------------------------------------------------------------


def test_new_hook_branch_unions_team_custom_api_too(db_session, seed):
    connector_team_scope.set_connector_team_hooks(team_visibility=_team_hook(seed))
    visible = _load_visible_runtime_connectors(
        db_session, user_id=int(seed.c.id), agent_team_id=T1
    )
    mcp_ids = {r.connector_id for r in visible if r.connector_type == "mcp"}
    capi_ids = {r.connector_id for r in visible if r.connector_type == "custom_api"}
    # T1's hook (see _team_hook above) grants both seed.team_s (mcp) and
    # seed.a_capi (custom_api). Both grants union in now.
    assert mcp_ids == {int(seed.active_own.id), int(seed.team_s.id)}
    assert capi_ids == {int(seed.capi_own.id), int(seed.a_capi.id)}


# ---------------------------------------------------------------------------
# The factory-runtime prefetch plan actually carries the agent's team
# id. Without this pin, the custom-API prefetch path silently resolves
# personal-only for every team agent while every other custom-API invariant
# stays green because none of them build the plan.
# ---------------------------------------------------------------------------


def test_factory_runtime_plan_carries_agent_team_id(db_session, seed):
    cfg = WebToolConfig(
        db=db_session,
        request=None,
        user_id=int(seed.c.id),
        connector_team_id=T1,
    )
    plan = cfg._build_factory_runtime_load_plan()
    assert plan.connector_team_id == T1 == cfg._connector_team_id


# ---------------------------------------------------------------------------
# Shape validation on the team-visibility hook's answer. This is an
# authorization input -- a malformed answer must fail loudly (raise), never
# be normalized, coerced, or defaulted to empty. Extra keys are accepted and
# ignored: only "mcp" and "custom_api" are ever read.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malformed_answer",
    [
        None,
        "not-a-dict",
        {"mcp": set()},  # missing "custom_api"
        {"custom_api": set()},  # missing "mcp"
        {"mcp": [1, 2], "custom_api": set()},  # list, not a set
        {"mcp": {"1", "2"}, "custom_api": set()},  # set of strings, not ints
        {"mcp": {True}, "custom_api": set()},  # bool, not accepted as int
    ],
    ids=[
        "none",
        "non-dict",
        "missing-custom_api",
        "missing-mcp",
        "list-not-set",
        "set-of-strings",
        "set-of-bools",
    ],
)
def test_team_connector_ids_raises_on_malformed_hook_answer(malformed_answer):
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: malformed_answer
    )
    with pytest.raises(ValueError):
        connector_team_scope.team_connector_ids(None, team_id=T1)


def test_team_connector_ids_accepts_and_ignores_extra_keys():
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: {
            "mcp": {1, 2},
            "custom_api": {3},
            # An unknown extra key with a value that would itself be
            # malformed if it were ever inspected -- proves the validator
            # only probes "mcp"/"custom_api" and does not iterate every key.
            "unexpected_extra_key": object(),
        }
    )
    result = connector_team_scope.team_connector_ids(None, team_id=T1)
    assert result["mcp"] == {1, 2}
    assert result["custom_api"] == {3}


@pytest.mark.asyncio
async def test_mcp_loader_seam_retypes_malformed_hook_answer(db_session, seed):
    # A hook that (through a type coercion bug on the application side)
    # returns the "mcp" id set as a string instead of a set. Without shape
    # validation this is a SQLite type-affinity fail-open case: the string
    # is silently iterated into single-character values rather than raising.
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: {"mcp": "12", "custom_api": set()}
    )
    cfg = WebToolConfig(
        db=db_session,
        request=None,
        user_id=int(seed.c.id),
        connector_team_id=T1,
        include_mcp_tools=True,
    )
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        await cfg._load_mcp_server_configs()
    assert excinfo.value.status_code == 503
    assert excinfo.value.details["reason"] == "team_scope_resolution_failed"
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_runtime_view_seam_retypes_malformed_hook_answer(db_session, seed):
    task = Task(
        user_id=seed.c.id,
        title="malformed hook runtime task",
        status=TaskStatus.PENDING,
        source="sdk",
        connector_runtime_selected_refs=[
            {"connector_type": "mcp", "connector_id": int(seed.active_own.id)}
        ],
    )
    db_session.add(task)
    db_session.flush()

    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: {"mcp": "12", "custom_api": set()}
    )
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        _load_custom_api_runtime_view_sync(
            db_session,
            task_id=str(task.id),
            connector_runtime_turn_id=None,
            user_id=int(seed.c.id),
            agent_team_id=T1,
        )
    assert excinfo.value.status_code == 503
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_resolve_or_raise_passes_a_typed_error_through_unchanged():
    """The shared wrap's ``except ConnectorRuntimeError: raise`` arm: a hook
    that already raises the typed error must reach the caller as that exact
    object -- not re-wrapped, not given a new cause -- so an inner seam's
    more specific reason survives to whatever renders the failure."""
    planted = ConnectorRuntimeError(
        "planted_code",
        "planted typed failure",
        details={"reason": "planted_inner_reason"},
        status_code=503,
    )

    def _raising_hook(db, *, team_id):
        raise planted

    connector_team_scope.set_connector_team_hooks(team_visibility=_raising_hook)
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        connector_team_scope.resolve_team_connector_ids_or_raise(
            None, team_id=T1, log_subject="passthrough-probe"
        )
    assert excinfo.value is planted
    assert excinfo.value.details["reason"] == "planted_inner_reason"


# ---------------------------------------------------------------------------
# The team-visibility wrapper restores the shared session after a failed
# hook too -- the sister guarantee to resolve_connector_access_or_raise's,
# on the sister wrapper.
# ---------------------------------------------------------------------------


def test_the_team_scope_wrapper_also_restores_the_session(db_session):
    """A hook that poisons the shared session via a failed ORM flush, then
    lets that failure propagate, must not leave the session unusable for
    whatever runs next in the same request."""
    poisoning_user_id = 900001
    db_session.add(
        User(id=poisoning_user_id, username="team-scope-poison", password_hash="x")
    )
    db_session.commit()

    def poisoning_team_visibility(db, *, team_id):
        # A duplicate primary key -- a real ORM flush failure, not a
        # simulated one -- propagates out of this hook uncaught.
        db.add(User(id=poisoning_user_id, username="dup", password_hash="x"))
        db.flush()
        return {"mcp": set(), "custom_api": set()}  # pragma: no cover - unreachable

    connector_team_scope.set_connector_team_hooks(
        team_visibility=poisoning_team_visibility
    )
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        connector_team_scope.resolve_team_connector_ids_or_raise(
            db_session, team_id=T1, log_subject=None
        )
    assert excinfo.value.status_code == 503

    # The session must be usable again immediately afterward.
    result = db_session.execute(select(1)).scalar()
    assert result == 1


# ---------------------------------------------------------------------------
# The access wrapper restores the shared session after a hook that raises
# outright, on a real session rather than the ``None`` stand-in the
# conversion tests use.
# ---------------------------------------------------------------------------


def test_the_access_wrapper_restores_the_session_when_the_hook_raises(db_session):
    """The conversion tests above hand this wrapper ``None`` for the session,
    so they say nothing about the restore. A hook that poisons the shared
    session via a failed ORM flush and lets that failure propagate must leave
    the session usable for whatever runs next in the same request, and the
    typed error must still carry the hook's own failure as its cause -- which
    is what tells this failure mode apart from a rejected answer."""
    poisoning_user_id = 900002
    db_session.add(
        User(id=poisoning_user_id, username="access-poison", password_hash="x")
    )
    db_session.commit()

    def poisoning_access(db, user_id, refs):
        # A duplicate primary key -- a real ORM flush failure, not a
        # simulated one -- propagates out of this hook uncaught.
        db.add(User(id=poisoning_user_id, username="dup", password_hash="x"))
        db.flush()
        return {}  # pragma: no cover - unreachable

    connector_team_scope.set_connector_team_hooks(access=poisoning_access)
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        connector_team_scope.resolve_connector_access_or_raise(
            db_session, 7, [("mcp", 11)]
        )
    assert excinfo.value.status_code == 503
    assert isinstance(excinfo.value.__cause__, SQLAlchemyError)

    # The session must be usable again immediately afterward.
    assert db_session.execute(select(1)).scalar() == 1


# ---------------------------------------------------------------------------
# The session boundary check on the hook gate: a hook must not end the
# caller's own transaction on a call site that asked for the check, and
# must be left alone entirely on one that did not.
# ---------------------------------------------------------------------------


def _open_transaction(db: Session) -> None:
    """Put ``db`` in a root transaction without writing anything, so a test
    controls whether the gate sees "already in a transaction" on entry
    without that state depending on unrelated setup calls."""
    db.execute(select(1))


def test_a_hook_that_returns_the_connection_is_refused(db_session):
    """``release_db_connection_if_clean`` (``models/database.py``) rolls
    back and returns the connection whenever the session has no pending
    writes -- exactly the state a hook is usually called in, since every
    call site asks its hook before the route's own ``commit()``. A hook
    that reuses it ends this session's own transaction just as surely as
    one that calls ``rollback()`` directly."""
    server = _create_mcp(db_session, "release-helper-probe")
    db_session.commit()

    def hook(db, *_a, **_k):
        assert release_db_connection_if_clean(db) is True
        return connector_team_scope.ConnectorDeleteDecision()

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    with pytest.raises(ConnectorHookSessionBoundaryError):
        connector_team_scope.delete_team_connector(
            db_session, 1, "mcp", int(server.id), caller_holds_lock=True
        )

    refreshed = db_session.get(MCPServer, server.id)
    assert refreshed is not None
    assert refreshed.name == "release-helper-probe"


@pytest.mark.parametrize(
    "hook_body",
    [
        pytest.param(lambda db: db.rollback(), id="bare-rollback"),
        pytest.param(
            lambda db: (db.rollback(), db.execute(text("select 1"))),
            id="rollback-then-a-text-statement",
        ),
        pytest.param(
            lambda db: (
                db.rollback(),
                db.add(User(username="hook-own-write", password_hash="x")),
                db.flush(),
            ),
            id="rollback-then-the-hooks-own-write",
        ),
        pytest.param(
            lambda db: (db.rollback(), db.query(User).all()),
            id="rollback-then-one-orm-read",
        ),
    ],
)
def test_every_way_of_ending_the_transaction_is_refused(db_session, hook_body):
    """Four different ways for a hook to end this session's own root
    transaction, all refused the same way. The last three rule out
    inferring the violation from the existing write-flag
    (``xagent_txn_may_have_written``) or from ``Session.in_transaction()``:
    a rollback followed by a write or a text statement starts a fresh
    transaction and can leave either signal looking clean, and a rollback
    followed by a bare ORM read does the same to ``in_transaction()``."""
    server = _create_mcp(db_session, "every-way-probe")
    db_session.commit()

    def hook(db, *_a, **_k):
        hook_body(db)
        return connector_team_scope.ConnectorDeleteDecision()

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    with pytest.raises(ConnectorHookSessionBoundaryError):
        connector_team_scope.delete_team_connector(
            db_session, 1, "mcp", int(server.id), caller_holds_lock=True
        )


def test_a_savepoint_the_hook_commits_is_not_a_violation(db_session):
    """A savepoint the hook opens and commits is the supported way for a
    hook to recover from a failure of its own. It does not end the
    caller's own transaction, so it must not be reported as a violation."""
    server = _create_mcp(db_session, "savepoint-commit-probe")
    db_session.commit()

    def hook(db, *_a, **_k):
        nested = db.begin_nested()
        db.add(User(username="within-committed-savepoint", password_hash="x"))
        db.flush()
        nested.commit()
        return connector_team_scope.ConnectorDeleteDecision()

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    connector_team_scope.delete_team_connector(
        db_session, 1, "mcp", int(server.id), caller_holds_lock=True
    )
    assert db_session.in_transaction()


def test_a_savepoint_the_hook_rolls_back_is_not_a_violation(db_session):
    """The rollback twin of the commit case above: opening and rolling
    back a savepoint is also a hook recovering from its own failure, not
    an end to the caller's own transaction."""
    server = _create_mcp(db_session, "savepoint-rollback-probe")
    db_session.commit()

    def hook(db, *_a, **_k):
        nested = db.begin_nested()
        db.add(User(username="within-rolled-back-savepoint", password_hash="x"))
        db.flush()
        nested.rollback()
        return connector_team_scope.ConnectorDeleteDecision()

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    connector_team_scope.delete_team_connector(
        db_session, 1, "mcp", int(server.id), caller_holds_lock=True
    )
    assert db_session.in_transaction()


def test_a_hook_that_writes_and_flushes_is_not_a_violation(db_session):
    """The ``deleted`` and ``renamed`` slots exist so a hook can write on
    the caller's session and ``flush()`` without ending its transaction --
    exactly this shape must pass."""
    server = _create_mcp(db_session, "write-flush-probe")
    db_session.commit()

    def hook(db, *_a, **_k):
        db.add(User(username="hook-flush-write", password_hash="x"))
        db.flush()
        return connector_team_scope.ConnectorDeleteDecision()

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    connector_team_scope.delete_team_connector(
        db_session, 1, "mcp", int(server.id), caller_holds_lock=True
    )
    assert db_session.in_transaction()


def test_a_hook_that_raises_is_not_reported_as_a_boundary_violation(db_session):
    """A hook that raises is not a contract violation -- the seam restores
    the session and propagates the failure as-is. The restore itself is an
    unconditional rollback and so moves the count on its own; comparing on
    that path (rather than only on the success path) would read every
    ordinary hook failure as a boundary violation instead of what it
    actually is."""
    server = _create_mcp(db_session, "raises-probe")
    db_session.commit()
    planted = ConnectorRuntimeError("planted_code", "planted failure", status_code=503)

    def hook(db, *_a, **_k):
        raise planted

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        connector_team_scope.delete_team_connector(
            db_session, 1, "mcp", int(server.id), caller_holds_lock=True
        )
    assert excinfo.value is planted


def test_an_undeclared_call_site_is_not_checked(db_session):
    """``caller_holds_lock`` defaults to ``False``. A call site that does
    not pass it is not checked at all, even when the hook ends the
    caller's own transaction."""
    server = _create_mcp(db_session, "undeclared-probe")
    db_session.commit()

    def hook(db, *_a, **_k):
        db.rollback()
        return connector_team_scope.ConnectorDeleteDecision()

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    result = connector_team_scope.delete_team_connector(
        db_session, 1, "mcp", int(server.id)
    )
    assert result == connector_team_scope.ConnectorDeleteDecision()


def test_a_reused_session_does_not_turn_the_second_undeclared_call_into_a_violation(
    db_session,
):
    """Two undeclared calls on the same session, a loading query in
    between -- the shape a cached hook-config session takes across two
    requests. The second call must not become a violation just because
    the session it was handed happens to already be in a transaction on
    entry; the decision is per call site, not inferred from session state."""
    first_server = _create_mcp(db_session, "reused-session-first")
    second_server = _create_mcp(db_session, "reused-session-second")
    db_session.commit()

    calls: list[str] = []

    def clean_hook(db, *_a, **_k):
        calls.append("clean")
        return connector_team_scope.ConnectorDeleteDecision()

    def ending_hook(db, *_a, **_k):
        calls.append("ending")
        db.rollback()
        return connector_team_scope.ConnectorDeleteDecision()

    connector_team_scope.set_connector_team_hooks(deleted=clean_hook)
    connector_team_scope.delete_team_connector(
        db_session, 1, "mcp", int(first_server.id)
    )

    # A loading query between the two calls, same as the route work a real
    # request does -- and what leaves the session already in a transaction
    # by the time the second call is made.
    db_session.query(MCPServer).filter(MCPServer.id == second_server.id).first()

    connector_team_scope.set_connector_team_hooks(deleted=ending_hook)
    connector_team_scope.delete_team_connector(
        db_session, 1, "mcp", int(second_server.id)
    )

    assert calls == ["clean", "ending"]


def test_a_declared_call_site_is_checked(db_session):
    """The positive twin of the undeclared test above: the same
    transaction-ending hook, on a call site that does pass
    ``caller_holds_lock=True``, is refused."""
    server = _create_mcp(db_session, "declared-probe")
    db_session.commit()

    def hook(db, *_a, **_k):
        db.rollback()
        return connector_team_scope.ConnectorDeleteDecision()

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    with pytest.raises(ConnectorHookSessionBoundaryError):
        connector_team_scope.delete_team_connector(
            db_session, 1, "mcp", int(server.id), caller_holds_lock=True
        )


def test_the_violation_log_line_names_the_slot(db_session, caplog):
    """The violation branch's log line names which of the five slots the
    call belongs to, not only the hook's own name: an operator reading it
    should not have to guess which slot the offending hook was
    installed on."""
    server = _create_mcp(db_session, "slot-in-log-probe")
    db_session.commit()

    def hook(db, *_a, **_k):
        db.rollback()
        return connector_team_scope.ConnectorDeleteDecision()

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    with caplog.at_level(logging.ERROR, logger=connector_team_scope.__name__):
        with pytest.raises(ConnectorHookSessionBoundaryError):
            connector_team_scope.delete_team_connector(
                db_session, 1, "mcp", int(server.id), caller_holds_lock=True
            )
    assert "for the deleted slot" in caplog.text


def test_a_duck_typed_session_skips_the_check_and_still_calls_the_hook(caplog):
    """A duck-typed object with no ``.info`` cannot report a count at all
    (``root_transaction_end_count`` returns ``None`` for it -- see
    ``tests/web/test_release_db_connection.py``), so the gate has nothing
    to compare and calls the hook exactly as it would with the check off,
    rather than failing on an object it cannot inspect. The skip is not
    silent: a skipped check and a passed check would otherwise look
    identical to whoever is operating this, so the gate logs it."""

    class _DuckSession:
        pass

    duck = _DuckSession()
    calls: list[object] = []

    def hook(db, *_a, **_k):
        calls.append(db)
        return connector_team_scope.ConnectorDeleteDecision()

    connector_team_scope.set_connector_team_hooks(deleted=hook)
    with caplog.at_level(logging.DEBUG, logger=connector_team_scope.__name__):
        result = connector_team_scope.delete_team_connector(
            duck, 1, "mcp", 1, caller_holds_lock=True
        )
    assert calls == [duck]
    assert result == connector_team_scope.ConnectorDeleteDecision()
    assert "session boundary check skipped" in caplog.text


def test_a_check_that_raises_refuses_after_restoring_the_session(
    db_session, monkeypatch
):
    """A hook that replaces ``.info`` with something that is not a mapping
    at all makes the post-call read itself raise. That failure must still
    refuse the request through the ordinary restore-then-reraise path, not
    skip the restore because the failure came from the check rather than
    from the hook."""
    server = _create_mcp(db_session, "corrupt-info-probe")
    db_session.commit()

    restored: list[object] = []
    original_restore = connector_team_scope._restore_session_after_hook_failure

    def spy_restore(db):
        restored.append(db)
        original_restore(db)

    monkeypatch.setattr(
        connector_team_scope, "_restore_session_after_hook_failure", spy_restore
    )

    def hook(db, *_a, **_k):
        db.info = "not-a-mapping"
        return connector_team_scope.ConnectorDeleteDecision()

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    with pytest.raises(AttributeError):
        connector_team_scope.delete_team_connector(
            db_session, 1, "mcp", int(server.id), caller_holds_lock=True
        )
    assert len(restored) == 1


def test_a_refused_delete_leaves_both_rows_in_place(db_session, monkeypatch):
    """A refused delete must have zero effect on committed state: the
    definition row and the caller's own ownership link both survive, the
    same as if the hook had never been called at all. That survival alone
    does not prove the violation branch restored the session -- this
    hook's own ``rollback()`` already leaves the session clean, so the two
    rows would still be there even if the branch skipped its restore. The
    spy below pins that restore down directly."""
    owner = _create_user(db_session, "delete-refusal-owner")
    server = _create_mcp(db_session, "refused-delete-probe", owner=owner)
    db_session.commit()

    restored: list[object] = []
    original_restore = connector_team_scope._restore_session_after_hook_failure

    def spy_restore(db):
        restored.append(db)
        original_restore(db)

    monkeypatch.setattr(
        connector_team_scope, "_restore_session_after_hook_failure", spy_restore
    )

    def hook(db, *_a, **_k):
        db.rollback()
        return connector_team_scope.ConnectorDeleteDecision()

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    with pytest.raises(ConnectorHookSessionBoundaryError):
        connector_team_scope.delete_team_connector(
            db_session, int(owner.id), "mcp", int(server.id), caller_holds_lock=True
        )

    # The refusal restores the session before it raises: without this the
    # violation branch's restore can be deleted and nothing goes red.
    assert len(restored) == 1
    assert db_session.get(MCPServer, server.id) is not None
    assert (
        db_session.query(UserMCPServer)
        .filter(UserMCPServer.mcpserver_id == server.id)
        .first()
        is not None
    )


def test_a_refusal_does_not_undo_what_the_hook_already_committed(db_session):
    """Restoring the session recovers a hook's uncommitted work only. A
    hook that ends the transaction by committing its own write first
    leaves that write durable -- the refusal that follows cannot and must
    not undo it. This is stated behavior, not a silent counterexample: it
    is exactly why the contract forbids ``commit()`` outright rather than
    treating it as something the check cleans up after."""
    server = _create_mcp(db_session, "hook-committed-probe")
    db_session.commit()

    def hook(db, *_a, **_k):
        db.add(User(username="hook-committed-row", password_hash="x"))
        db.commit()
        return connector_team_scope.ConnectorDeleteDecision()

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    with pytest.raises(ConnectorHookSessionBoundaryError):
        connector_team_scope.delete_team_connector(
            db_session, 1, "mcp", int(server.id), caller_holds_lock=True
        )

    assert (
        db_session.query(User).filter(User.username == "hook-committed-row").first()
        is not None
    )


def test_a_hook_that_ends_the_transaction_and_answers_a_rejected_shape_is_a_validation_failure(
    db_session,
):
    """A hook can both end the caller's transaction and answer a shape the
    validator rejects. ``validate`` runs right after the hook call, above
    the boundary comparison, so this is refused as the rejected answer --
    ``ConnectorRuntimeError`` -- not as ``ConnectorHookSessionBoundaryError``.
    Reordering the two checks would turn this into a boundary violation
    instead."""

    def hook(db, *_a, **_k):
        db.rollback()
        return None  # not a dict -- the access answer validator rejects this

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(access=hook)
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        connector_team_scope.resolve_connector_access_or_raise(
            db_session, 1, [("mcp", 1)], caller_holds_lock=True
        )
    assert excinfo.value.details == {"reason": "connector_access_resolution_failed"}

    # The session must be usable again immediately afterward.
    assert db_session.execute(select(1)).scalar() == 1


def test_two_hook_calls_in_one_request_are_compared_one_at_a_time(db_session):
    """Two declared calls in the same request, the first clean and the
    second a violation. The count is compared once per call, not against
    a single value captured once for the whole request -- only the second
    call is refused."""
    first_server = _create_mcp(db_session, "two-calls-first")
    second_server = _create_mcp(db_session, "two-calls-second")
    db_session.commit()

    def clean_hook(db, *_a, **_k):
        return connector_team_scope.ConnectorDeleteDecision()

    def ending_hook(db, *_a, **_k):
        db.rollback()
        return connector_team_scope.ConnectorDeleteDecision()

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=clean_hook)
    connector_team_scope.delete_team_connector(
        db_session, 1, "mcp", int(first_server.id), caller_holds_lock=True
    )

    connector_team_scope.set_connector_team_hooks(deleted=ending_hook)
    with pytest.raises(ConnectorHookSessionBoundaryError):
        connector_team_scope.delete_team_connector(
            db_session, 1, "mcp", int(second_server.id), caller_holds_lock=True
        )


def test_a_non_count_written_after_the_transaction_ended_is_refused(db_session):
    """A hook that ends the transaction and then leaves a non-count value
    in the counter key must still be refused for ending the transaction --
    the read that raises must not be mistaken for "nothing to compare"
    and skipped. The refusal here travels the read-raises-``TypeError``
    path rather than the boundary-violation path -- both refuse, but they
    log differently."""
    server = _create_mcp(db_session, "post-end-bad-value-probe")
    db_session.commit()

    def hook(db, *_a, **_k):
        db.rollback()
        db.info[_ROOT_TXN_END_COUNT_KEY] = True
        return connector_team_scope.ConnectorDeleteDecision()

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    with pytest.raises(TypeError):
        connector_team_scope.delete_team_connector(
            db_session, 1, "mcp", int(server.id), caller_holds_lock=True
        )


def test_a_non_count_already_in_the_key_refuses_before_the_hook_runs(db_session):
    """A non-count value already sitting in the counter key before the
    hook is ever called must refuse the request without calling the hook
    at all -- there is nothing yet to restore, since the hook never ran."""
    server = _create_mcp(db_session, "pre-existing-bad-value-probe")
    db_session.commit()

    calls: list[object] = []

    def hook(db, *_a, **_k):
        calls.append(db)
        return connector_team_scope.ConnectorDeleteDecision()

    _open_transaction(db_session)
    db_session.info[_ROOT_TXN_END_COUNT_KEY] = "garbage"
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    with pytest.raises(TypeError):
        connector_team_scope.delete_team_connector(
            db_session, 1, "mcp", int(server.id), caller_holds_lock=True
        )
    assert calls == []


@pytest.mark.parametrize(
    "raised",
    [
        pytest.param(ValueError("plain hook failure"), id="plain-value-error"),
        pytest.param(
            ConnectorHookSessionBoundaryError("hook raised the boundary error itself"),
            id="the-boundary-error-itself",
        ),
    ],
)
def test_the_session_is_restored_exactly_once_whatever_the_hook_raised(
    db_session, monkeypatch, raised
):
    """The restore must run exactly once regardless of what the hook
    raised, including when the hook raises this module's own boundary
    error directly. Matching on the exception's type instead of tracking
    whether a restore already ran would restore zero times for that
    second case."""
    server = _create_mcp(db_session, "restore-once-probe")
    db_session.commit()

    restore_calls: list[object] = []
    original_restore = connector_team_scope._restore_session_after_hook_failure

    def spy_restore(db):
        restore_calls.append(db)
        original_restore(db)

    monkeypatch.setattr(
        connector_team_scope, "_restore_session_after_hook_failure", spy_restore
    )

    def hook(db, *_a, **_k):
        raise raised

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(deleted=hook)
    with pytest.raises(type(raised)):
        connector_team_scope.delete_team_connector(
            db_session, 1, "mcp", int(server.id), caller_holds_lock=True
        )

    assert len(restore_calls) == 1


# ---------------------------------------------------------------------------
# The application-level handler for the new error, and the access slot's
# pass-through arm.
# ---------------------------------------------------------------------------


def _http_request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "DELETE",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "scheme": "http",
            "server": ("testserver", 80),
        }
    )


def test_the_boundary_handler_is_registered():
    from xagent.web.app import app

    assert app.exception_handlers[ConnectorHookSessionBoundaryError] is (
        connector_hook_session_boundary_error_handler
    )


async def test_the_boundary_handler_answers_500_and_one_detail():
    """500, not 503: 503 announces a transient outage and invites a retry,
    which a hook that ends the caller's transaction is not. The body
    names nothing about the hook -- the operator reads the log line, the
    caller does not."""
    response = await connector_hook_session_boundary_error_handler(
        _http_request("/api/custom-apis/7"),
        ConnectorHookSessionBoundaryError("hook 'leaky_hook' ended the transaction"),
    )
    assert response.status_code == 500
    body = json.loads(response.body)
    assert body == {"detail": "Connector team integration is unavailable."}
    # The hook's name is in the log line, not in what the caller reads.
    assert b"leaky_hook" not in response.body


def test_the_access_slot_lets_the_boundary_error_through_its_wrapper(db_session):
    """``access`` has no call site in this repository today, so this is
    constructed directly against the wrapper rather than through a route.
    The seam's own transient-outage error gets folded into
    ``ConnectorRuntimeError`` by the surrounding ``except Exception``; this
    one must not -- a permanent defect in the installing application's
    code is a different failure than an outage, and folding it in would
    also give this slot a different answer than every other checked slot
    for the same failure."""

    def hook(db, user_id, refs):
        db.rollback()
        return {}

    _open_transaction(db_session)
    connector_team_scope.set_connector_team_hooks(access=hook)
    with pytest.raises(ConnectorHookSessionBoundaryError):
        connector_team_scope.resolve_connector_access_or_raise(
            db_session, 1, [("mcp", 1)], caller_holds_lock=True
        )


@pytest.mark.parametrize("route_name", ["update_custom_api", "delete_custom_api"])
def test_a_bare_call_site_lets_the_error_escape_unchanged(db_session, route_name):
    """Neither custom_api call site sits inside a ``try`` of the route's
    own: nothing there stands between the hook call and the route's
    return. The boundary error reaches a direct caller of the route
    exactly as raised, the same way it reaches the application's own
    exception handler in production."""
    from xagent.web.api.custom_api import (
        CustomApiUpdate,
        delete_custom_api,
        update_custom_api,
    )

    owner = _create_user(db_session, f"bare-call-site-owner-{route_name}")
    api = _create_custom_api(
        db_session, f"bare-call-site-api-{route_name}", owner=owner
    )
    db_session.commit()
    current_user = SimpleNamespace(id=owner.id, is_admin=False)

    def deleted_hook(db, *_a, **_k):
        db.rollback()
        return connector_team_scope.ConnectorDeleteDecision()

    def renamed_hook(db, *_a, **_k):
        db.rollback()

    if route_name == "delete_custom_api":
        connector_team_scope.set_connector_team_hooks(deleted=deleted_hook)
        with pytest.raises(ConnectorHookSessionBoundaryError):
            delete_custom_api(int(api.id), current_user=current_user, db=db_session)
    else:
        connector_team_scope.set_connector_team_hooks(renamed=renamed_hook)
        payload = CustomApiUpdate(name=f"bare-call-site-api-{route_name}-renamed")
        with pytest.raises(ConnectorHookSessionBoundaryError):
            update_custom_api(
                int(api.id), payload, current_user=current_user, db=db_session
            )


def test_a_committing_rename_hook_carries_the_routes_own_field_writes_with_it(
    db_session,
):
    """The route assigns ``api.name`` -- and any other field in the same
    payload -- well before it calls the rename hook, and does not commit
    until after the hook returns. A hook that ends the transaction with
    ``commit()`` instead of raising or rolling back therefore makes those
    staged assignments durable too, before the request is refused. This
    pins the known limitation named in this module's docstring: restoring
    the session after a hook failure recovers only a hook's own
    uncommitted work, never work the route staged ahead of it."""
    from xagent.web.api.custom_api import CustomApiUpdate, update_custom_api

    owner = _create_user(db_session, "committing-rename-hook-owner")
    api = _create_custom_api(db_session, "committing-rename-hook-api", owner=owner)
    db_session.commit()
    current_user = SimpleNamespace(id=owner.id, is_admin=False)

    def renamed_hook(db, *_a, **_k):
        db.commit()

    connector_team_scope.set_connector_team_hooks(renamed=renamed_hook)
    new_name = "committing-rename-hook-api-renamed"
    payload = CustomApiUpdate(name=new_name)
    with pytest.raises(ConnectorHookSessionBoundaryError):
        update_custom_api(
            int(api.id), payload, current_user=current_user, db=db_session
        )

    persisted = db_session.query(CustomApi).filter(CustomApi.id == api.id).one()
    assert persisted.name == new_name


# ---------------------------------------------------------------------------
# T-15: the call site table in the module docstring and the actual call
# sites in the two route modules must agree, in both directions.
# ---------------------------------------------------------------------------

_HOOK_ENTRY_POINTS = {
    "delete_team_connector",
    "rename_team_connector",
    "resolve_one_connector_access_or_raise",
    "resolve_connector_access_or_raise",
}


def _declared_call_sites() -> dict[str, bool]:
    """{"module.function": whether the call passes caller_holds_lock=True}.

    Read off the source of both route modules. A call is attributed to the
    nearest enclosing function definition, and both ``def`` and ``async def``
    count, because some call sites live in coroutines and a scan that only
    walks ``ast.FunctionDef`` silently misses them.
    """
    import ast
    import inspect

    found: dict[str, bool] = {}
    for module_name in ("custom_api", "mcp"):
        module = __import__(f"xagent.web.api.{module_name}", fromlist=[module_name])
        tree = ast.parse(inspect.getsource(module))
        parent: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # A bare name (``delete_team_connector(...)``) and a qualified
            # attribute access (``connector_team_scope.delete_team_connector(...)``)
            # are the same call site under two import styles. Direction two
            # below -- catching a call the table has no row for -- is the
            # only mechanical backstop for the "declare per call site,
            # default off" choice, so missing the qualified form here would
            # silently defeat it for exactly the call it exists to catch.
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            else:
                continue
            if func_name not in _HOOK_ENTRY_POINTS:
                continue
            enclosing = parent.get(node)
            while enclosing is not None and not isinstance(
                enclosing, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                enclosing = parent.get(enclosing)
            assert enclosing is not None, "hook call outside any function"
            declared = any(
                kw.arg == "caller_holds_lock"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            )
            key = f"{module_name}.{enclosing.name}"
            assert key not in found, f"two hook calls attributed to {key}"
            found[key] = declared
    return found


def _table_rows() -> dict[str, bool]:
    """{"module.function": the declaration the module docstring states}."""
    import re

    rows: dict[str, bool] = {}
    pattern = re.compile(r"^\|\s*``([\w.]+)``\s*\|.*\|\s*``(True|False)``\s*\|\s*$")
    for line in (connector_team_scope.__doc__ or "").splitlines():
        match = pattern.match(line.strip())
        if match:
            rows[match.group(1)] = match.group(2) == "True"
    return rows


def test_the_call_site_table_and_the_call_sites_agree():
    table = _table_rows()
    code = _declared_call_sites()
    assert table, "the call site table in the module docstring did not parse"
    # Direction one: nothing in the table may claim a declaration the call
    # site does not make.
    assert {k: v for k, v in table.items() if k in code} == {
        k: v for k, v in code.items() if k in table
    }
    # Direction two: no hook call site may exist without a row. This is the
    # half that catches a new lock-holding call site nobody registered --
    # direction one is blind to it.
    assert set(code) == set(table)
