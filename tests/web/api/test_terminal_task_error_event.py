"""Frame-shape contracts for ``create_terminal_task_error_event``.

Pinned here: the four call sites that pass no ``code`` still get the same
six-key frame, and a ``code`` that survives validation is written onto the
frame under its own key with nothing else alongside it.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any

import pytest

import xagent
from xagent.core.tools.adapters.vibe import (
    connector_runtime as connector_runtime_module,
)
from xagent.web.api.v1.errors import V1ErrorCode
from xagent.web.api.websocket import create_terminal_task_error_event
from xagent.web.services.client_error_messages import (
    CONNECTOR_RUNTIME_CLIENT_ERROR_CODES,
)

BASE_FIELDS = {"type", "message", "task_id", "task", "error", "timestamp"}

# Anchored on a real package file rather than assumed relative to this test
# file: xagent is a namespace package, so it has no single __file__ of its
# own, but the first path entry is the actual source tree to scan.
SRC_ROOT = Path(xagent.__path__[0])
RAISE_CALL_NAMES = {"_raise_runtime_error", "ConnectorRuntimeError"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
    ],
    ids=["neither"],
)
def test_terminal_error_event_shape_unchanged(kwargs: dict[str, Any]) -> None:
    """A caller that passes no code gets the same six-key frame."""

    event = create_terminal_task_error_event(1, "x", **kwargs)

    assert set(event.keys()) == BASE_FIELDS


def test_terminal_error_event_carries_a_valid_code() -> None:
    event = create_terminal_task_error_event(1, "x", code="missing_runtime_context")

    assert set(event.keys()) == BASE_FIELDS | {"code"}
    assert event["code"] == "missing_runtime_context"
    assert "details" not in event


def test_unknown_code_is_dropped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``code`` is checked against ``CONNECTOR_RUNTIME_CLIENT_ERROR_CODES``.

    ``ConnectorRuntimeError`` types its code as a bare ``str`` and stores it
    without validation, so an unlisted value reaching the wire is a question
    of what raise sites happen to exist today, not of what the code enforces.
    """

    with caplog.at_level(logging.ERROR):
        event = create_terminal_task_error_event(
            1,
            "x",
            code="not_a_listed_code",
        )

    assert set(event.keys()) == BASE_FIELDS
    assert "not_a_listed_code" not in json.dumps(event)

    dropped = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR and "dropped=code" in record.getMessage()
    ]
    assert len(dropped) == 1
    assert "not_a_listed_code" in dropped[0]


@pytest.mark.parametrize("code", [["not", "hashable"], 7, object()])
def test_a_non_string_code_is_dropped_without_raising(
    code: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """The type gate runs before the membership test.

    ``ConnectorRuntimeError`` types its code as a bare ``str`` and stores it
    unvalidated, and this builder runs inside an ``except`` that only logs a
    failed broadcast -- so a non-string value must cost the argument, not the
    frame. Only the unhashable case actually needs the gate: without it a list
    raises inside the frozenset membership test, while a hashable non-string
    (an int, a bare object) is simply not a member and already takes the drop
    path. All three are pinned so the outcome is the same shape either way.
    """
    with caplog.at_level(logging.ERROR):
        event = create_terminal_task_error_event(1, "x", code=code)
    assert set(event.keys()) == BASE_FIELDS
    dropped = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR and "dropped=code" in record.getMessage()
    ]
    assert len(dropped) == 1


@pytest.mark.parametrize(
    "code",
    [
        "connector_not_found",
        "invalid_runtime_context",
        "missing_runtime_context",
        "runtime_context_immutable",
        "runtime_secret_not_allowed",
        "runtime_secret_unavailable",
        "scheduled_secret_unavailable",
        "connector_runtime_unavailable",
    ],
)
def test_every_connector_runtime_code_survives_the_closed_set(code: str) -> None:
    """All eight connector-runtime codes are members, so none is dropped."""

    event = create_terminal_task_error_event(1, "x", code=code)

    assert event["code"] == code


def test_the_closed_set_is_the_connector_runtime_subset_of_v1() -> None:
    """The closed set is a curated subset of V1ErrorCode, not the whole enum.

    Membership excludes the two authorization-outcome codes: nothing raises
    them as a ``ConnectorRuntimeError`` today, and each one is the outcome of
    an authorization check -- the kind of fact this frame must not carry.
    """

    assert CONNECTOR_RUNTIME_CLIENT_ERROR_CODES == {
        "connector_not_found",
        "invalid_runtime_context",
        "missing_runtime_context",
        "runtime_context_immutable",
        "runtime_secret_not_allowed",
        "runtime_secret_unavailable",
        "scheduled_secret_unavailable",
        "connector_runtime_unavailable",
    }
    assert CONNECTOR_RUNTIME_CLIENT_ERROR_CODES <= {
        member.value for member in V1ErrorCode
    }


def test_a_non_connector_v1_code_is_dropped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A V1ErrorCode member outside the connector-runtime family is dropped.

    ``invalid_api_key`` is a real member of ``V1ErrorCode`` -- the /v1 error
    surface -- but it is not a connector-runtime code, so it must not reach
    this frame.
    """

    with caplog.at_level(logging.ERROR):
        event = create_terminal_task_error_event(1, "x", code="invalid_api_key")

    assert "code" not in event
    dropped = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR and "dropped=code" in record.getMessage()
    ]
    assert len(dropped) == 1


@pytest.mark.parametrize(
    "code",
    ["mcp_oauth_authorization_failed", "delegated_authorization_failed"],
)
def test_authorization_outcome_codes_are_dropped(code: str) -> None:
    """These two codes are the outcome of an authorization check.

    Neither is raised as a ``ConnectorRuntimeError`` in this repository
    today, and both are deliberately absent from the closed set: this frame
    reaches anonymous widget and share-link visitors, and the outcome of an
    authorization check is exactly the kind of fact it must not carry.
    """

    event = create_terminal_task_error_event(1, "x", code=code)

    assert "code" not in event


def _raise_site_error_names() -> set[str]:
    """The ``ERROR_*`` names this source tree actually produces as a code.

    Walks every ``.py`` file for two shapes: a call to
    ``_raise_runtime_error(<Name>, ...)`` or ``ConnectorRuntimeError(<Name>,
    ...)`` whose first positional argument is a bare name, and a bare
    ``return <Name>`` -- some raise sites choose the code dynamically
    through a small dispatch function (for example, one runtime-secret code
    or its scheduled-trigger variant, picked by the failing task's source)
    and pass the result through a local variable rather than the constant
    itself, so the constant only appears literally at the ``return``. Either
    shape is read as "this name is a code value somewhere in the raise
    path," not as a full trace of which raise call it eventually reaches.
    """

    names: set[str] = set()
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                if isinstance(node.value, ast.Name) and node.value.id.startswith(
                    "ERROR_"
                ):
                    names.add(node.value.id)
                continue
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            func_name = (
                func.id
                if isinstance(func, ast.Name)
                else (func.attr if isinstance(func, ast.Attribute) else None)
            )
            if func_name not in RAISE_CALL_NAMES or not node.args:
                continue
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Name) and first_arg.id.startswith("ERROR_"):
                names.add(first_arg.id)
    return names


def test_every_client_code_is_named_at_a_producer_site() -> None:
    """Every code in the closed set is named where connector-runtime errors are made.

    A code that reaches the closed set ahead of its producer is an allowance
    with no expiry date, so this test derives the producer names fresh from
    the AST instead of trusting a hand-maintained list. What it pins is
    narrower than "raised somewhere": a code counts as produced when its
    constant is the first argument of a ``ConnectorRuntimeError`` or
    ``_raise_runtime_error`` call, or is returned by one of the helpers that
    pick the code before such a call. A helper that returns a code and is
    never called would satisfy this test; reachability of the helper is not
    checked here.
    """

    error_names = _raise_site_error_names()
    assert error_names, "the AST scan found no ERROR_* raise sites at all"
    resolved_codes = {
        getattr(connector_runtime_module, name)
        for name in error_names
        if hasattr(connector_runtime_module, name)
    }
    assert CONNECTOR_RUNTIME_CLIENT_ERROR_CODES <= resolved_codes
