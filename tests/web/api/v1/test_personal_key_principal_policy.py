"""The application principal policy runs on every personal-key route."""

import logging
import re

import pytest
from fastapi.routing import APIRoute, iter_route_contexts

from xagent.templates.manager import TemplateManager
from xagent.web.api.v1 import deps
from xagent.web.api.v1.deps import (
    UserPrincipalSnapshot,
    get_principal_from_api_key,
    get_user_from_personal_key,
)
from xagent.web.services.personal_key_scope import set_personal_key_principal_policy

from ..conftest import app_for_tests, client
from .test_auth import _assert_invalid_api_key, _create_personal_key

pytestmark = pytest.mark.usefixtures("_test_db")

POLICY_LOGGER = "xagent.web.services.personal_key_scope"
POLICY_FAILURE_TEXT = "policy backend exploded"
#: Every dependency that admits a personal key.
PERSONAL_KEY_DEPENDENCIES = {get_user_from_personal_key, get_principal_from_api_key}


@pytest.fixture(autouse=True)
def _clear_principal_policy():
    set_personal_key_principal_policy(None)
    yield
    set_personal_key_principal_policy(None)


@pytest.fixture
def _template_manager(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = TemplateManager(templates_root=tmp_path)
    monkeypatch.setattr(app_for_tests.state, "template_manager", manager, raising=False)


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


def _dependency_calls(node):
    yield node.call
    for child in node.dependencies:
        yield from _dependency_calls(child)


def _personal_key_routes() -> list[tuple[str, str]]:
    """Every (method, path) whose dependency tree resolves a personal key."""
    routes = []
    for route in iter_route_contexts(app_for_tests.routes):
        if not isinstance(route.original_route, APIRoute):
            continue
        tree = route.dependant  # codespell:ignore dependant
        if PERSONAL_KEY_DEPENDENCIES.intersection(_dependency_calls(tree)):
            routes.extend((method, route.path) for method in route.methods)
    return sorted(routes)


@pytest.mark.parametrize(
    ("outcome", "log_level"),
    [
        (True, None),
        (False, logging.WARNING),
        (None, logging.WARNING),
        ("yes", logging.WARNING),
        (1, logging.WARNING),
        (RuntimeError(POLICY_FAILURE_TEXT), logging.ERROR),
    ],
    ids=["true", "false", "none", "string", "integer", "raises"],
)
def test_personal_key_policy_requires_literal_true(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    outcome: object,
    log_level: int | None,
) -> None:
    full_key, prefix = _create_personal_key()
    user_id = client.get("/v1/me", headers=_bearer(full_key)).json()["user_id"]
    seen: list[UserPrincipalSnapshot] = []

    def policy(user: UserPrincipalSnapshot) -> object:
        seen.append(user)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    pads: list[None] = []
    monkeypatch.setattr(deps, "verify_dummy", lambda: pads.append(None))
    set_personal_key_principal_policy(policy)
    with caplog.at_level(logging.DEBUG):
        resp = client.get("/v1/me", headers=_bearer(full_key))

    assert len(seen) == 1
    assert isinstance(seen[0], UserPrincipalSnapshot) and seen[0].id == user_id
    records = [r for r in caplog.records if r.name == POLICY_LOGGER]
    if log_level is None:
        assert resp.status_code == 200, resp.text
        assert records == [] and pads == []
        return
    _assert_invalid_api_key(resp)
    assert len(pads) == 1
    assert [r.levelno for r in records] == [log_level]
    assert records[0].args == (prefix, user_id)
    assert POLICY_FAILURE_TEXT not in records[0].getMessage()
    assert records[0].exc_info is None


def test_personal_key_policy_runs_only_after_the_key_verifies() -> None:
    full_key, _prefix = _create_personal_key()
    wrong_secret = full_key[:-1] + ("A" if full_key[-1] != "A" else "B")
    seen: list[object] = []

    def policy(user: object) -> bool:
        seen.append(user)
        return True

    set_personal_key_principal_policy(policy)
    resp = client.get("/v1/me", headers=_bearer(wrong_secret))

    _assert_invalid_api_key(resp)
    assert seen == []


@pytest.mark.usefixtures("_template_manager")
def test_personal_key_policy_applies_to_every_personal_key_route() -> None:
    routes = _personal_key_routes()
    # One long-standing route of each router. The walk finds every other route
    # by itself, so this only guards against it coming back empty or partial
    # and the loop below passing without checking anything.
    assert {
        ("GET", "/v1/me"),
        ("POST", "/v1/agents"),
        ("POST", "/v1/chat/tasks"),
        ("GET", "/v1/features"),
    } <= set(routes), routes
    headers = _bearer(_create_personal_key()[0])

    for method, template in routes:
        path = re.sub(r"\{[^}]+\}", "1", template)

        set_personal_key_principal_policy(None)
        unset = client.request(method, path, headers=headers)
        assert unset.status_code != 401 and unset.status_code < 500, (
            method,
            template,
            unset.text,
        )

        set_personal_key_principal_policy(lambda _user: False)
        denied = client.request(method, path, headers=headers)
        assert denied.status_code == 401, (method, template, denied.text)
        _assert_invalid_api_key(denied)
