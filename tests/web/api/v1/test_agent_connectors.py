"""Integration tests for reading and replacing an agent's connectors over /v1."""

import logging

import pytest

from xagent.web.models.agent import Agent, AgentOrigin
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.services import connector_team_scope, mcp_server_management
from xagent.web.services.agent_store import AgentStore
from xagent.web.services.agent_team_scope import (
    AgentTeamScope,
    set_agent_team_hooks,
)

from ..conftest import _direct_db_session, _register_second_user, client
from .test_agent_skills import _user_id, team_scopes  # noqa: F401,F811
from .test_mcp_servers import (  # noqa: F401  (_no_team_hooks is an autouse fixture)
    Holder,
    _error,
    _holder,
    _no_team_hooks,
    _seed,
)

pytestmark = pytest.mark.usefixtures("_test_db")

_DEFAULT_TOOLS_MESSAGE = (
    "This agent uses the default tool set; choose its tools in the web app first."
)


# ===== helpers =====


@pytest.fixture
def alice() -> Holder:
    holder = _holder("alice")
    _seed("crm", owner=holder)
    _seed("My-Server", owner=holder)
    _seed("ops", owner=holder)
    db = _direct_db_session()
    try:
        api = CustomApi(name="Billing API")
        db.add(api)
        db.flush()
        db.add(
            UserCustomApi(user_id=holder.user_id, custom_api_id=api.id, is_owner=True)
        )
        db.commit()
    finally:
        db.close()
    return holder


def _seed_agent(
    owner: str,
    tool_categories,
    *,
    team_id: int | None = None,
    visibility: str = "team",
    origin: str = AgentOrigin.USER.value,
) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=_user_id(owner),
            name=f"agent {db.query(Agent).count()}",
            tool_categories=tool_categories,
            team_id=team_id,
            visibility=visibility,
            origin=origin,
        )
        db.add(agent)
        db.commit()
        return int(agent.id)
    finally:
        db.close()


def _stored(agent_id: int):
    db = _direct_db_session()
    try:
        return db.get(Agent, agent_id).tool_categories
    finally:
        db.close()


def _get(holder: Holder, agent_id: int):
    return client.get(f"/v1/agents/{agent_id}/connectors", headers=holder.headers)


def _put(holder: Holder, agent_id: int, connectors, all_connectors=False):
    body = {"connectors": connectors}
    if all_connectors is not None:
        body["all_connectors"] = all_connectors
    return client.put(
        f"/v1/agents/{agent_id}/connectors", headers=holder.headers, json=body
    )


def _count_writes(monkeypatch) -> list[int]:
    writes: list[int] = []
    real = AgentStore.update_agent_fields

    def counted(self, user_id, agent_id, *args, **kwargs):
        writes.append(agent_id)
        return real(self, user_id, agent_id, *args, **kwargs)

    monkeypatch.setattr(AgentStore, "update_agent_fields", counted)
    return writes


def _count_list_hook(monkeypatch, before=None) -> list[str]:
    """Record each call of the team list hook; ``before`` runs ahead of it."""
    events: list[str] = []
    real = mcp_server_management.visible_team_connector_ids

    def counted(db, user_id):
        events.append("hook")
        if before is not None:
            before()
        return real(db, user_id)

    monkeypatch.setattr(mcp_server_management, "visible_team_connector_ids", counted)
    return events


# ===== the connectors are reachable exactly where the agent is =====

# (cell, the caller's role in team 1, how the agent is stored, reachable)
_ACCESS = [
    ("own", "member", {"owner": "alice"}, True),
    ("team-visible", "member", {"owner": "bob", "team_id": 1}, True),
    (
        "admins-only-to-a-member",
        "member",
        {"owner": "bob", "team_id": 1, "visibility": "admins"},
        False,
    ),
    (
        "admins-only-to-an-admin",
        "admin",
        {"owner": "bob", "team_id": 1, "visibility": "admins"},
        True,
    ),
    ("other-team", "admin", {"owner": "bob", "team_id": 2}, False),
    (
        "workforce-manager",
        "member",
        {"owner": "alice", "origin": AgentOrigin.WORKFORCE_GENERATED_MANAGER.value},
        False,
    ),
    ("nonexistent", "member", None, False),
]


@pytest.mark.parametrize("method", ["GET", "PUT"])
@pytest.mark.parametrize(
    ("cell", "role", "stored_as", "reachable"), _ACCESS, ids=[c[0] for c in _ACCESS]
)
def test_connector_list_is_reachable_exactly_where_the_agent_is(
    cell,
    role,
    stored_as,
    reachable,
    method,
    alice,
    team_scopes,  # noqa: F811
    monkeypatch,
):
    assert len(_ACCESS) == 7
    _register_second_user("bob")
    team_scopes[alice.user_id] = AgentTeamScope(
        team_id=1, is_team_admin=role == "admin"
    )
    seeded = ["basic_tools", "mcp:seeded"]
    agent_id = (
        999_999
        if stored_as is None
        else _seed_agent(tool_categories=seeded, **stored_as)
    )
    hook_calls = _count_list_hook(monkeypatch)

    resp = _get(alice, agent_id) if method == "GET" else _put(alice, agent_id, [])

    if not reachable:
        _error(resp, 404, "agent_not_found")
        assert hook_calls == [], "access is refused before any connector is read"
        if stored_as is not None:
            assert _stored(agent_id) == seeded
        return
    assert resp.status_code == 200, resp.text
    assert hook_calls == ["hook"]
    if method == "GET":
        assert resp.json() == {
            "agent_id": agent_id,
            "connectors": ["seeded"],
            "all_connectors": False,
            "unresolved": ["seeded"],
        }
    else:
        assert _stored(agent_id) == ["basic_tools"]


def test_agent_id_beyond_the_id_column_is_refused(alice):
    _error(_get(alice, 2**31), 422, "invalid_input")


def test_default_tool_set_reads_as_every_connector(alice, monkeypatch):
    agent_id = _seed_agent("alice", None)
    hook_calls = _count_list_hook(monkeypatch)

    resp = _get(alice, agent_id)

    assert resp.json() == {
        "agent_id": agent_id,
        "connectors": [],
        "all_connectors": True,
        "unresolved": [],
    }
    assert hook_calls == []


# ===== a replacement keeps order and checks only the names it adds =====

# (cell, stored before, connectors, all_connectors, status, stored after, writes,
#  expected response fields or expected error)
_REPLACE = [
    (
        "default_tool_set",
        None,
        ["crm"],
        False,
        409,
        None,
        0,
        {"code": "agent_tools_unconfigured", "message": _DEFAULT_TOOLS_MESSAGE},
    ),
    (
        "unchanged_list_is_not_written",
        ["basic_tools", "mcp:crm", "web_tools"],
        ["crm"],
        False,
        200,
        ["basic_tools", "mcp:crm", "web_tools"],
        0,
        {"connectors": ["crm"]},
    ),
    (
        "invisible_added_name",
        ["basic_tools"],
        ["ghost"],
        False,
        400,
        ["basic_tools"],
        0,
        {
            "code": "invalid_input",
            "message": "These connectors are not visible to the key holder: 'ghost'",
        },
    ),
    (
        "custom_api_name",
        ["basic_tools"],
        ["Billing API"],
        False,
        200,
        ["basic_tools", "mcp:Billing API"],
        1,
        {"unresolved": []},
    ),
    (
        "folded_match",
        ["basic_tools"],
        ["my_server"],
        False,
        200,
        ["basic_tools", "mcp:my_server"],
        1,
        {"unresolved": []},
    ),
    (
        "invisible_old_name_is_kept",
        ["basic_tools", "mcp:gone"],
        ["gone", "crm"],
        False,
        200,
        ["basic_tools", "mcp:gone", "mcp:crm"],
        1,
        {"unresolved": ["gone"]},
    ),
    (
        "connector_entries_move_to_the_end",
        ["mcp:ops", "basic_tools", "web_tools"],
        ["My-Server", "ops"],
        False,
        200,
        ["basic_tools", "web_tools", "mcp:My-Server", "mcp:ops"],
        1,
        {},
    ),
    (
        "exact_duplicates_collapse",
        ["basic_tools"],
        ["crm", "crm"],
        False,
        200,
        ["basic_tools", "mcp:crm"],
        1,
        {},
    ),
    (
        "folded_duplicates_stay",
        ["basic_tools"],
        ["crm", "CRM"],
        False,
        200,
        ["basic_tools", "mcp:crm", "mcp:CRM"],
        1,
        {},
    ),
    (
        "blank_name",
        ["basic_tools", "mcp:crm"],
        ["crm", "   "],
        False,
        400,
        ["basic_tools", "mcp:crm"],
        0,
        {"code": "invalid_input", "message": "Connector names cannot be blank."},
    ),
    (
        "unassignable_category_is_dropped",
        ["basic_tools", "other"],
        ["crm"],
        False,
        200,
        ["basic_tools", "mcp:crm"],
        1,
        {"connectors": ["crm"]},
    ),
    (
        "stored_duplicates_are_not_rewritten",
        ["basic_tools", "mcp:crm", "mcp:crm"],
        ["crm"],
        False,
        200,
        ["basic_tools", "mcp:crm", "mcp:crm"],
        0,
        {"connectors": ["crm", "crm"]},
    ),
    (
        "all_false_drops_the_category",
        ["basic_tools", "mcp"],
        ["crm"],
        False,
        200,
        ["basic_tools", "mcp:crm"],
        1,
        {"all_connectors": False},
    ),
    (
        "all_true_keeps_the_category",
        ["basic_tools", "mcp"],
        ["crm"],
        True,
        200,
        ["basic_tools", "mcp", "mcp:crm"],
        1,
        {"all_connectors": True},
    ),
    (
        "all_true_alone_is_a_change",
        ["basic_tools", "mcp:crm"],
        ["crm"],
        True,
        200,
        ["basic_tools", "mcp", "mcp:crm"],
        1,
        {"all_connectors": True},
    ),
    (
        "missing_all_connectors",
        ["basic_tools"],
        ["crm"],
        None,
        422,
        ["basic_tools"],
        0,
        {"code": "invalid_input"},
    ),
]


@pytest.mark.parametrize(
    (
        "cell",
        "before",
        "names",
        "all_connectors",
        "status",
        "after",
        "writes",
        "expect",
    ),
    _REPLACE,
    ids=[r[0] for r in _REPLACE],
)
def test_replacement_keeps_order_and_checks_only_added_names(
    cell,
    before,
    names,
    all_connectors,
    status,
    after,
    writes,
    expect,
    alice,
    monkeypatch,
):
    assert len(_REPLACE) == 16
    agent_id = _seed_agent("alice", before)
    write_calls = _count_writes(monkeypatch)

    resp = _put(alice, agent_id, names, all_connectors)

    assert resp.status_code == status, resp.text
    assert _stored(agent_id) == after
    assert len(write_calls) == writes
    if status == 200:
        body = resp.json()
        assert body["agent_id"] == agent_id
        assert {k: body[k] for k in expect} == expect
    else:
        error = _error(resp, status, expect["code"])
        if "message" in expect:
            assert error["message"] == expect["message"]


@pytest.mark.parametrize(
    "names",
    [[f"c{i}" for i in range(101)], ["x" * 101], [""]],
    ids=["over_100_items", "name_over_100_chars", "empty_name"],
)
def test_oversized_or_empty_names_are_refused(names, alice):
    agent_id = _seed_agent("alice", ["basic_tools"])
    _error(_put(alice, agent_id, names), 422, "invalid_input")
    assert _stored(agent_id) == ["basic_tools"]


def test_refusal_lists_five_names_and_cuts_each_to_100_characters(alice):
    agent_id = _seed_agent("alice", ["basic_tools"])
    names = ["y" * 100, "a1", "a2", "a3", "a4", "a5"]

    error = _error(_put(alice, agent_id, names), 400, "invalid_input")

    assert error["message"] == (
        "These connectors are not visible to the key holder: "
        + ", ".join(repr(name)[:100] for name in names[:5])
    )
    assert _stored(agent_id) == ["basic_tools"]


def test_connectors_and_custom_apis_shared_by_the_team_count_as_visible(alice):
    other = _holder("bob")
    server_id = _seed("team-crm", owner=other)
    db = _direct_db_session()
    try:
        api = CustomApi(name="Team API")
        db.add(api)
        db.flush()
        db.add(
            UserCustomApi(user_id=other.user_id, custom_api_id=api.id, is_owner=True)
        )
        db.commit()
        api_id = int(api.id)
    finally:
        db.close()
    connector_team_scope.set_connector_team_hooks(
        visibility=lambda _db, _user_id: {
            "mcp": {server_id},
            "custom_api": {api_id},
        }
    )
    agent_id = _seed_agent("alice", ["basic_tools"])

    resp = _put(alice, agent_id, ["team-crm", "Team API"])

    assert resp.status_code == 200, resp.text
    assert resp.json()["unresolved"] == []
    assert _stored(agent_id) == ["basic_tools", "mcp:team-crm", "mcp:Team API"]


def test_a_tool_list_stored_as_text_is_read_and_written_as_a_list(alice):
    agent_id = _seed_agent("alice", '["basic_tools", "mcp:crm"]')

    got = _get(alice, agent_id)
    assert got.status_code == 200, got.text
    assert got.json()["connectors"] == ["crm"]

    resp = _put(alice, agent_id, ["crm", "ops"])

    assert resp.status_code == 200, resp.text
    assert _stored(agent_id) == ["basic_tools", "mcp:crm", "mcp:ops"]


# ===== a team agent's selection is validated by the team =====


def _team_agent(alice, validator, before=("basic_tools", "mcp:crm")) -> int:
    _register_second_user("bob")
    set_agent_team_hooks(
        scope=lambda _db, user_id: (
            AgentTeamScope(team_id=1, is_team_admin=False)
            if user_id == alice.user_id
            else None
        ),
        connector_validator=validator,
    )
    return _seed_agent("bob", list(before), team_id=1)


@pytest.fixture
def _clear_agent_team_hooks():
    yield
    set_agent_team_hooks()


def _refusal(details_connectors, message_names):
    return {
        "connectors": details_connectors,
        "message": "The agent's team cannot use these connectors: "
        + message_names
        + ". Share them with the team in the web app; a connector you disabled "
        "must be enabled first.",
    }


def _unshared_unless_all(_db, _user_id, _team_id, categories):
    return [{"type": "mcp", "id": 2, "name": "other"}] if "mcp" in categories else []


_REFUSALS = [
    (
        "each_connector_gets_a_reason",
        lambda *_: [
            {"type": "mcp", "id": 1, "name": "x"},
            {"type": "mcp", "name": "y", "reason": "unresolved"},
        ],
        [],
        False,
        _refusal(
            [
                {"name": "x", "reason": "not_shared"},
                {"name": "y", "reason": "unresolved"},
            ],
            "'x', 'y'",
        ),
    ),
    (
        "own_disabled_connector",
        lambda *_: [{"type": "mcp", "name": "crm-off", "reason": "unresolved"}],
        ["crm-off"],
        False,
        _refusal([{"name": "crm-off", "reason": "disabled"}], "'crm-off'"),
    ),
    (
        "disabled_but_enabled_under_a_folded_twin",
        lambda *_: [{"type": "mcp", "name": "crm-off", "reason": "unresolved"}],
        ["crm-off"],
        False,
        _refusal([{"name": "crm-off", "reason": "unresolved"}], "'crm-off'"),
    ),
    (
        "disabled_but_shared_by_the_team",
        lambda *_: [{"type": "mcp", "name": "crm-off", "reason": "unresolved"}],
        ["crm-off"],
        False,
        _refusal([{"name": "crm-off", "reason": "unresolved"}], "'crm-off'"),
    ),
    (
        "unknown_reason_reads_as_not_shared",
        lambda *_: [{"type": "mcp", "name": "x", "reason": "something_new"}],
        [],
        False,
        _refusal([{"name": "x", "reason": "not_shared"}], "'x'"),
    ),
    (
        "all_connectors_adds_the_category",
        _unshared_unless_all,
        ["crm"],
        True,
        _refusal([{"name": "other", "reason": "not_shared"}], "'other'"),
    ),
]


@pytest.mark.parametrize(
    ("cell", "validator", "names", "all_connectors", "expect"),
    _REFUSALS,
    ids=[r[0] for r in _REFUSALS],
)
@pytest.mark.usefixtures("_clear_agent_team_hooks")
def test_team_agent_refusal_lists_each_connector_with_a_reason(
    cell, validator, names, all_connectors, expect, alice
):
    assert len(_REFUSALS) == 6
    _seed("crm-off", owner=alice, is_active=False)
    if cell == "disabled_but_enabled_under_a_folded_twin":
        _seed("CRM_off", owner=alice)
    if cell == "disabled_but_shared_by_the_team":
        shared_id = _seed("CRM_off")
        connector_team_scope.set_connector_team_hooks(
            visibility=lambda _db, _user_id: {
                "mcp": {shared_id},
                "custom_api": set(),
            }
        )
    seen: list[list[str]] = []

    def recording(db, user_id, team_id, categories):
        seen.append(list(categories))
        return validator(db, user_id, team_id, categories)

    agent_id = _team_agent(alice, recording)

    error = _error(
        _put(alice, agent_id, names, all_connectors), 400, "connector_not_shared"
    )

    assert error["details"] == {"connectors": expect["connectors"]}
    assert error["message"] == expect["message"]
    assert ("mcp" in seen[-1]) == all_connectors
    assert _stored(agent_id) == ["basic_tools", "mcp:crm"]


# ===== the visible names are read before the agent row is locked =====


def test_visible_names_are_read_before_the_agent_row_is_locked(alice, monkeypatch):
    agent_id = _seed_agent("alice", ["basic_tools"])
    events: list[tuple] = []
    real = AgentStore.get_owned_agent

    def spy(self, *args, **kwargs):
        events.append(("agent", kwargs.get("for_update", False)))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(AgentStore, "get_owned_agent", spy)
    _count_list_hook(monkeypatch, before=lambda: events.append(("hook",)))

    assert _put(alice, agent_id, ["crm"]).status_code == 200
    assert events == [("agent", False), ("hook",), ("agent", True)]


def test_access_is_checked_before_the_visible_names_are_read(alice, monkeypatch):
    hook_calls = _count_list_hook(monkeypatch)

    _error(_put(alice, 999_999, ["crm"]), 404, "agent_not_found")
    _error(_get(alice, 999_999), 404, "agent_not_found")
    assert hook_calls == []


def test_an_agent_deleted_while_names_are_read_is_not_found(alice, monkeypatch):
    agent_id = _seed_agent("alice", ["basic_tools"])

    def delete_agent():
        db = _direct_db_session()
        try:
            db.query(Agent).filter(Agent.id == agent_id).delete()
            db.commit()
        finally:
            db.close()

    writes = _count_writes(monkeypatch)
    _count_list_hook(monkeypatch, before=delete_agent)

    _error(_put(alice, agent_id, ["crm"]), 404, "agent_not_found")
    assert writes == []


def test_the_write_decides_on_the_agent_as_it_is_under_the_lock(alice, monkeypatch):
    agent_id = _seed_agent("alice", ["basic_tools"])

    def another_edit():
        db = _direct_db_session()
        try:
            db.get(Agent, agent_id).tool_categories = ["basic_tools", "web_tools"]
            db.commit()
        finally:
            db.close()

    _count_list_hook(monkeypatch, before=another_edit)

    assert _put(alice, agent_id, ["crm"]).status_code == 200
    assert _stored(agent_id) == ["basic_tools", "web_tools", "mcp:crm"]


# ===== a failing visibility read is not an empty set =====


@pytest.mark.parametrize("method", ["GET", "PUT"])
def test_a_failing_visibility_read_is_not_an_empty_set(method, alice):
    agent_id = _seed_agent("alice", ["basic_tools"])

    def failing(_db, _user_id):
        raise RuntimeError("team backend exploded")

    connector_team_scope.set_connector_team_hooks(visibility=failing)

    resp = _get(alice, agent_id) if method == "GET" else _put(alice, agent_id, ["crm"])

    _error(resp, 500, "internal_error")
    assert _stored(agent_id) == ["basic_tools"]


# ===== a write leaves one log line =====


@pytest.mark.usefixtures("_clear_agent_team_hooks")
def test_a_failing_team_validator_writes_nothing(alice, monkeypatch):
    def failing(*_args):
        raise RuntimeError("team backend exploded")

    agent_id = _team_agent(alice, failing)
    writes = _count_writes(monkeypatch)

    _error(_put(alice, agent_id, ["crm", "ops"]), 500, "internal_error")
    assert writes == [agent_id]
    assert _stored(agent_id) == ["basic_tools", "mcp:crm"]


@pytest.mark.parametrize("outcome", ["written", "unchanged", "refused"])
def test_a_write_logs_one_line_without_names(outcome, alice, caplog):
    _seed("zz-secret-conn", owner=alice)
    _seed("zz-other-conn", owner=alice)
    agent_id = _seed_agent("alice", ["basic_tools", "mcp:crm", "mcp:ops"])
    names = {
        "written": ["ops", "zz-secret-conn", "zz-other-conn"],
        "unchanged": ["crm", "ops"],
        "refused": ["zz-unseen-conn"],
    }[outcome]
    caplog.clear()

    with caplog.at_level(logging.INFO):
        resp = _put(alice, agent_id, names)

    assert resp.status_code == (400 if outcome == "refused" else 200), resp.text
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("personal key agent connectors write:")
    ]
    if outcome != "written":
        assert lines == []
        return
    prefix = alice.key.split("_")[2]
    assert lines == [
        f"personal key agent connectors write: key_prefix={prefix} "
        f"user_id={alice.user_id} agent_id={agent_id} added=2 removed=1 "
        "all_connectors=False"
    ]
    assert "secret" not in "\n".join(r.getMessage() for r in caplog.records)
