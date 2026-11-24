"""Integration tests for ``PATCH /v1/agents/{agent_id}`` (name, description, instructions).

The route changes the three text fields through the write the web app uses.
Every refusal leaves the agent row, the agent cache and the log untouched. A
failure inside the write is reported as an unknown outcome, a failure reading
the response after the write is not, and in both the stored change stays.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from xagent.web.models.agent import Agent
from xagent.web.schemas.v1 import V1AgentUpdateRequest
from xagent.web.services import agent_management
from xagent.web.services.agent_host_settings import set_agent_host_settings_writer
from xagent.web.services.agent_management import AgentManagementRuntime
from xagent.web.services.agent_store import AgentStore
from xagent.web.services.agent_team_scope import (
    AgentTeamScope,
    set_agent_team_scope_hook,
)
from xagent.web.services.workforce_snapshot import _fingerprint_agent_payload

from ..conftest import _admin_headers, _direct_db_session, client
from .test_agent_detail import (
    _CELLS,
    NOT_FOUND,
    _add_user,
    _install_role,
    _seed_column,
    _user_id,
)
from .test_management import _bearer
from .test_skills import _KEY_KINDS, _error, _headers_for, _key_for

pytestmark = pytest.mark.usefixtures("_test_db")

TEXT_FIELDS = ("name", "description", "instructions")
UPDATE_LOG = "personal key agent update:"
UNKNOWN_LOG = "personal key agent update outcome unknown:"
CACHE_CLEAR = "xagent.web.services.agent_store.invalidate_agent_cache"
# Request fields that are not among the three the route accepts.
UNSUPPORTED_FIELDS = (
    "execution_mode",
    "knowledge_bases",
    "skills",
    "tool_categories",
    "suggested_prompts",
    "prompt_context_fields",
    "logo_url",
    "status",
    "widget_enabled",
    "allowed_domains",
    "widget_key",
    "share_enabled",
    "share_token",
    "share_updated_at",
    "visibility",
    "logo_base64",
    "host_settings",
)


@pytest.fixture(autouse=True)
def _reset_team_hook():
    yield
    set_agent_team_scope_hook(None)


# ===== helpers =====


class Holder:
    """The key holder: a personal key, its prefix and the user id."""

    def __init__(self) -> None:
        issued = _key_for(_admin_headers())
        self.key = issued["full_key"]
        self.prefix = issued["key_prefix"]
        self.user_id = _user_id("admin")


def _seed(owner_id: int, **columns: Any) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(user_id=owner_id, **{"name": "original", **columns})
        db.add(agent)
        db.commit()
        return int(agent.id)
    finally:
        db.close()


def _row(agent_id: int) -> dict[str, Any]:
    """Every column of the agent row, as a fresh session reads it."""
    db = _direct_db_session()
    try:
        agent = db.get(Agent, agent_id)
        return {
            column.key: getattr(agent, column.key)
            for column in sa_inspect(Agent).column_attrs
        }
    finally:
        db.close()


def _patch(holder: Holder, agent_id: int, body: Any):
    return client.patch(
        f"/v1/agents/{agent_id}", headers=_bearer(holder.key), json=body
    )


def _update_records(caplog: pytest.LogCaptureFixture, prefix: str = UPDATE_LOG):
    return [
        record
        for record in caplog.records
        if record.getMessage().startswith(prefix)
        and record.name == agent_management.logger.name
    ]


def _logged_text(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(
        record.getMessage() + repr(record.args) for record in caplog.records
    )


# ===== only a live personal key opens the route =====


@pytest.mark.parametrize("kind", _KEY_KINDS)
def test_new_agent_routes_accept_only_a_live_personal_key(kind):
    assert len(_KEY_KINDS) == 4
    resp = client.patch(
        "/v1/agents/1", headers=_headers_for(kind), json={"description": "x"}
    )

    _error(resp, 401, "invalid_api_key")


# ===== who can change =====


@pytest.mark.parametrize(
    ("role", "column", "status"), _CELLS, ids=[f"{r}-{c}" for r, c, _ in _CELLS]
)
def test_update_visibility_matrix(role, column, status):
    holder = Holder()
    bob = _add_user("bob")
    _install_role(role, holder.user_id)
    agent_id = _seed_column(column, holder.user_id, bob)
    before = _row(agent_id) if column != "nonexistent" else None

    resp = _patch(holder, agent_id, {"description": "changed"})

    assert resp.status_code == status, resp.text
    if status == 404:
        assert resp.json() == NOT_FOUND
        # An agent that is hidden looks exactly like one that does not exist.
        assert resp.content == _patch(holder, 999_998, {"description": "x"}).content
        if before is not None:
            assert _row(agent_id) == before
    else:
        assert resp.json()["description"] == "changed"
        assert _row(agent_id)["description"] == "changed"


# ===== a refusal changes nothing =====

_TOO_LONG = "can be at most 10000 characters unless it is no longer than the stored"
_TAKEN = "Agent with this name already exists."
_ABSOLUTE = "at most 1000000 characters"
# (id, request body, status, text the message contains)
_REFUSALS = [
    ("name-too-long", {"name": "n" * 201}, 422, "at most 200 characters"),
    ("nul", {"description": "a\x00b"}, 422, "NUL"),
    ("lone-surrogate", '{"instructions": "\\ud800"}', 422, "valid string"),
    ("over-absolute", {"instructions": "x" * 1_000_001}, 422, _ABSOLUTE),
    # The length is refused first, so the scan for NUL never runs.
    (
        "over-absolute-with-nul",
        {"description": "\x00" + "x" * 1_000_000},
        422,
        _ABSOLUTE,
    ),
    ("null", {"name": None}, 400, "name cannot be null; omit it to keep"),
    ("instructions-too-long", {"instructions": "x" * 10_001}, 400, _TOO_LONG),
    ("description-too-long", {"description": "x" * 10_001}, 400, _TOO_LONG),
    ("name-taken", {"name": "taken"}, 400, _TAKEN),
    ("name-taken-at-the-index", {"name": "taken"}, 400, _TAKEN),
    (
        "too-long-and-taken",
        {"name": "taken", "instructions": "x" * 10_001},
        400,
        _TOO_LONG,
    ),
    ("no-key", {"description": "x"}, 401, "Invalid"),
    ("not-found", {"description": "x"}, 404, ""),
]


@pytest.mark.parametrize(
    ("body", "status", "text"),
    [row[1:] for row in _REFUSALS],
    ids=[row[0] for row in _REFUSALS],
)
def test_rejected_update_changes_nothing(request, caplog, body, status, text):
    case = request.node.callspec.id
    holder = Holder()
    agent_id = _seed(holder.user_id, name="mine", description="d", instructions="i")
    _seed(holder.user_id, name="taken")
    before = _row(agent_id)
    headers = {} if case == "no-key" else _bearer(holder.key)
    target = 999_999 if case == "not-found" else agent_id

    # The pre-check says the name is free, so the unique index has to refuse it.
    precheck = (
        patch.object(AgentStore, "agent_name_exists", return_value=False)
        if case == "name-taken-at-the-index"
        else contextlib.nullcontext()
    )
    with patch(CACHE_CLEAR) as cleared, precheck, caplog.at_level(logging.INFO):
        if isinstance(body, str):
            resp = client.patch(
                f"/v1/agents/{target}",
                headers={**headers, "Content-Type": "application/json"},
                content=body,
            )
        else:
            resp = client.patch(f"/v1/agents/{target}", headers=headers, json=body)

    assert resp.status_code == status, resp.text
    assert text in resp.json()["error"]["message"] or status == 404
    assert _row(agent_id) == before
    cleared.assert_not_called()
    assert _update_records(caplog) == []
    assert _update_records(caplog, UNKNOWN_LOG) == []


# ===== the five states of each field =====

_STORED = {"name": "stored name", "description": "stored d", "instructions": "stored i"}
# (state, value sent, status, stored value afterwards or None for "unchanged")
_STATES = [
    ("same", "SAME", 200, None),
    ("different", "new value", 200, "new value"),
    ("null", None, 400, None),
    ("empty", "", None, ""),
    ("number", 123, 422, None),
    ("nul", "a\x00b", 422, None),
    ("lone-surrogate", "SURROGATE", 422, None),
]


def _state_cases():
    for field in TEXT_FIELDS:
        for state, value, status, after in _STATES:
            if state == "empty" and field == "name":
                status = 422
            elif state == "empty":
                status = 200
            yield pytest.param(field, value, status, after, id=f"{field}-{state}")
    yield pytest.param("name", "n" * 201, 422, None, id="name-over-200")


@pytest.mark.parametrize(("field", "value", "status", "after"), _state_cases())
def test_update_field_five_states(field, value, status, after):
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)
    before = _row(agent_id)
    if value == "SURROGATE":
        # A lone surrogate cannot be encoded by the client, so the JSON text
        # carries it as an escape.
        resp = client.patch(
            f"/v1/agents/{agent_id}",
            headers={**_bearer(holder.key), "Content-Type": "application/json"},
            content='{"%s": "\\ud800"}' % field,
        )
    else:
        sent = _STORED[field] if value == "SAME" else value
        resp = _patch(holder, agent_id, {field: sent})

    assert resp.status_code == status, resp.text
    if value is None:
        assert resp.json()["error"] == {
            "code": "invalid_input",
            "message": f"{field} cannot be null; omit it to keep the stored value.",
        }
    row = _row(agent_id)
    if status != 200:
        assert row == before
        return
    assert resp.json()[field] == row[field]
    if after is None:
        # The same value: nothing is written, not even the update time.
        assert row == before
    else:
        assert row[field] == after
        assert {k: v for k, v in row.items() if k not in (field, "updated_at")} == {
            k: v for k, v in before.items() if k not in (field, "updated_at")
        }


def test_an_empty_body_takes_no_lock_and_reads_as_get(monkeypatch, caplog):
    """``{}`` has nothing to decide, so it takes no lock: while another
    session holds SQLite's write lock it returns what GET returns."""
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)
    before = _row(agent_id)
    reads, _writes = _spy_store(monkeypatch)
    other = _direct_db_session()
    other.connection().exec_driver_sql("BEGIN IMMEDIATE")
    try:
        with patch(CACHE_CLEAR) as cleared, caplog.at_level(logging.INFO):
            resp = _patch(holder, agent_id, {})
        read = client.get(f"/v1/agents/{agent_id}", headers=_bearer(holder.key))
    finally:
        other.rollback()
        other.close()

    assert resp.status_code == 200, resp.text
    assert resp.content == read.content
    assert resp.json()["instructions"] == "stored i"
    # One read for the PATCH and one for the GET, neither under the row lock.
    assert [kw.get("for_update") for kw, _agent in reads] == [None, None]
    assert _row(agent_id) == before
    cleared.assert_not_called()
    assert _update_records(caplog) == []
    missing = _patch(holder, 999_999, {})
    assert missing.status_code == 404 and missing.json() == NOT_FOUND


_EXTRA_BODIES = [
    {"foo": 1},
    {"foo": 1, "description": "x"},
    {"models": {"general": 1}},
    {"models": None},
    {"name": "x", "models": {}},
]
_REFUSED_FIELD_BODIES = [{field: 1} for field in UNSUPPORTED_FIELDS]


@pytest.mark.parametrize(
    "body",
    _EXTRA_BODIES + _REFUSED_FIELD_BODIES,
    ids=["+".join(body) for body in _EXTRA_BODIES + _REFUSED_FIELD_BODIES],
)
def test_only_three_fields_are_accepted(caplog, body):
    assert len(UNSUPPORTED_FIELDS) == 17
    # The request model declares exactly the three fields.
    assert set(V1AgentUpdateRequest.model_fields) == set(TEXT_FIELDS)
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)
    before = _row(agent_id)
    extra = next(key for key in body if key not in TEXT_FIELDS)

    with patch(CACHE_CLEAR) as cleared, caplog.at_level(logging.INFO):
        resp = _patch(holder, agent_id, body)

    error = _error(resp, 422, "invalid_input")
    assert f"Extra inputs are not permitted ({extra})" in error["message"]
    assert _row(agent_id) == before
    cleared.assert_not_called()
    assert _update_records(caplog) == []


def test_every_update_field_is_described():
    assert all(
        field.description for field in V1AgentUpdateRequest.model_fields.values()
    )


# ===== how the write is made =====


def _spy_store(monkeypatch: pytest.MonkeyPatch):
    """Record each locked read and each write the store is asked for."""
    reads: list[tuple[dict[str, Any], Agent | None]] = []
    writes: list[tuple[dict[str, Any], dict[str, Any]]] = []
    real_get = AgentStore.get_owned_agent
    real_write = AgentStore.update_agent_fields

    def get_owned_agent(self, *args, **kwargs):
        agent = real_get(self, *args, **kwargs)
        reads.append((kwargs, agent))
        return agent

    def update_agent_fields(self, user_id, agent_id, updates, **kwargs):
        writes.append((dict(updates), kwargs))
        return real_write(self, user_id, agent_id, updates, **kwargs)

    monkeypatch.setattr(AgentStore, "get_owned_agent", get_owned_agent)
    monkeypatch.setattr(AgentStore, "update_agent_fields", update_agent_fields)
    return reads, writes


@pytest.mark.parametrize(
    ("body", "written"),
    [
        ({"name": "renamed"}, {"name"}),
        ({"description": "d2"}, {"description"}),
        (
            {"name": "renamed", "description": "d2", "instructions": "i2"},
            set(TEXT_FIELDS),
        ),
        # A field that already has the sent value is not written.
        ({"name": "stored name", "instructions": "i2"}, {"instructions"}),
        ({"name": "stored name"}, set()),
    ],
    ids=["name", "description", "all", "partly-unchanged", "unchanged"],
)
def test_update_writes_once_through_update_agent_fields(monkeypatch, body, written):
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)
    reads, writes = _spy_store(monkeypatch)

    assert _patch(holder, agent_id, body).status_code == 200

    # The agent is read under the row lock and, after a write, once more
    # without it for the response.
    expected_reads = [True, None] if written else [True]
    assert [kw.get("for_update") for kw, _agent in reads] == expected_reads
    locked = reads[0][1]
    assert locked is not None
    if not written:
        assert writes == []
        return
    assert len(writes) == 1
    updates, kwargs = writes[0]
    assert set(updates) == written
    assert kwargs["agent"] is locked


def _before_the_lock(monkeypatch: pytest.MonkeyPatch, act) -> list[int]:
    """Run ``act`` once right before the update takes its first lock.

    On SQLite that is the database write lock, taken before anything is read,
    so what ``act`` does is something that happens between the request being
    accepted and the update reading the agent.
    """
    ran: list[int] = []
    real = agent_management.begin_sqlite_write_transaction

    def begin_sqlite_write_transaction(db):
        if not ran:
            act()
            ran.append(1)
        real(db)

    monkeypatch.setattr(
        agent_management,
        "begin_sqlite_write_transaction",
        begin_sqlite_write_transaction,
    )
    return ran


def _set_columns(agent_id: int, **columns: Any) -> None:
    db = _direct_db_session()
    try:
        agent = db.get(Agent, agent_id)
        for key, value in columns.items():
            setattr(agent, key, value)
        db.commit()
    finally:
        db.close()


def test_access_lost_before_the_lock_refuses_the_update(monkeypatch, caplog):
    holder = Holder()
    bob = _add_user("bob")
    scope = AgentTeamScope(team_id=1, is_team_admin=False)
    set_agent_team_scope_hook(
        lambda _db, user_id: scope if int(user_id) == holder.user_id else None
    )
    agent_id = _seed(bob, team_id=1, description="d")
    ran = _before_the_lock(
        monkeypatch, lambda: _set_columns(agent_id, visibility="admins")
    )
    with patch(CACHE_CLEAR) as cleared, caplog.at_level(logging.INFO):
        resp = _patch(holder, agent_id, {"description": "changed"})

    assert ran == [1], "the change before the lock did not happen"
    assert resp.json() == NOT_FOUND and resp.status_code == 404
    row = _row(agent_id)
    assert row["visibility"] == "admins" and row["description"] == "d"
    cleared.assert_not_called()
    assert _update_records(caplog) == []


# ===== the log line =====

_LOG_REQUESTS = [
    ("name", {"name": "renamed"}, "name"),
    ("instructions", {"instructions": "SECRET-BODY"}, "instructions"),
    (
        "all",
        {"name": "n2", "description": "SECRET-D", "instructions": "SECRET-BODY"},
        "description,instructions,name",
    ),
]


@pytest.mark.parametrize(
    ("body", "fields"),
    [row[1:] for row in _LOG_REQUESTS],
    ids=[row[0] for row in _LOG_REQUESTS],
)
def test_update_log_line(caplog, body, fields):
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)

    with caplog.at_level(logging.INFO):
        resp = _patch(holder, agent_id, body)

    assert resp.status_code == 200, resp.text
    records = _update_records(caplog)
    assert len(records) == 1
    assert records[0].args == (holder.prefix, holder.user_id, agent_id, fields)
    assert records[0].levelno == logging.INFO
    assert "SECRET" not in _logged_text(caplog)


def test_cancelled_update_still_logs(caplog):
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)
    entered, release = threading.Event(), threading.Event()
    real_write = AgentStore.update_agent_fields

    def blocked_write(self, *args, **kwargs):
        entered.set()
        assert release.wait(10)
        return real_write(self, *args, **kwargs)

    async def cancel_while_writing() -> None:
        task = asyncio.create_task(
            AgentManagementRuntime().update_agent(
                user_id=holder.user_id,
                key_prefix=holder.prefix,
                agent_id=agent_id,
                changes={"instructions": "after cancel"},
            )
        )
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    with (
        patch.object(AgentStore, "update_agent_fields", blocked_write),
        caplog.at_level(logging.INFO),
    ):
        asyncio.run(cancel_while_writing())

    assert _row(agent_id)["instructions"] == "after cancel"
    records = _update_records(caplog)
    assert len(records) == 1
    assert records[0].args == (holder.prefix, holder.user_id, agent_id, "instructions")


# ===== side effects, by field =====


@pytest.mark.parametrize(
    ("field", "fingerprint_changes"),
    [("name", False), ("description", False), ("instructions", True)],
)
def test_update_workforce_fingerprint_by_field(field, fingerprint_changes):
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)

    def fingerprint() -> dict[str, Any]:
        db = _direct_db_session()
        try:
            return _fingerprint_agent_payload(db.get(Agent, agent_id))
        finally:
            db.close()

    before = fingerprint()
    assert _patch(holder, agent_id, {field: "changed"}).status_code == 200

    assert (fingerprint() != before) is fingerprint_changes


@pytest.mark.parametrize("team_id", [None, 1], ids=["personal", "team"])
def test_update_invalidates_agent_cache(team_id):
    holder = Holder()
    if team_id is not None:
        scope = AgentTeamScope(team_id=team_id, is_team_admin=False)
        set_agent_team_scope_hook(lambda _db, _user_id: scope)
    agent_id = _seed(holder.user_id, team_id=team_id, **_STORED)

    with patch(CACHE_CLEAR) as cleared:
        assert _patch(holder, agent_id, {"name": "renamed"}).status_code == 200

    cleared.assert_called_once_with(holder.user_id, agent_id, team_id)


@pytest.mark.parametrize(
    "body",
    [
        {"name": "renamed", "description": "d2", "instructions": "i2"},
        {"instructions": ""},
    ],
    ids=["all", "empty-instructions"],
)
def test_update_response_equals_next_get(body):
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)

    resp = _patch(holder, agent_id, body)
    read = client.get(f"/v1/agents/{agent_id}", headers=_bearer(holder.key))

    assert resp.status_code == 200 and read.status_code == 200
    assert resp.json() == read.json()


# ===== the response after the write =====


def test_agent_hidden_after_the_commit_returns_404(caplog):
    """A team admin makes the agent admins-only between the update's commit
    and its response: the change stays stored and the member gets 404."""
    holder = Holder()
    bob = _add_user("bob")
    scope = AgentTeamScope(team_id=1, is_team_admin=False)
    set_agent_team_scope_hook(
        lambda _db, user_id: scope if int(user_id) == holder.user_id else None
    )
    agent_id = _seed(bob, team_id=1, instructions="team text")
    missing = _patch(holder, 999_999, {"description": "changed"})
    real_refresh = Session.refresh
    ran: list[int] = []

    def refresh(self, instance, *args, **kwargs):
        if not ran and isinstance(instance, Agent):
            ran.append(1)
            _set_columns(agent_id, visibility="admins", instructions="ADMIN-ONLY")
        return real_refresh(self, instance, *args, **kwargs)

    with patch.object(Session, "refresh", refresh), caplog.at_level(logging.INFO):
        resp = _patch(holder, agent_id, {"description": "changed"})

    assert ran == [1], "the admin's change did not happen"
    assert missing.status_code == 404
    assert resp.status_code == 404 and resp.content == missing.content
    row = _row(agent_id)
    assert row["description"] == "changed" and row["instructions"] == "ADMIN-ONLY"
    assert len(_update_records(caplog)) == 1
    assert _update_records(caplog, UNKNOWN_LOG) == []


def test_response_comes_from_one_read_after_the_commit():
    """Another writer changes the agent after the update's commit: every
    response field, ``unresolved_models`` included, shows that writer's row."""
    holder = Holder()
    agent_id = _seed(holder.user_id, instructions="before")
    real_write = AgentStore.update_agent_fields

    def write_then_another_writer(self, *args, **kwargs):
        written = real_write(self, *args, **kwargs)
        _set_columns(
            agent_id, instructions="another writer", models={"general": 999_999}
        )
        return written

    with patch.object(AgentStore, "update_agent_fields", write_then_another_writer):
        resp = _patch(holder, agent_id, {"description": "changed"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["description"] == "changed"
    assert body["instructions"] == "another writer"
    assert body["models"] == {"general": 999_999}
    assert body["unresolved_models"] == ["general"]


def test_failure_reading_the_response_after_the_commit(caplog):
    """The change is stored and logged at INFO before the response is read,
    so a failure there is not an unknown outcome."""
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)
    boom = OperationalError("SELECT secret_table", {}, Exception("secret detail"))
    real_get = AgentStore.get_owned_agent

    def get_owned_agent(self, *args, **kwargs):
        if not kwargs.get("for_update"):
            raise boom
        return real_get(self, *args, **kwargs)

    with (
        patch.object(AgentStore, "get_owned_agent", get_owned_agent),
        caplog.at_level(logging.INFO),
    ):
        resp = _patch(holder, agent_id, {"instructions": "changed"})

    _error(resp, 500, "internal_error")
    assert _row(agent_id)["instructions"] == "changed"
    assert [record.args for record in _update_records(caplog)] == [
        (holder.prefix, holder.user_id, agent_id, "instructions")
    ]
    assert _update_records(caplog, UNKNOWN_LOG) == []


def test_sqlite_write_lock_serializes_length_decisions():
    """Two updates of a 15000-character text, to 10000 and to 14000: the
    second waits for the first's write lock, then reads 10000 and is refused.

    SQLite ignores the row lock, so this holds only because the update takes
    SQLite's write lock before it reads anything.
    """
    holder = Holder()
    agent_id = _seed(holder.user_id, instructions="x" * 15_000)
    first_read, release_first = threading.Event(), threading.Event()
    second_started, second_read = threading.Event(), threading.Event()
    real_get = AgentStore.get_owned_agent
    outcomes: dict[str, Any] = {}

    def get_owned_agent(self, *args, **kwargs):
        agent = real_get(self, *args, **kwargs)
        if kwargs.get("for_update"):
            if threading.current_thread().name == "first":
                first_read.set()
                assert release_first.wait(10)
            else:
                second_read.set()
        return agent

    def run(length: int) -> None:
        name = threading.current_thread().name
        if name == "second":
            second_started.set()
        try:
            AgentManagementRuntime._update_agent_sync(
                user_id=holder.user_id,
                key_prefix=holder.prefix,
                agent_id=agent_id,
                changes={"instructions": name[0] * length},
            )
            outcomes[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            outcomes[name] = exc

    first = threading.Thread(target=run, args=(10_000,), name="first")
    second = threading.Thread(target=run, args=(14_000,), name="second")
    with patch.object(AgentStore, "get_owned_agent", get_owned_agent):
        first.start()
        try:
            assert first_read.wait(10)
            second.start()
            assert second_started.wait(10)
            # While the first holds the write lock the second may not read.
            read_too_early = second_read.wait(1.0)
        finally:
            release_first.set()
            first.join(15)
            second.join(15)

    assert not read_too_early
    assert outcomes["first"] == "ok"
    assert isinstance(outcomes["second"], agent_management.AgentTextTooLongError)
    assert _row(agent_id)["instructions"] == "f" * 10_000


def test_locked_database_returns_500_and_writes_nothing(monkeypatch, caplog):
    """The SQLite write lock cannot be taken: 500 before the agent is read,
    nothing written or cleared from the cache, and neither update log line."""
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)
    before = _row(agent_id)
    reads, writes = _spy_store(monkeypatch)

    def locked(_db):
        raise OperationalError("BEGIN IMMEDIATE", {}, Exception("database is locked"))

    monkeypatch.setattr(agent_management, "begin_sqlite_write_transaction", locked)
    with patch(CACHE_CLEAR) as cleared, caplog.at_level(logging.INFO):
        resp = _patch(holder, agent_id, {"instructions": "changed"})

    _error(resp, 500, "internal_error")
    assert reads == [] and writes == []
    assert _row(agent_id) == before
    cleared.assert_not_called()
    assert _update_records(caplog) == []
    assert _update_records(caplog, UNKNOWN_LOG) == []


# ===== nothing else is touched =====


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"name": "renamed", "instructions": "i2"}, 200),
        ({"name": "taken"}, 400),
        ({"description": "x" * 10_001}, 400),
        ({"visibility": "admins"}, 422),
    ],
    ids=["success", "refused-name", "refused-length", "refused-field"],
)
def test_update_never_calls_the_host_settings_writer(body, status):
    holder = Holder()
    agent_id = _seed(
        holder.user_id,
        tool_categories=["a"],
        knowledge_bases=["kb"],
        skills=["s"],
        description="d",
    )
    _seed(holder.user_id, name="taken")
    before = _row(agent_id)

    calls: list[tuple[Any, ...]] = []
    set_agent_host_settings_writer(lambda *args: calls.append(args))
    try:
        resp = _patch(holder, agent_id, body)
    finally:
        set_agent_host_settings_writer(None)

    assert resp.status_code == status, resp.text
    assert calls == []
    changed = set(body) | {"updated_at"} if status == 200 else set()
    row = _row(agent_id)
    assert {k: v for k, v in row.items() if k not in changed} == {
        k: v for k, v in before.items() if k not in changed
    }


# ===== failures before and after the commit =====


def test_failure_after_commit_keeps_the_change(caplog):
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)
    boom = OperationalError("SELECT secret_table", {}, Exception("secret detail"))
    real_commit = Session.commit

    with (
        patch.object(Session, "refresh", side_effect=boom),
        caplog.at_level(logging.INFO),
    ):
        resp = _patch(holder, agent_id, {"instructions": "SECRET-BODY"})

    _error(resp, 500, "internal_error")
    assert "secret" not in resp.text.lower()
    assert _row(agent_id)["instructions"] == "SECRET-BODY"
    unknown = _update_records(caplog, UNKNOWN_LOG)
    assert len(unknown) == 1
    assert unknown[0].levelno == logging.ERROR
    assert unknown[0].args == (holder.prefix, holder.user_id, agent_id, "instructions")
    assert _update_records(caplog) == []
    assert "SECRET-BODY" not in _logged_text(caplog)

    # Sending the request again changes nothing and logs nothing.
    caplog.clear()
    commits: list[int] = []

    def counting_commit(self):
        commits.append(1)
        return real_commit(self)

    with (
        patch.object(Session, "commit", counting_commit),
        caplog.at_level(logging.INFO),
    ):
        retry = _patch(holder, agent_id, {"instructions": "SECRET-BODY"})

    assert retry.status_code == 200, retry.text
    assert retry.json()["instructions"] == "SECRET-BODY"
    assert commits == []
    assert _update_records(caplog) == [] and _update_records(caplog, UNKNOWN_LOG) == []


def test_failure_before_commit_logs_outcome_unknown(caplog):
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)
    before = _row(agent_id)
    boom = OperationalError("UPDATE secret_table", {}, Exception("secret detail"))

    with (
        patch.object(AgentStore, "update_agent_fields", side_effect=boom),
        patch(CACHE_CLEAR) as cleared,
        caplog.at_level(logging.INFO),
    ):
        resp = _patch(holder, agent_id, {"instructions": "SECRET-BODY"})

    _error(resp, 500, "internal_error")
    assert "secret" not in resp.text.lower()
    assert _row(agent_id) == before
    cleared.assert_not_called()
    unknown = _update_records(caplog, UNKNOWN_LOG)
    assert len(unknown) == 1
    assert unknown[0].args == (holder.prefix, holder.user_id, agent_id, "instructions")
    assert _update_records(caplog) == []
    assert "SECRET-BODY" not in _logged_text(caplog)


# ===== text may not grow past its limit =====

_UNCHANGED, _GROWN, _SHORTER = "unchanged", "changed-same-length", "shorter"
# (stored length or None, sent length, kind of request, status)
_LENGTHS = [
    (0, 10_000, _GROWN, 200),
    (0, 10_001, _GROWN, 400),
    (15_000, 15_000, _UNCHANGED, 200),
    (15_000, 15_000, _GROWN, 200),
    (15_000, 14_000, _SHORTER, 200),
    (15_000, 15_001, _GROWN, 400),
    (None, 10_001, _GROWN, 400),
    (1_000_001, 1_000_001, _UNCHANGED, 422),
]


@pytest.mark.parametrize("field", ["instructions", "description"])
@pytest.mark.parametrize(
    ("stored", "sent", "kind", "status"),
    _LENGTHS,
    ids=[f"{s}-{n}-{k}" for s, n, k, _ in _LENGTHS],
)
def test_long_text_can_be_sent_back_but_not_grown(
    monkeypatch, field, stored, sent, kind, status
):
    holder = Holder()
    stored_text = None if stored is None else "a" * stored
    agent_id = _seed(holder.user_id, **{field: stored_text})
    before = _row(agent_id)
    _reads, writes = _spy_store(monkeypatch)
    # A changed value of the same length differs in its last character only.
    value = "a" * sent if kind == _UNCHANGED else "a" * (sent - 1) + "b"

    resp = _patch(holder, agent_id, {field: value})

    assert resp.status_code == status, resp.text
    row = _row(agent_id)
    if status != 200:
        assert row == before and writes == []
    elif kind == _UNCHANGED:
        assert row == before and writes == []
        assert resp.json()[field] == value
    else:
        assert row[field] == value
        assert len(writes) == 1 and set(writes[0][0]) == {field}
