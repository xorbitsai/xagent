"""Integration tests for ``PATCH /v1/agents/{agent_id}``.

The route changes the name, description, instructions and model slots through
the write the web app uses. Every refusal leaves the agent row, the agent
cache and the log untouched. A failure inside the write is reported as an
unknown outcome, a failure reading the response after the write is not, and
in both the stored change stays.
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
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.core.tools.adapters.vibe.agent_model_resolution import AGENT_MODEL_KEYS
from xagent.web import models as _all_models  # noqa: F401  (registers every table)
from xagent.web.models.agent import Agent
from xagent.web.models.agent_version import AgentVersion
from xagent.web.models.database import Base
from xagent.web.models.user import User, UserModel
from xagent.web.schemas.v1 import V1AgentModelSlots, V1AgentUpdateRequest
from xagent.web.services import agent_management
from xagent.web.services.agent_host_settings import set_agent_host_settings_writer
from xagent.web.services.agent_management import AgentManagementRuntime
from xagent.web.services.agent_store import AgentStore
from xagent.web.services.agent_team_scope import (
    AgentTeamScope,
    set_agent_team_scope_hook,
)
from xagent.web.services.agent_versions import model_slot_is_unset
from xagent.web.services.model_service import (
    set_model_visibility_hook,
    set_visible_user_ids_hook,
)
from xagent.web.services.workforce_snapshot import _fingerprint_agent_payload

from ..conftest import _admin_headers, _direct_db_session, _register_second_user, client
from .test_agent_detail import (
    _CELL_NAMES,
    _CELLS,
    NOT_FOUND,
    World,
    _add_model,
    _add_user,
    _build_world,
    _install_role,
    _seed_agent,
    _seed_column,
    _user_id,
)
from .test_management import _bearer
from .test_skills import _KEY_KINDS, _error, _headers_for, _key_for

pytestmark = pytest.mark.usefixtures("_test_db")

TEXT_FIELDS = ("name", "description", "instructions")
UPDATE_FIELDS = (*TEXT_FIELDS, "models")
UPDATE_LOG = "personal key agent update:"
UNKNOWN_LOG = "personal key agent update outcome unknown:"
CACHE_CLEAR = "xagent.web.services.agent_store.invalidate_agent_cache"
# Request fields that are not among the four the route accepts.
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
    set_visible_user_ids_hook(None)
    set_model_visibility_hook(None)


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
_NULL_MODELS = "models cannot be null; omit it to keep the stored value."
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
    ("models-null", {"models": None}, 400, _NULL_MODELS),
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


@pytest.mark.parametrize("body", [{}, {"models": {}}], ids=["empty", "no-slot"])
def test_an_empty_body_takes_no_lock_and_reads_as_get(monkeypatch, caplog, body):
    """A body that names nothing to change has nothing to decide, so it takes
    no lock: while another session holds SQLite's write lock it returns what
    GET returns."""
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)
    before = _row(agent_id)
    reads, _writes = _spy_store(monkeypatch)
    other = _direct_db_session()
    other.connection().exec_driver_sql("BEGIN IMMEDIATE")
    try:
        with patch(CACHE_CLEAR) as cleared, caplog.at_level(logging.INFO):
            resp = _patch(holder, agent_id, body)
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
    missing = _patch(holder, 999_999, body)
    assert missing.status_code == 404 and missing.json() == NOT_FOUND


_EXTRA_BODIES = [
    {"foo": 1},
    {"foo": 1, "description": "x"},
    {"name": "x", "foo": 1, "models": {}},
]
_REFUSED_FIELD_BODIES = [{field: 1} for field in UNSUPPORTED_FIELDS]


@pytest.mark.parametrize(
    "body",
    _EXTRA_BODIES + _REFUSED_FIELD_BODIES,
    ids=["+".join(body) for body in _EXTRA_BODIES + _REFUSED_FIELD_BODIES],
)
def test_only_four_fields_are_accepted(caplog, body):
    assert len(UNSUPPORTED_FIELDS) == 17
    # The request model declares exactly the four fields.
    assert set(V1AgentUpdateRequest.model_fields) == set(UPDATE_FIELDS)
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)
    before = _row(agent_id)
    extra = next(key for key in body if key not in UPDATE_FIELDS)

    with patch(CACHE_CLEAR) as cleared, caplog.at_level(logging.INFO):
        resp = _patch(holder, agent_id, body)

    error = _error(resp, 422, "invalid_input")
    assert f"Extra inputs are not permitted ({extra})" in error["message"]
    assert _row(agent_id) == before
    cleared.assert_not_called()
    assert _update_records(caplog) == []


@pytest.mark.parametrize("schema", [V1AgentUpdateRequest, V1AgentModelSlots])
def test_every_update_field_is_described(schema):
    assert all(field.description for field in schema.model_fields.values())


def test_request_slots_are_the_slots_merged():
    assert tuple(V1AgentModelSlots.model_fields) == AGENT_MODEL_KEYS


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
        (lambda ok: {"models": {"visual": ok}}, {"models"}),
        (lambda ok: {"name": "renamed", "models": {"visual": ok}}, {"name", "models"}),
        (lambda ok: {"models": {"general": ok}}, set()),
    ],
    ids=[
        "name",
        "description",
        "all",
        "partly-unchanged",
        "unchanged",
        "models",
        "name-and-models",
        "same-slot",
    ],
)
def test_update_writes_once_through_update_agent_fields(monkeypatch, body, written):
    holder = Holder()
    ok = _add_model("ok", links=((holder.user_id, True, False),))
    agent_id = _seed(holder.user_id, models={"general": ok}, **_STORED)
    reads, writes = _spy_store(monkeypatch)
    body = body(ok) if callable(body) else body

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
    (
        "two-slots",
        lambda ok: {"models": {"general": ok, "visual": ok}},
        "models[general,visual]",
    ),
    (
        "every-field",
        lambda ok: {
            "name": "n2",
            "description": "SECRET-D",
            "instructions": "SECRET-BODY",
            "models": {"compact": ok, "general": ok, "small_fast": ok, "visual": ok},
        },
        "description,instructions,models[general,small_fast,visual,compact],name",
    ),
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
    ok = _add_model("ok", links=((holder.user_id, True, False),))
    body = body(ok) if callable(body) else body

    with caplog.at_level(logging.INFO):
        resp = _patch(holder, agent_id, body)

    assert resp.status_code == 200, resp.text
    records = _update_records(caplog)
    assert len(records) == 1
    assert records[0].args == (holder.prefix, holder.user_id, agent_id, fields)
    assert records[0].levelno == logging.INFO
    assert "SECRET" not in _logged_text(caplog)


# Stored slot values cleared with null: (id, value, written?). A value that
# is unset (model_slot_is_unset) is kept as stored; any other is written.
_SLOT_CLEARS = [
    ("int-zero", 0, False),
    ("false", False, False),
    ("blank", "", False),
    ("float-zero", 0.0, True),
    ("empty-list", [], True),
    ("empty-dict", {}, True),
]

# (id, stored columns, request body) that update_agent_fields keeps as stored.
_KEPT_AS_STORED = [
    (
        "instructions-whitespace",
        {"instructions": "stored i"},
        {"instructions": "  stored i\n"},
    ),
    *(
        (
            f"cleared-{name}-slot",
            {"models": {"general": value}},
            {"models": {"general": None}},
        )
        for name, value, written in _SLOT_CLEARS
        if not written
    ),
]


@pytest.mark.parametrize(
    ("stored", "body"),
    [row[1:] for row in _KEPT_AS_STORED],
    ids=[row[0] for row in _KEPT_AS_STORED],
)
def test_a_value_kept_as_stored_is_not_written_or_logged(caplog, stored, body):
    holder = Holder()
    agent_id = _seed(holder.user_id, **stored)
    before = _row(agent_id)

    with patch(CACHE_CLEAR) as cleared, caplog.at_level(logging.INFO):
        resp = _patch(holder, agent_id, body)

    assert resp.status_code == 200, resp.text
    assert resp.json()["unresolved_models"] == []
    assert _row(agent_id) == before
    cleared.assert_not_called()
    assert _update_records(caplog) == []
    db = _direct_db_session()
    try:
        assert (
            db.query(AgentVersion).filter(AgentVersion.agent_id == agent_id).count()
            == 0
        )
    finally:
        db.close()


@pytest.mark.parametrize(
    "value",
    [value for _, value, written in _SLOT_CLEARS if written],
    ids=[name for name, _, written in _SLOT_CLEARS if written],
)
def test_clearing_a_listed_slot_writes_null(caplog, value):
    holder = Holder()
    agent_id = _seed(holder.user_id, models={"general": value})
    before = client.get(f"/v1/agents/{agent_id}", headers=_bearer(holder.key))
    assert before.json()["unresolved_models"] == ["general"]

    with patch(CACHE_CLEAR) as cleared, caplog.at_level(logging.INFO):
        resp = _patch(holder, agent_id, {"models": {"general": None}})

    assert resp.status_code == 200, resp.text
    assert resp.json()["models"] == {"general": None}
    assert resp.json()["unresolved_models"] == []
    assert _row(agent_id)["models"] == {"general": None}
    cleared.assert_called_once_with(holder.user_id, agent_id, None)
    records = _update_records(caplog)
    assert len(records) == 1
    assert records[0].args == (
        holder.prefix,
        holder.user_id,
        agent_id,
        "models[general]",
    )
    db = _direct_db_session()
    try:
        versions = (
            db.query(AgentVersion)
            .filter(AgentVersion.agent_id == agent_id)
            .order_by(AgentVersion.version_no)
            .all()
        )
        assert [v.kind for v in versions] == ["initial", "save"]
        assert versions[0].content["models"] == {"general": value}
        assert versions[1].content["models"] == {"general": None}
        assert versions[1].actor_kind == "api_key"
    finally:
        db.close()


def test_slot_clear_table_follows_the_unset_rule():
    for _, value, written in _SLOT_CLEARS:
        assert model_slot_is_unset(value) is (not written), value


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
    [
        ("name", False),
        ("description", False),
        ("instructions", True),
        ("models", True),
    ],
)
def test_update_workforce_fingerprint_by_field(field, fingerprint_changes):
    holder = Holder()
    agent_id = _seed(holder.user_id, **_STORED)
    ok = _add_model("ok", links=((holder.user_id, True, False),))

    def fingerprint() -> dict[str, Any]:
        db = _direct_db_session()
        try:
            return _fingerprint_agent_payload(db.get(Agent, agent_id))
        finally:
            db.close()

    before = fingerprint()
    body = {"models": {"general": ok}} if field == "models" else {field: "changed"}
    assert _patch(holder, agent_id, body).status_code == 200

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


@pytest.mark.parametrize("field", ["description", "models"])
def test_response_comes_from_one_read_after_the_commit(field):
    """Another writer changes the agent after the update's commit: every
    response field, ``unresolved_models`` included, shows that writer's row."""
    holder = Holder()
    agent_id = _seed(holder.user_id, instructions="before")
    ok = _add_model("ok", links=((holder.user_id, True, False),))
    sent = {"description": "changed", "models": {"visual": ok}}[field]
    real_write = AgentStore.update_agent_fields

    def write_then_another_writer(self, *args, **kwargs):
        written = real_write(self, *args, **kwargs)
        _set_columns(
            agent_id, instructions="another writer", models={"general": 999_999}
        )
        return written

    with patch.object(AgentStore, "update_agent_fields", write_then_another_writer):
        resp = _patch(holder, agent_id, {field: sent})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    if field == "description":
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
        (lambda ok: {"models": {"visual": ok}}, 200),
        (lambda ok: {"models": {"visual": ok + 100}}, 400),
    ],
    ids=[
        "success",
        "refused-name",
        "refused-length",
        "refused-field",
        "models",
        "refused-models",
    ],
)
def test_update_never_calls_the_host_settings_writer(body, status):
    holder = Holder()
    ok = _add_model("ok", links=((holder.user_id, True, False),))
    body = body(ok) if callable(body) else body
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


# ===== model slots: what the body may say =====

_BAD_SLOT_BODIES = [
    {"foo": 1},
    {"general": True},
    {"general": False},
    {"general": "5"},
    {"general": 5.0},
    {"general": 0},
    {"general": -1},
    {"general": 2**31},
    {"general": 1, "foo": 1},
]


@pytest.mark.parametrize(
    "slots", _BAD_SLOT_BODIES, ids=[repr(slots) for slots in _BAD_SLOT_BODIES]
)
def test_invalid_slot_values_are_refused_without_a_write(caplog, slots):
    holder = Holder()
    agent_id = _seed(holder.user_id, models={"general": 999_999}, **_STORED)
    before = _row(agent_id)

    with patch(CACHE_CLEAR) as cleared, caplog.at_level(logging.INFO):
        resp = _patch(holder, agent_id, {"models": slots})

    _error(resp, 422, "invalid_input")
    assert _row(agent_id) == before
    cleared.assert_not_called()
    assert _update_records(caplog) == []


# ===== model slots: merge and comparison =====


def _stored_variants(old: int, stale: int) -> dict[str, Any]:
    return {
        "dict": {"general": old, "compact": stale},
        "none": None,
        "corrupt-list": [1, 2],
        "unknown-key": {"foo": 3, "general": old, "compact": stale},
    }


# Requests that are applied, as functions of (usable model, stale model).
_SLOT_REQUESTS = {
    "empty": lambda ok, stale: {},
    "clear-general": lambda ok, stale: {"general": None},
    "set-visual": lambda ok, stale: {"visual": ok},
    "unchanged-stale": lambda ok, stale: {"general": ok, "compact": stale},
}


@pytest.mark.parametrize("stored_kind", list(_stored_variants(1, 2)))
@pytest.mark.parametrize("request_kind", list(_SLOT_REQUESTS))
def test_update_models_merges_by_slot(stored_kind, request_kind):
    holder = Holder()
    me = holder.user_id
    old = _add_model("old", links=((me, True, False),))
    ok = _add_model("ok", links=((me, True, False),))
    stale = _add_model("stale", active=False, links=((me, True, False),))
    stored = _stored_variants(old, stale)[stored_kind]
    agent_id = _seed(me, models=stored)
    before = _row(agent_id)
    sent = _SLOT_REQUESTS[request_kind](ok, stale)
    base = dict(stored) if isinstance(stored, dict) else {}
    differs = {slot: pk for slot, pk in sent.items() if base.get(slot) != pk}

    resp = _patch(holder, agent_id, {"models": sent})

    # The stale model is refused as a new value and kept when it is stored.
    if request_kind == "unchanged-stale" and base.get("compact") != stale:
        _error(resp, 400, "model_not_assignable")
        assert resp.json()["error"]["details"]["models"] == [
            {"slot": "compact", "id": stale, "reason": "unavailable"}
        ]
        assert _row(agent_id) == before
        return
    assert resp.status_code == 200, resp.text
    if not differs:
        assert _row(agent_id) == before
        assert resp.json()["models"] == (stored if isinstance(stored, dict) else None)
        return
    expected = {**base, **differs}
    assert resp.json()["models"] == expected
    assert _row(agent_id)["models"] == expected
    assert resp.json()["unresolved_models"] == (
        ["compact"] if expected.get("compact") == stale else []
    )


@pytest.mark.parametrize("stored", ["true", "float", "string"])
def test_noncanonical_stored_slot_can_be_fixed(stored):
    holder = Holder()
    ok = _add_model("ok", links=((holder.user_id, True, False),))
    # ``True`` is the integer 1, so it stands for this model only when it is 1.
    assert ok == 1
    value = {"true": True, "float": 1.0, "string": "1"}[stored]
    agent_id = _seed(holder.user_id, models={"general": value})
    before = client.get(f"/v1/agents/{agent_id}", headers=_bearer(holder.key))
    assert before.json()["unresolved_models"] == ["general"]

    resp = _patch(holder, agent_id, {"models": {"general": ok}})

    assert resp.status_code == 200, resp.text
    assert resp.json()["unresolved_models"] == []
    stored_value = _row(agent_id)["models"]["general"]
    assert stored_value == 1 and type(stored_value) is int


def test_two_refused_slots_are_reported_in_slot_order():
    holder = Holder()
    me = holder.user_id
    image = _add_model("image", category="image", links=((me, True, False),))
    agent_id = _seed(me)

    resp = _patch(holder, agent_id, {"models": {"compact": 999_999, "general": image}})

    error = _error(resp, 400, "model_not_assignable")
    assert error["details"]["models"] == [
        {"slot": "general", "id": image, "reason": "not_llm"},
        {"slot": "compact", "id": 999_999, "reason": "unavailable"},
    ]
    assert error["message"] == (
        "These model slots cannot use the requested model: "
        "general (not_llm), compact (unavailable)."
    )


# ===== model slots: who may use the model =====

_COLUMN_NAMES = ["H=O", "H!=O", "O-left"]
# Result of changing a slot to each kind of model, per column of the table of
# who the agent's owner is. Anything not listed is accepted.
_ALWAYS_UNAVAILABLE = {"m5", "m6", "m7", "m8", "m9"}
_OUTCOMES = {
    "H=O": {**dict.fromkeys(_ALWAYS_UNAVAILABLE, "unavailable")}
    | {"m10": "not_llm", "m11": "not_llm"},
    "H!=O": {
        **dict.fromkeys(_ALWAYS_UNAVAILABLE, "unavailable"),
        "m1": "owner_cannot_use",
        "m2": "unavailable",
        "m10": "not_llm",
        "m11": "not_llm",
    },
    "O-left": {
        **dict.fromkeys(_ALWAYS_UNAVAILABLE, "unavailable"),
        "m1": "owner_cannot_use",
        "m2": "unavailable",
        "m4": "owner_cannot_use",
        "m10": "not_llm",
        "m11": "not_llm",
    },
}
_OWNER_CANNOT_USE_TAIL = (
    " The agent's owner cannot use this model, so tasks started with the "
    "agent's runtime keys would fall back to another model. Choose a model "
    "with owner_can_use true in GET /v1/agents/{agent_id}/assignable-models, or "
    "change it in the web app."
)


def _install_team(scopes: dict[int, AgentTeamScope], team_admin: int) -> None:
    set_agent_team_scope_hook(lambda _db, user_id: scopes.get(int(user_id)))

    def sharers(db, user_id):
        admins = [int(u.id) for u in db.query(User).filter(User.is_admin)]
        scope = scopes.get(int(user_id)) if user_id is not None else None
        return admins + ([team_admin] if scope and scope.team_id == 1 else [])

    set_visible_user_ids_hook(sharers)


def _owner_leaves_team(world: World) -> None:
    """Move the agent's owner out of the team the holder and the agent are in."""
    hana, tara = world.holder, _user_id("tara")
    _install_team(
        {
            hana: AgentTeamScope(team_id=1, is_team_admin=False),
            tara: AgentTeamScope(team_id=1, is_team_admin=True),
        },
        tara,
    )


def _column_world(column: str) -> World:
    world = _build_world(column == "H=O")
    if column == "O-left":
        _owner_leaves_team(world)
    return world


def _agent_owner(world: World) -> int:
    return int(_row(world.agent_id)["user_id"])


@pytest.mark.parametrize("cell", _CELL_NAMES)
@pytest.mark.parametrize("column", _COLUMN_NAMES)
def test_listing_for_agent_matches_patch(column, cell):
    world = _column_world(column)
    model_id = world.models[cell]
    before = _row(world.agent_id)
    headers = _bearer(world.key)

    listed = client.get(
        f"/v1/agents/{world.agent_id}/assignable-models", headers=headers
    ).json()
    in_list = any(row["id"] == model_id and row["owner_can_use"] for row in listed)
    resp = client.patch(
        f"/v1/agents/{world.agent_id}",
        headers=headers,
        json={"models": {"general": model_id}},
    )

    reason = _OUTCOMES[column].get(cell)
    assert (resp.status_code == 200) == (reason is None), (column, cell, resp.text)
    assert in_list == (reason is None), (column, cell)
    if reason is None:
        assert _row(world.agent_id)["models"] == {"general": model_id}
        return
    assert _row(world.agent_id) == before
    entry = {"slot": "general", "id": model_id, "reason": reason}
    if reason == "owner_cannot_use":
        entry["owner_user_id"] = _agent_owner(world)
    error = _error(resp, 400, "model_not_assignable")
    assert error["details"] == {"models": [entry]}
    assert error["message"] == (
        f"These model slots cannot use the requested model: general ({reason})."
        + (_OWNER_CANNOT_USE_TAIL if reason == "owner_cannot_use" else "")
    )


def test_a_stored_model_is_not_checked_when_it_is_sent_back():
    world = _build_world(False)
    inactive = world.models["m5"]
    _set_columns(world.agent_id, models={"general": inactive})

    resp = client.patch(
        f"/v1/agents/{world.agent_id}",
        headers=_bearer(world.key),
        json={"models": {"general": inactive}},
    )
    listed = client.get(
        f"/v1/agents/{world.agent_id}/assignable-models", headers=_bearer(world.key)
    ).json()

    assert resp.status_code == 200, resp.text
    assert inactive not in {row["id"] for row in listed}
    assert resp.json()["unresolved_models"] == ["general"]


def test_slot_rule_has_one_owner():
    world = _build_world(False)
    ok = world.models["m3"]
    calls: list[int] = []
    from xagent.web.services import model_service

    real = model_service.slot_assignable_models

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    headers = _bearer(world.key)
    for send in (
        lambda: client.get(
            f"/v1/agents/{world.agent_id}/assignable-models", headers=headers
        ),
        lambda: client.patch(
            f"/v1/agents/{world.agent_id}",
            headers=headers,
            json={"models": {"general": ok}},
        ),
        lambda: client.get(f"/v1/agents/{world.agent_id}", headers=headers),
    ):
        calls.clear()
        with patch.object(model_service, "slot_assignable_models", spy):
            assert send().status_code == 200
        assert calls, "the request did not go through slot_assignable_models"


# ===== model slots: refusals change nothing =====

_MODEL_REFUSALS = [
    ("unavailable", {"models": {"general": "m9"}}, 400, "(unavailable)"),
    ("not-llm", {"models": {"general": "m10"}}, 400, "(not_llm)"),
    (
        "text-and-model",
        {"description": "new", "models": {"general": "m9"}},
        400,
        "(unavailable)",
    ),
    ("owner-cannot-use", {"models": {"general": "m1"}}, 400, "(owner_cannot_use)"),
    ("owner-left", {"models": {"general": "m4"}}, 400, "(owner_cannot_use)"),
    ("taken-and-model", {"name": "taken", "models": {"general": "m1"}}, 400, _TAKEN),
    (
        "too-long-taken-model",
        {"name": "taken", "instructions": "x" * 10_001, "models": {"general": "m1"}},
        400,
        _TOO_LONG,
    ),
    ("model-read-fails", {"models": {"general": "m3"}}, 500, ""),
]


@pytest.mark.parametrize(
    ("body", "status", "text"),
    [row[1:] for row in _MODEL_REFUSALS],
    ids=[row[0] for row in _MODEL_REFUSALS],
)
def test_rejected_model_update_changes_nothing(request, caplog, body, status, text):
    case = request.node.callspec.id
    world = _column_world("O-left" if case == "owner-left" else "H!=O")
    body = {
        **body,
        "models": {slot: world.models[cell] for slot, cell in body["models"].items()},
    }
    _set_columns(world.agent_id, name="mine")
    _seed(_agent_owner(world), name="taken", team_id=1)
    before = _row(world.agent_id)
    boom = OperationalError("SELECT secret_table", {}, Exception("secret detail"))
    failing = (
        patch(
            "xagent.web.services.model_service.rank_name_matched_models",
            side_effect=boom,
        )
        if status == 500
        else contextlib.nullcontext()
    )

    with patch(CACHE_CLEAR) as cleared, failing, caplog.at_level(logging.INFO):
        resp = client.patch(
            f"/v1/agents/{world.agent_id}", headers=_bearer(world.key), json=body
        )

    assert resp.status_code == status, resp.text
    assert text in resp.json()["error"]["message"]
    assert "secret" not in resp.text
    assert _row(world.agent_id) == before
    cleared.assert_not_called()
    assert _update_records(caplog) == []
    assert _update_records(caplog, UNKNOWN_LOG) == []


# ===== model slots: decided under the row lock =====


def test_update_decides_models_on_the_agent_under_the_lock(monkeypatch):
    holder = Holder()
    me = holder.user_id
    ok = _add_model("ok", links=((me, True, False),))
    other = _add_model("other", links=((me, True, False),))
    agent_id = _seed(me, models={"foo": 1})
    # Another request changes ``visual`` after this one was accepted.
    ran = _before_the_lock(
        monkeypatch,
        lambda: _set_columns(agent_id, models={"foo": 1, "visual": other}),
    )

    resp = _patch(holder, agent_id, {"models": {"general": ok}})

    assert ran == [1], "the change before the lock did not happen"
    assert resp.status_code == 200, resp.text
    assert _row(agent_id)["models"] == {"foo": 1, "visual": other, "general": ok}


@pytest.mark.postgresql
def test_concurrent_slot_updates_both_survive(monkeypatch):
    """Two updates of different slots of one agent read it at once unless the
    row lock keeps them apart."""
    with disposable_database_factory("xagent_v1_agent_update") as make_database:
        engine = make_database("update")
        Base.metadata.create_all(bind=engine)
        factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        monkeypatch.setattr(agent_management, "get_session_local", lambda: factory)

        with factory() as db:
            user = User(username="holder", password_hash="x", is_admin=False)
            db.add(user)
            db.flush()
            pks = []
            for name in ("one", "two"):
                model = _all_models.model.Model(
                    model_id=name,
                    category="llm",
                    model_provider="openai",
                    model_name=name,
                    api_key="test-key",
                    abilities=["chat"],
                    is_active=True,
                )
                db.add(model)
                db.flush()
                db.add(UserModel(user_id=user.id, model_id=model.id, is_owner=True))
                pks.append(int(model.id))
            agent = Agent(user_id=user.id, name="agent")
            db.add(agent)
            db.commit()
            holder_id, agent_id = int(user.id), int(agent.id)

        # Each update waits, after reading the agent, until the other has read
        # it too. Under the row lock the second read waits for the first
        # commit, so the first wait runs out and both slots survive; without
        # the lock both read the old row and one slot is lost. The wait is
        # long so that a second update slow to start still reads the old row.
        barrier = threading.Barrier(2)
        real_changed_slots = agent_management._changed_slots

        def read_then_wait(base, requested):
            with contextlib.suppress(threading.BrokenBarrierError):
                barrier.wait(10)
            return real_changed_slots(base, requested)

        errors: list[BaseException] = []

        def update(slot: str, pk: int) -> None:
            try:
                AgentManagementRuntime._update_agent_sync(
                    user_id=holder_id,
                    key_prefix="xag_personal_t",
                    agent_id=agent_id,
                    changes={"models": {slot: pk}},
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with patch.object(agent_management, "_changed_slots", read_then_wait):
            threads = [
                threading.Thread(target=update, args=("general", pks[0])),
                threading.Thread(target=update, args=("visual", pks[1])),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(30)
        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        with factory() as db:
            stored = db.get(Agent, agent_id).models
        assert stored == {"general": pks[0], "visual": pks[1]}


# ===== the baseline flow a batch client follows =====


class Flow:
    """A key holder's requests, counted."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.count = 0

    def send(self, method: str, url: str, **kwargs: Any):
        self.count += 1
        return client.request(method, url, headers=_bearer(self.key), **kwargs)


def _team_world(bob_leaves: bool):
    """Holder alice administers team 1; bob is a member; three team agents."""
    _admin_headers()
    sysadmin = _user_id("admin")
    alice_key = _key_for(_register_second_user("alice"))["full_key"]
    alice, bob = _user_id("alice"), _add_user("bob")
    _install_team(
        {
            alice: AgentTeamScope(team_id=1, is_team_admin=True),
            bob: AgentTeamScope(team_id=2 if bob_leaves else 1, is_team_admin=False),
        },
        alice,
    )
    stale = _add_model("stale", active=False, links=((sysadmin, True, True),))
    models = {
        "system": _add_model("system", links=((sysadmin, True, True),)),
        "team-admin": _add_model("team-admin", links=((alice, True, True),)),
        "private": _add_model("private", links=((alice, True, False),)),
    }
    agents = [
        _seed_agent(alice, team_id=1, models={"visual": stale}),
        _seed_agent(alice, team_id=1),
        _seed_agent(bob, team_id=1, models={"visual": stale}),
    ]
    return alice_key, alice, bob, agents, models


def _batch_flow(flow: Flow, agents: list[int], pick: dict[int, int]):
    """List, read one agent per owner, list models per owner, then change.

    Returns the change responses and the model listing read for each owner.
    """
    assert flow.send("GET", "/v1/agents").status_code == 200
    for agent_id in (agents[0], agents[2]):
        detail = flow.send("GET", f"/v1/agents/{agent_id}").json()
        assert detail["unresolved_models"] == ["visual"]
    listings = [
        flow.send("GET", f"/v1/agents/{agent_id}/assignable-models").json()
        for agent_id in (agents[0], agents[2])
    ]
    assert all(listings)
    responses = [
        flow.send(
            "PATCH",
            f"/v1/agents/{agent_id}",
            json={"models": {"visual": pick[agent_id]}},
        )
        for agent_id in agents
    ]
    return responses, listings


def _assert_reads_match(flow: Flow, agents: list[int], responses: list[Any]) -> None:
    """Each agent reads back exactly as its change response said."""
    for agent_id, resp in zip(agents, responses):
        assert resp.status_code == 200 and resp.json()["unresolved_models"] == []
        assert flow.send("GET", f"/v1/agents/{agent_id}").json() == resp.json()


def test_batch_client_flow_with_a_shared_model():
    key, _alice, _bob, agents, models = _team_world(False)
    flow = Flow(key)

    responses, _listings = _batch_flow(
        flow, agents, dict.fromkeys(agents, models["system"])
    )
    _assert_reads_match(flow, agents, responses)

    assert flow.count == 1 + 2 + 2 + 3 + 3


def test_batch_client_flow_when_only_one_owner_can_use_the_model():
    key, _alice, bob, agents, models = _team_world(False)
    flow = Flow(key)

    # The holder's private model suits the holder's agents, not bob's.
    responses, listings = _batch_flow(
        flow, agents, dict.fromkeys(agents, models["private"])
    )
    assert [resp.status_code for resp in responses] == [200, 200, 400]
    assert responses[2].json()["error"]["details"]["models"] == [
        {
            "slot": "visual",
            "id": models["private"],
            "reason": "owner_cannot_use",
            "owner_user_id": bob,
        }
    ]
    # bob's listing, already read, marks it, so the client picks a shared model.
    marks = {row["id"]: row["owner_can_use"] for row in listings[1]}
    assert marks[models["private"]] is False and marks[models["system"]] is True
    responses[2] = flow.send(
        "PATCH",
        f"/v1/agents/{agents[2]}",
        json={"models": {"visual": models["system"]}},
    )
    _assert_reads_match(flow, agents, responses)

    # The refused request costs one more request than the shared-model flow.
    assert flow.count == 1 + 2 + 2 + 3 + 3 + 1


def test_batch_client_flow_when_the_owner_left_the_team():
    # Known exception to the baseline: this agent has to be saved in the web app.
    key, _alice, bob, agents, models = _team_world(True)
    flow = Flow(key)

    resp = flow.send(
        "PATCH",
        f"/v1/agents/{agents[2]}",
        json={"models": {"visual": models["team-admin"]}},
    )

    error = _error(resp, 400, "model_not_assignable")
    assert error["details"]["models"] == [
        {
            "slot": "visual",
            "id": models["team-admin"],
            "reason": "owner_cannot_use",
            "owner_user_id": bob,
        }
    ]
