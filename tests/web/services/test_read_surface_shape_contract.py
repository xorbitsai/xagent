"""Shape-contract coverage for the five call sites that switched from the
old transcript reader (``get_latest_waiting_question``) to the read
surface's tuple adapter (``get_pending_interaction_question``).

The switch itself is two textual changes (function name, argument) at
each of the four call sites that invoke
``get_pending_interaction_question`` -- the websocket status-reassertion
event below is not one of those four; it reuses the ``task_info``
event's already-fetched tuple instead of calling the adapter a second
time for a value that cannot have changed. Already covered structurally
by the static guards in ``test_interaction_rollout_guards.py``. What
this file proves is behavioral: that the switch did not change what any
of the five call sites puts on the wire. Each of the compatibility
view's seven tiers is
constructed once (real writes for every tier except the two the database
schema or a code path makes otherwise unreachable, exactly as
``test_task_interaction_service.py`` and ``test_task_interaction_read.py``
already do), then every call site's own wire shape -- an HTTP JSON body
for four of them, a detached in-process snapshot for the other two -- is
asserted against that tier's known projection.

Call sites and how each is driven. Each is exercised through the same
entry point production uses -- no test-only seam is invented for any of
the five, so what these cells pin is the wire shape a client would see:

  1. ``GET /api/chat/task/{id}``            -- chat.py, HTTP
  2. ``GET /api/chat/task/{id}/status``      -- chat.py, HTTP
  3. ``_load_historical_stream_snapshot_sync``'s ``task_info`` event
                                              -- websocket.py, direct call
  4. the same function's waiting-status reassertion event
                                              -- websocket.py, direct call
  5. ``GET /v1/chat/tasks/{id}``             -- v1/tasks.py, HTTP

Sites 1 and 2 share one assertion body (their response shape is
identical); the other three are asserted separately because each
surfaces the same tuple through a different wire shape -- an empty
field, a default message, and a null object respectively.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import xagent.web.api.websocket as websocket_module
import xagent.web.services.task_interaction_service as interaction_service_module
from tests.web.services.task_interaction_schema_shared import anchor_event_id
from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE
from xagent.core.agent.transcript import build_assistant_transcript_content
from xagent.web.api.agents import router as agents_router
from xagent.web.api.auth import auth_router
from xagent.web.api.chat import chat_router
from xagent.web.api.v1 import v1_router
from xagent.web.models.database import (
    Base,
    get_db,
    get_engine,
    get_session_local,
    init_db,
)
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_interaction import (
    INTERACTION_PROTOCOL_VERSION,
    TaskInteractionRequest,
)
from xagent.web.models.user import User
from xagent.web.services import ops_signals
from xagent.web.services import task_interaction_read as read_surface
from xagent.web.services.chat_history_service import persist_assistant_message
from xagent.web.services.task_lease_service import TASK_RUN_ID_TRACE_FIELD

_LEGACY_INTERACTIONS = [{"type": "text_input", "label": "Legacy Control"}]

# The exact shape ``TaskInteractionRequestPayloadV1.interactions[0]``
# dumps to (``mode="json"``) for the single text_input control every
# active-row fixture below writes into ``request_payload``. Matches
# test_task_interaction_read.py's own A9/A15 assertions -- this is the
# native tier's known, fixed shape, not something this file derives.
_NATIVE_INTERACTIONS = [
    {
        "type": "text_input",
        "field": "env",
        "label": "Environment",
        "options": None,
        "placeholder": None,
        "multiline": False,
        "min": None,
        "max": None,
        "default_value": None,
        "accept": None,
        "multiple": False,
    }
]

_NATIVE_QUESTION = "Which environment?"
_DEFAULT_WAITING_MESSAGE = "Task waiting for user response"


@pytest.fixture(autouse=True)
def _restore_ops_signals() -> Iterator[None]:
    """Several tiers below (T2, T3, T3') exercise real degradation paths
    -- ``checkpoint_pk_anchor_dangling``, ``checkpoint_load_unavailable``,
    etc. -- that ``ops_signals.py`` deliberately never clears (see that
    module's docstrings: an unclearable signal is how it reports "this is
    a property of persisted data, not something this process can observe
    getting fixed"). Left alone, those signals stay latched in the
    process-level registry for every test that runs afterward in the same
    process, e.g. ``tests/web/test_health_degradations.py``, whose
    baseline-``/health`` assertion expects the registry empty. Snapshot
    before each test here and restore after, so this file's own
    degradation traffic never leaks past it.
    """
    baseline = ops_signals.active_degradations()
    yield
    current = ops_signals.active_degradations()
    for name in current:
        if name not in baseline:
            ops_signals.clear_degradation(name)
    for name, detail in baseline.items():
        if current.get(name) != detail:
            ops_signals.register_degradation(name, detail)


# ---------------------------------------------------------------------------
# Shared test environment: one FastAPI app wired with every router the five
# call sites need (chat_router for 1/2, v1_router for 5, plus auth_router
# and agents_router to mint the JWT and the agent API key both need), one
# SQLite file, one user, one agent, one API key. Built once per module: the
# five call sites are read-only against tasks this file creates fresh per
# tier, so there is nothing for per-test isolation to protect here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Environment:
    client: TestClient
    session_factory: Callable[[], Session]
    user_id: int
    agent_id: int
    jwt_headers: dict[str, str]
    api_key_headers: dict[str, str]


@pytest.fixture(scope="module")
def _environment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Environment]:
    db_path = tmp_path_factory.mktemp("read_surface_shape_contract") / "test.db"
    init_db(db_url=f"sqlite:///{db_path}")
    engine = get_engine()
    session_factory = get_session_local()

    def _override_get_db() -> Iterator[Session]:
        db = None
        try:
            db = next(get_db())
            yield db
        finally:
            if db is not None:
                db.close()

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(agents_router)
    app.include_router(chat_router)
    app.include_router(v1_router)
    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)

    status_response = client.get("/api/auth/setup-status")
    assert status_response.status_code == 200, status_response.text
    if status_response.json().get("needs_setup", True):
        setup_response = client.post(
            "/api/auth/setup-admin",
            json={
                "username": "admin",
                "email": "admin@example.com",
                "password": "admin123",
            },
        )
        assert setup_response.status_code == 200, setup_response.text

    login = client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin123"}
    )
    assert login.status_code == 200, login.text
    jwt_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    db = session_factory()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        user_id = int(admin.id)
    finally:
        db.close()

    agent_resp = client.post(
        "/api/agents",
        headers=jwt_headers,
        json={
            "name": "shape contract agent",
            "description": "read surface shape contract fixture agent",
            "instructions": "not exercised -- no task in this file runs",
            "execution_mode": "balanced",
        },
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = int(agent_resp.json()["id"])

    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=jwt_headers)
    assert key_resp.status_code == 200, key_resp.text
    api_key_headers = {"Authorization": f"Bearer {key_resp.json()['full_key']}"}

    yield _Environment(
        client=client,
        session_factory=session_factory,
        user_id=user_id,
        agent_id=agent_id,
        jwt_headers=jwt_headers,
        api_key_headers=api_key_headers,
    )

    Base.metadata.drop_all(bind=engine)


# ---------------------------------------------------------------------------
# Tier construction. Every tier is one fresh Task -- own run_id, own
# interaction_protocol_version -- plus whatever rows that tier needs.
# ``source="sdk"`` on every task: the v1 call site's ownership resolver
# (_resolve_task_or_404) only accepts SDK-sourced tasks, and driving all
# five call sites against the identical task rows is the point.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Tier:
    name: str
    task_id: int
    expected_question: str | None
    expected_interactions: list[dict[str, Any]] | None


def _make_task(
    db: Session, *, env: _Environment, marker: int | None, run_id: str
) -> Task:
    task = Task(
        user_id=env.user_id,
        agent_id=env.agent_id,
        title=f"shape-contract-{run_id}",
        description="read surface shape contract fixture task",
        status=TaskStatus.WAITING_FOR_USER,
        source="sdk",
        run_id=run_id,
        interaction_protocol_version=marker,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _make_trace_event(db: Session, *, task_id: int, run_partition: str) -> int:
    event = TraceEvent(
        task_id=task_id,
        event_id=f"shape-contract-trace-{uuid.uuid4()}",
        event_type=str(CHECKPOINT_EVENT_TYPE),
        timestamp=datetime.now(timezone.utc),
        build_id=None,
        data={
            TASK_RUN_ID_TRACE_FIELD: run_partition,
            "checkpoint_type": "agent_execution_checkpoint",
            "execution_id": "exec-1",
        },
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return int(event.id)


def _make_active_row(
    db: Session,
    *,
    task_id: int,
    run_id: str,
    resume_trace_event_id: int,
    resume_run_partition: str,
    request_payload: dict[str, Any] | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    row = TaskInteractionRequest(
        task_id=task_id,
        run_id=run_id,
        kind="clarification",
        protocol_version=INTERACTION_PROTOCOL_VERSION,
        status="active",
        active_slot=1,
        origin="internal",
        request_payload=request_payload
        if request_payload is not None
        else {
            "message": _NATIVE_QUESTION,
            "interactions": [
                {"type": "text_input", "field": "env", "label": "Environment"}
            ],
        },
        request_idempotency_key=f"shape-contract-key-{uuid.uuid4()}",
        resume_trace_event_id=resume_trace_event_id,
        resume_event_id=anchor_event_id(db, resume_trace_event_id),
        resume_execution_id="exec-1",
        resume_locator_format="trace_event_pk_v1",
        resume_checkpoint_type="agent_execution_checkpoint",
        resume_run_partition=resume_run_partition,
        created_at=now,
        expires_at=now + timedelta(minutes=15),
    )
    db.add(row)
    db.commit()


def _build_t0(env: _Environment) -> _Tier:
    """Marker NULL, a live legacy question row. Fast path never calls the
    rich view (proven in test_task_interaction_read.py's A5; not repeated
    here, this file only pins the wire shape)."""

    db = env.session_factory()
    try:
        run_id = f"t0-{uuid.uuid4()}"
        task = _make_task(db, env=env, marker=None, run_id=run_id)
        task_id = int(task.id)
        raw_content = "T0 legacy question"
        persist_assistant_message(
            db,
            task_id,
            env.user_id,
            raw_content,
            message_type="question",
            interactions=_LEGACY_INTERACTIONS,
        )
    finally:
        db.close()
    # persist_assistant_message stores the interaction descriptors folded
    # into the transcript text itself (build_assistant_transcript_content),
    # not just the raw string passed in -- the stored question is that
    # combined form, which is what every call site actually reads back.
    expected_question = build_assistant_transcript_content(
        raw_content, _LEGACY_INTERACTIONS
    )
    return _Tier("T0", task_id, expected_question, _LEGACY_INTERACTIONS)


def _build_t1(env: _Environment) -> _Tier:
    """Marker == 1, no active native row: the rich view falls back to the
    legacy transcript."""

    db = env.session_factory()
    try:
        run_id = f"t1-{uuid.uuid4()}"
        task = _make_task(db, env=env, marker=1, run_id=run_id)
        task_id = int(task.id)
        raw_content = "T1 legacy question"
        persist_assistant_message(
            db,
            task_id,
            env.user_id,
            raw_content,
            message_type="question",
            interactions=_LEGACY_INTERACTIONS,
        )
    finally:
        db.close()
    expected_question = build_assistant_transcript_content(
        raw_content, _LEGACY_INTERACTIONS
    )
    return _Tier("T1", task_id, expected_question, _LEGACY_INTERACTIONS)


def _build_t2(env: _Environment) -> _Tier:
    """Marker == 1, an active row whose anchor resolves: the native
    projection."""

    db = env.session_factory()
    try:
        run_id = f"t2-{uuid.uuid4()}"
        task = _make_task(db, env=env, marker=1, run_id=run_id)
        task_id = int(task.id)
        trace_event_id = _make_trace_event(db, task_id=task_id, run_partition=run_id)
        _make_active_row(
            db,
            task_id=task_id,
            run_id=run_id,
            resume_trace_event_id=trace_event_id,
            resume_run_partition=run_id,
        )
    finally:
        db.close()
    return _Tier("T2", task_id, _NATIVE_QUESTION, _NATIVE_INTERACTIONS)


def _build_t3(env: _Environment) -> _Tier:
    """Marker == 1, an active row whose anchor trace event sits on a
    different run partition -- ``anchor_dangling`` through a real write,
    no monkeypatch. Question text survives, controls do not."""

    db = env.session_factory()
    try:
        run_id = f"t3-{uuid.uuid4()}"
        task = _make_task(db, env=env, marker=1, run_id=run_id)
        task_id = int(task.id)
        trace_event_id = _make_trace_event(
            db, task_id=task_id, run_partition=f"{run_id}-different"
        )
        _make_active_row(
            db,
            task_id=task_id,
            run_id=run_id,
            resume_trace_event_id=trace_event_id,
            resume_run_partition=run_id,
        )
    finally:
        db.close()
    return _Tier("T3", task_id, _NATIVE_QUESTION, None)


def _build_t3_prime(env: _Environment, monkeypatch: pytest.MonkeyPatch) -> _Tier:
    """Marker == 1, an active row whose anchor fetch raises --
    ``checkpoint_unavailable``. The patch matches on (entity, ident) so it
    only intercepts this tier's own trace event lookup, regardless of
    which of the five call sites' own Session instance performs it."""

    db = env.session_factory()
    try:
        run_id = f"t3p-{uuid.uuid4()}"
        task = _make_task(db, env=env, marker=1, run_id=run_id)
        task_id = int(task.id)
        trace_event_id = _make_trace_event(db, task_id=task_id, run_partition=run_id)
        _make_active_row(
            db,
            task_id=task_id,
            run_id=run_id,
            resume_trace_event_id=trace_event_id,
            resume_run_partition=run_id,
        )
    finally:
        db.close()

    original_get = Session.get

    def _raising_get(
        self: Session, entity: Any, ident: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if entity is TraceEvent and ident == trace_event_id:
            raise sa.exc.OperationalError(
                "SELECT 1", {}, Exception("simulated checkpoint fetch failure")
            )
        return original_get(self, entity, ident, *args, **kwargs)

    monkeypatch.setattr(Session, "get", _raising_get)
    return _Tier("T3'", task_id, _NATIVE_QUESTION, None)


def _build_t3_double_prime(env: _Environment, monkeypatch: pytest.MonkeyPatch) -> _Tier:
    """Marker == 1, an active row whose protocol_version the schema's own
    CHECK constraint makes unwritable in this state (active rows are
    pinned to version 1) -- same construction test_task_interaction_read.py's
    A12 uses: a real active row, then the accessor that reads it patched to
    hand back a copy with the field mutated in memory, never persisted.
    The patch requeries through whatever Session the call site passes in,
    so it is safe across all five call sites' own sessions."""

    db = env.session_factory()
    try:
        run_id = f"t3dp-{uuid.uuid4()}"
        task = _make_task(db, env=env, marker=1, run_id=run_id)
        task_id = int(task.id)
        trace_event_id = _make_trace_event(db, task_id=task_id, run_partition=run_id)
        _make_active_row(
            db,
            task_id=task_id,
            run_id=run_id,
            resume_trace_event_id=trace_event_id,
            resume_run_partition=run_id,
        )
    finally:
        db.close()

    original_active_row = interaction_service_module._active_native_row

    def _fake_active_row(db2: Session, active_task_id: int) -> Any:
        row = original_active_row(db2, active_task_id)
        if row is not None and active_task_id == task_id:
            row.protocol_version = 2
        return row

    monkeypatch.setattr(
        interaction_service_module, "_active_native_row", _fake_active_row
    )
    return _Tier("T3''", task_id, None, None)


def _build_t3_triple_prime(env: _Environment) -> _Tier:
    """Marker == 1, an active row whose request_payload does not parse
    against the v1 shape -- a real write, no monkeypatch needed: payload
    parsing runs before anchor resolution, so this is reachable without
    faking anything about the trace event."""

    db = env.session_factory()
    try:
        run_id = f"t3tp-{uuid.uuid4()}"
        task = _make_task(db, env=env, marker=1, run_id=run_id)
        task_id = int(task.id)
        trace_event_id = _make_trace_event(db, task_id=task_id, run_partition=run_id)
        _make_active_row(
            db,
            task_id=task_id,
            run_id=run_id,
            resume_trace_event_id=trace_event_id,
            resume_run_partition=run_id,
            request_payload={"not": "a valid v1 payload"},
        )
    finally:
        db.close()
    return _Tier("T3'''", task_id, None, None)


_TIER_BUILDERS: dict[str, Callable[[_Environment, pytest.MonkeyPatch], _Tier]] = {
    "T0": lambda env, monkeypatch: _build_t0(env),
    "T1": lambda env, monkeypatch: _build_t1(env),
    "T2": lambda env, monkeypatch: _build_t2(env),
    "T3": lambda env, monkeypatch: _build_t3(env),
    "T3'": _build_t3_prime,
    "T3''": _build_t3_double_prime,
    "T3'''": lambda env, monkeypatch: _build_t3_triple_prime(env),
}

_TIER_NAMES = list(_TIER_BUILDERS)


def _build_tier(
    tier_name: str, env: _Environment, monkeypatch: pytest.MonkeyPatch
) -> _Tier:
    return _TIER_BUILDERS[tier_name](env, monkeypatch)


def _disable_websocket_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(websocket_module, "cache_get", lambda *_a, **_k: None)
    monkeypatch.setattr(websocket_module, "cache_set", lambda *_a, **_k: None)


def _task_info_event_data(snapshot: Any) -> dict[str, Any]:
    matches = [e for e in snapshot.events if e.get("event_type") == "task_info"]
    assert len(matches) == 1, snapshot.events
    return dict(matches[0]["data"])


def _status_reassertion_event(snapshot: Any) -> dict[str, Any]:
    matches = [e for e in snapshot.events if e.get("type") == "task_waiting_for_user"]
    assert len(matches) == 1, snapshot.events
    return dict(matches[0])


# ---------------------------------------------------------------------------
# The seven-tier matrix, one assertion body per call-site shape. Sites 1
# and 2 (chat.py's two HTTP endpoints) are identical in shape, so they
# share one parametrized test; sites 3, 4, 5 each get their own because
# each surfaces the same tuple through a different wire shape.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier_name", _TIER_NAMES)
def test_chat_detail_and_status_project_the_tier(
    tier_name: str, _environment: _Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    tier = _build_tier(tier_name, _environment, monkeypatch)
    for path in (
        f"/api/chat/task/{tier.task_id}",
        f"/api/chat/task/{tier.task_id}/status",
    ):
        resp = _environment.client.get(path, headers=_environment.jwt_headers)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["waiting_question"] == tier.expected_question, path
        assert body["waiting_interactions"] == tier.expected_interactions, path


@pytest.mark.parametrize("tier_name", _TIER_NAMES)
def test_websocket_task_info_event_projects_the_tier(
    tier_name: str, _environment: _Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    tier = _build_tier(tier_name, _environment, monkeypatch)
    _disable_websocket_cache(monkeypatch)

    snapshot = websocket_module._load_historical_stream_snapshot_sync(
        tier.task_id,
        actor_user_id=_environment.user_id,
        actor_is_admin=False,
    )
    assert snapshot is not None

    data = _task_info_event_data(snapshot)
    assert data["waiting_question"] == tier.expected_question
    assert data["waiting_interactions"] == tier.expected_interactions


@pytest.mark.parametrize("tier_name", _TIER_NAMES)
def test_websocket_status_reassertion_projects_the_tier(
    tier_name: str, _environment: _Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    tier = _build_tier(tier_name, _environment, monkeypatch)
    _disable_websocket_cache(monkeypatch)

    snapshot = websocket_module._load_historical_stream_snapshot_sync(
        tier.task_id,
        actor_user_id=_environment.user_id,
        actor_is_admin=False,
    )
    assert snapshot is not None

    event = _status_reassertion_event(snapshot)
    if tier.expected_question:
        assert event["message"] == tier.expected_question
        assert event["question"] == tier.expected_question
    else:
        assert event["message"] == _DEFAULT_WAITING_MESSAGE
        assert "question" not in event
    if isinstance(tier.expected_interactions, list):
        assert event["interactions"] == tier.expected_interactions
    else:
        assert "interactions" not in event


@pytest.mark.parametrize("tier_name", _TIER_NAMES)
def test_v1_task_snapshot_projects_the_tier(
    tier_name: str, _environment: _Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    tier = _build_tier(tier_name, _environment, monkeypatch)
    resp = _environment.client.get(
        f"/v1/chat/tasks/{tier.task_id}", headers=_environment.api_key_headers
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    if tier.expected_question is None:
        assert body["pending_interaction"] is None
    else:
        assert body["pending_interaction"]["question"] == tier.expected_question
        assert body["pending_interaction"]["interactions"] == tier.expected_interactions


# ---------------------------------------------------------------------------
# Both slots empty is one outcome (T3'''), but it looks different on each
# of the three distinct wire shapes -- pinned individually so each is a
# named, traceable cell rather than only an incidental pass inside the
# matrix above.
# ---------------------------------------------------------------------------


def test_chat_task_detail_shows_an_empty_field_when_both_slots_are_empty(
    _environment: _Environment,
) -> None:
    tier = _build_t3_triple_prime(_environment)
    resp = _environment.client.get(
        f"/api/chat/task/{tier.task_id}", headers=_environment.jwt_headers
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["waiting_question"] is None
    assert body["waiting_interactions"] is None


def test_websocket_status_reassertion_falls_back_to_default_message_when_both_slots_are_empty(
    _environment: _Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    tier = _build_t3_triple_prime(_environment)
    _disable_websocket_cache(monkeypatch)

    snapshot = websocket_module._load_historical_stream_snapshot_sync(
        tier.task_id,
        actor_user_id=_environment.user_id,
        actor_is_admin=False,
    )
    assert snapshot is not None

    event = _status_reassertion_event(snapshot)
    assert event["message"] == _DEFAULT_WAITING_MESSAGE
    assert "question" not in event
    assert "interactions" not in event


def test_v1_pending_interaction_is_null_when_both_slots_are_empty(
    _environment: _Environment,
) -> None:
    tier = _build_t3_triple_prime(_environment)
    resp = _environment.client.get(
        f"/v1/chat/tasks/{tier.task_id}", headers=_environment.api_key_headers
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["pending_interaction"] is None


# ---------------------------------------------------------------------------
# Mixed-version matrix, old writer / new reader: a question the legacy writer wrote is
# still readable through the new read surface, whether or not the reading
# task ever picked up a protocol marker. Asserted at the read surface's
# own entry point, not through any one call site -- the point is the
# adapter's own contract, which every call site inherits identically.
# ---------------------------------------------------------------------------


def test_old_writer_new_reader_legacy_question_is_still_readable_without_a_marker(
    _environment: _Environment,
) -> None:
    db = _environment.session_factory()
    try:
        run_id = f"mixed-null-marker-{uuid.uuid4()}"
        task = _make_task(db, env=_environment, marker=None, run_id=run_id)
        persist_assistant_message(
            db,
            int(task.id),
            _environment.user_id,
            "Old writer, no marker ever set",
            message_type="question",
        )
        db.refresh(task)

        question, interactions = read_surface.get_pending_interaction_question(db, task)

        assert question is not None
        assert question.startswith("Old writer, no marker ever set")
        assert interactions is None
    finally:
        db.close()


def test_a_relabelled_question_reaches_the_wire_instead_of_the_default_message(
    _environment: _Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The superseded fallback's outcome on the one wire shape that has
    something to fall back to: the websocket status reassertion event. A
    task whose only question row carries ``question_superseded`` still
    reaches this event with the question itself, not with the generic
    waiting message the event falls back to when both slots are empty."""

    db = _environment.session_factory()
    try:
        run_id = f"relabelled-{uuid.uuid4()}"
        task = _make_task(db, env=_environment, marker=None, run_id=run_id)
        task_id = int(task.id)
        raw_content = "Relabelled question"
        persist_assistant_message(
            db,
            task_id,
            _environment.user_id,
            raw_content,
            message_type="question_superseded",
            interactions=_LEGACY_INTERACTIONS,
        )
    finally:
        db.close()
    expected_question = build_assistant_transcript_content(
        raw_content, _LEGACY_INTERACTIONS
    )

    _disable_websocket_cache(monkeypatch)
    snapshot = websocket_module._load_historical_stream_snapshot_sync(
        task_id,
        actor_user_id=_environment.user_id,
        actor_is_admin=False,
    )
    assert snapshot is not None

    event = _status_reassertion_event(snapshot)
    assert event["message"] != _DEFAULT_WAITING_MESSAGE
    assert event["message"] == expected_question
    assert event["question"] == expected_question
    assert event["interactions"] == _LEGACY_INTERACTIONS


def test_old_writer_new_reader_legacy_question_is_still_readable_with_no_active_row(
    _environment: _Environment,
) -> None:
    """Old writer, new reader, marker present: the reading task has
    picked up the marker (interaction_protocol_version == 1) but the
    interaction table holds no active row for it -- the rich view's own
    "no active row" tier -- so the legacy writer's question is still the
    honest answer."""

    db = _environment.session_factory()
    try:
        run_id = f"mixed-marker-one-no-active-row-{uuid.uuid4()}"
        task = _make_task(db, env=_environment, marker=1, run_id=run_id)
        persist_assistant_message(
            db,
            int(task.id),
            _environment.user_id,
            "New reader, no active native row",
            message_type="question",
        )
        db.refresh(task)

        question, interactions = read_surface.get_pending_interaction_question(db, task)

        assert question is not None
        assert question.startswith("New reader, no active native row")
        assert interactions is None
    finally:
        db.close()
