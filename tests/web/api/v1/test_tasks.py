"""Integration tests for /v1/chat/tasks/* endpoints.

Endpoints covered:
  - POST /v1/chat/tasks
  - POST /v1/chat/tasks/{id}/messages
  - GET  /v1/chat/tasks/{id}
  - GET  /v1/chat/tasks/{id}/steps

Tests mock the background-execution kickoff so the suite doesn't need
to spin up an actual AgentService / LLM. The behaviors under test are
HTTP shape + DB rows + which background helper was called with which
arguments -- not the LLM call itself. The steps endpoint exercises
real :class:`TraceEvent` rows inserted directly into the test DB to
drive the mapping.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec
from xagent.web.api.v1 import tasks as v1_tasks
from xagent.web.models.agent import Agent
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.task import (
    Task,
    TaskConnectorRuntimeContext,
    TaskStatus,
    TraceEvent,
)
from xagent.web.services.connector_runtime import (
    ConnectorRuntimeValues,
    drop_ephemeral_runtime_values_for_testing,
    get_ephemeral_runtime_values,
    load_connector_runtime_view,
    pop_ephemeral_runtime_values,
    set_connector_runtime_resolver_for_testing,
    store_ephemeral_runtime_values,
)
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    cache_get,
    set_cache_backend_for_testing,
    task_steps_key,
)
from xagent.web.services.mcp_oauth import MCPAuthorizationChallenge
from xagent.web.tools.config import (
    ResolvedToken,
    TokenRequest,
    WebToolConfig,
    set_oauth_token_resolver_hook,
)

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)


def _create_agent_with_key_for(headers: dict[str, str]) -> tuple[int, str]:
    """Create an agent + api key under the user owning ``headers``."""
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "v1 tasks test agent",
            "description": "test",
            "instructions": "you are a test agent",
            "execution_mode": "balanced",
        },
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]
    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    return agent_id, key_resp.json()["full_key"]


# Opt this file into the shared conftest ``_test_db`` fixture; see the
# note in test_agent_api_keys.py for why we use ``usefixtures`` with a
# string name rather than importing the fixture.
pytestmark = pytest.mark.usefixtures("_test_db")


# ===== helpers =====


def _create_agent_with_key() -> tuple[int, str]:
    """Create one agent under the admin user + generate its API key.

    Returns: (agent_id, full_key)
    """
    headers = _admin_headers()
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "v1 tasks test agent",
            "description": "test",
            "instructions": "you are a test agent",
            "execution_mode": "balanced",
        },
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]

    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    return agent_id, key_resp.json()["full_key"]


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


def _install_runtime_mcp_connector(
    agent_id: int,
    *,
    name: str = "ShiftCare",
    selected: bool = True,
    required: bool = True,
    auth_selector_required: bool = False,
    secret_required: bool = False,
    delegated_authorization_binding: bool = False,
) -> int:
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        runtime_input_schema = {
            "context": {
                "account_id": {
                    "type": "string",
                    "required": required,
                }
            }
        }
        if auth_selector_required:
            runtime_input_schema["auth_selector"] = {
                "resource_owner_key": {
                    "type": "string",
                    "required": True,
                }
            }
        if secret_required:
            runtime_input_schema["secrets"] = {
                "authorization": {
                    "type": "string",
                    "required": True,
                }
            }

        runtime_bindings = [
            {
                "source": {
                    "input_type": "context",
                    "key": "account_id",
                },
                "target": {
                    "target_type": "mcp_meta",
                    "key": "account_id",
                },
            }
        ]
        if delegated_authorization_binding:
            runtime_bindings.append(
                {
                    "source": {
                        "input_type": "secrets",
                        "key": "authorization",
                    },
                    "target": {
                        "target_type": "transport_headers",
                        "key": "Authorization",
                    },
                }
            )

        server = MCPServer(
            name=name,
            description=f"{name} MCP",
            managed="external",
            transport="streamable_http",
            url=f"https://{name.lower()}.mcp.test",
            runtime_input_schema=runtime_input_schema,
            runtime_bindings=runtime_bindings,
            allow_delegated_authorization=delegated_authorization_binding,
        )
        db.add(server)
        db.flush()
        db.add(
            UserMCPServer(
                user_id=int(agent.user_id),
                mcpserver_id=int(server.id),
                is_owner=True,
                can_edit=True,
                is_active=True,
            )
        )
        if selected:
            agent.tool_categories = [f"mcp:{name}"]
        db.commit()
        return int(server.id)
    finally:
        db.close()


# We mock ``TaskTurnOrchestrator._schedule_bg`` (the actual bg coroutine
# spawn) rather than the public ``start_new_turn`` / ``append_turn`` so
# the orchestrator's claim + persist logic still runs and tests can
# verify DB writes (atomic claim flipped status, user messages got
# persisted, etc.). Only the asyncio.create_task / agent execution is
# stubbed.
#
# Scope: file-local. ``autouse=True`` means every test in this module
# gets the mock automatically. GET-only tests (which never call the
# orchestrator) are unaffected; POST tests assert on ``await_count`` /
# ``await_args``. Other test files (e.g. test_steps_mapping.py,
# test_auth.py) are NOT affected because pytest fixture scoping is
# per-module.
#
# Fixture name kept as ``mock_start_task`` to minimize churn across the
# existing test surface; conceptually it now mocks "bg scheduling".
@pytest.fixture(autouse=True)
def mock_start_task():
    # Patch the lease-aware bg scheduler so the orchestrator's atomic
    # claim + transcript persist logic still runs against a real DB;
    # only the asyncio.create_task / agent execution is stubbed.
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


# ===== POST /v1/chat/tasks =====


def test_create_task_happy_path(mock_start_task):
    """Returns 202 + task_id, writes hidden SDK Task + input,
    persists first user message, kicks off background, and leaves the
    task readable through the SDK API surface.
    """
    agent_id, full_key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first user message"},
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["agent_id"] == agent_id
    # POST atomically claims RUNNING before returning 202, so the
    # response body reports the post-claim state, not 'pending'.
    assert body["status"] == "running"
    assert body["run_id"]
    assert body["state_version"] == 1
    assert body["control_state"] == "running"
    assert "task_id" in body
    assert "created_at" in body
    task_id = body["task_id"]

    # DB: Task row exists, owned by admin user, source='sdk', input set.
    # POST atomically claims RUNNING before returning 202, so the row
    # is already RUNNING from the moment the response lands.
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        assert task is not None
        assert task.agent_id == agent_id
        assert task.source == "sdk"
        assert task.is_visible is False
        assert task.input == "first user message"
        assert task.status == TaskStatus.RUNNING

        # task_chat_messages: one user-role message written
        from xagent.web.models.chat_message import TaskChatMessage

        msgs = (
            db.query(TaskChatMessage).filter(TaskChatMessage.task_id == task_id).all()
        )
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "first user message"
    finally:
        db.close()

    sdk_task = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert sdk_task.status_code == 200, sdk_task.text
    assert sdk_task.json()["task_id"] == task_id
    assert sdk_task.json()["run_id"] == body["run_id"]
    assert sdk_task.json()["state_version"] == body["state_version"]
    assert sdk_task.json()["control_state"] == "running"

    # Background kickoff was called exactly once for this task. The
    # scheduler receives a ``TaskTurnPayload`` carrying both transcript
    # and execution channels.
    assert mock_start_task.call_count == 1
    kwargs = mock_start_task.call_args.kwargs
    assert kwargs["task_id"] == task_id
    assert kwargs["payload"].transcript_message == "first user message"


def test_create_task_records_key_usage_but_polling_does_not(mock_start_task):
    """Creating/appending tasks bumps usage; polling status/steps does not."""

    from xagent.web.models.agent_api_key import AgentApiKey

    agent_id, full_key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "hello"},
        },
    )
    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]

    db = _direct_db_session()
    try:
        row = db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).first()
        assert row is not None
        assert row.last_used_at is not None
        assert row.usage_month == datetime.now(UTC).strftime("%Y-%m")
        assert row.usage_month_calls == 1
    finally:
        db.close()

    for _ in range(3):
        assert (
            client.get(
                f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key)
            ).status_code
            == 200
        )
        assert (
            client.get(
                f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
            ).status_code
            == 200
        )

    db = _direct_db_session()
    try:
        row = db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).first()
        assert row.usage_month_calls == 1
        db.query(Task).filter(Task.id == task_id).update(
            {"status": TaskStatus.COMPLETED}
        )
        db.commit()
    finally:
        db.close()

    appended = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "second turn"},
        },
    )
    assert appended.status_code == 202, appended.text

    db = _direct_db_session()
    try:
        row = db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).first()
        assert row.usage_month_calls == 2
    finally:
        db.close()


# ===== POST /v1/chat/files + message.files =====


def test_upload_and_attach_files_to_task(mock_start_task):
    """Upload via /v1/chat/files, then attach the file_id to a task turn.

    Asserts the returned file_id round-trips into the turn payload: the
    execution_message carries the file reference context (so the agent
    sees it) while the transcript stays the raw user text.
    """
    agent_id, full_key = _create_agent_with_key()

    up = client.post(
        "/v1/chat/files",
        headers=_bearer(full_key),
        files=[("files", ("lesson_plan.txt", b"lesson content", "text/plain"))],
    )
    assert up.status_code == 200, up.text
    files = up.json()["files"]
    assert len(files) == 1
    file_id = files[0]["file_id"]
    assert files[0]["filename"] == "lesson_plan.txt"

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {
                "role": "user",
                "content": "check this lesson plan",
                "files": [file_id],
            },
        },
    )
    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]

    payload = mock_start_task.call_args.kwargs["payload"]
    # Transcript is the raw text; execution channel is file-enriched.
    assert payload.transcript_message == "check this lesson plan"
    assert payload.execution_message is not None
    assert file_id in payload.execution_message
    assert "UPLOADED FILES" in payload.execution_message
    assert payload.attachments
    assert any(a.get("file_id") == file_id for a in payload.attachments)

    # File is bound to the task after the turn is claimed (not before).
    from xagent.web.models.uploaded_file import UploadedFile

    db = _direct_db_session()
    try:
        rec = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
        assert rec is not None
        assert rec.task_id == task_id
    finally:
        db.close()


def test_upload_rejects_unsupported_type_with_v1_envelope(mock_start_task):
    """Unsupported extension -> 400 invalid_input in the v1 envelope.

    Guards against the shared ``store_uploaded_files`` leaking its bare
    HTTPException (a 500 {"detail": ...}) past the v1 error handler.
    """
    _agent_id, full_key = _create_agent_with_key()

    up = client.post(
        "/v1/chat/files",
        headers=_bearer(full_key),
        files=[("files", ("payload.xyz", b"junk", "application/octet-stream"))],
    )
    assert up.status_code == 400, up.text
    body = up.json()
    assert body["error"]["code"] == "invalid_input"
    # Must be the stable v1 envelope, not FastAPI's default {"detail": ...}.
    assert "detail" not in body


def test_attach_unknown_file_id_is_rejected_without_orphan(mock_start_task):
    """A bad file_id -> 400 and NO task row is created (no orphan)."""
    agent_id, full_key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {
                "role": "user",
                "content": "check this",
                "files": ["does-not-exist"],
            },
        },
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "invalid_input"

    # #2 regression: validation happens before task creation, so nothing is
    # left behind. The background kickoff must never have fired either.
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.agent_id == agent_id).count() == 0, (
            "a bad file id must not leave an orphan task"
        )
    finally:
        db.close()
    assert mock_start_task.call_count == 0


def test_partial_file_set_is_all_or_nothing(mock_start_task):
    """[good, bad] -> 400; the good file is left unbound (not half-attached)."""
    agent_id, full_key = _create_agent_with_key()

    up = client.post(
        "/v1/chat/files",
        headers=_bearer(full_key),
        files=[("files", ("good.txt", b"content", "text/plain"))],
    )
    good_id = up.json()["files"][0]["file_id"]

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {
                "role": "user",
                "content": "check these",
                "files": [good_id, "typo-id"],
            },
        },
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "invalid_input"

    from xagent.web.models.uploaded_file import UploadedFile

    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.agent_id == agent_id).count() == 0
        rec = db.query(UploadedFile).filter(UploadedFile.file_id == good_id).first()
        assert rec is not None
        assert rec.task_id is None, "good file must stay unbound on all-or-nothing"
    finally:
        db.close()
    assert mock_start_task.call_count == 0


def test_repeated_file_id_is_deduped(mock_start_task):
    """The same file_id twice in one request attaches once, not twice."""
    agent_id, full_key = _create_agent_with_key()

    up = client.post(
        "/v1/chat/files",
        headers=_bearer(full_key),
        files=[("files", ("dup.txt", b"content", "text/plain"))],
    )
    file_id = up.json()["files"][0]["file_id"]

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {
                "role": "user",
                "content": "check it",
                "files": [file_id, file_id],
            },
        },
    )
    assert resp.status_code == 202, resp.text

    payload = mock_start_task.call_args.kwargs["payload"]
    # One context line and one chip, despite the duplicate id.
    assert payload.execution_message.count(f"file_id={file_id}") == 1
    assert [a.get("file_id") for a in payload.attachments] == [file_id]


def test_append_message_with_files(mock_start_task):
    """The append route also accepts and attaches files."""
    agent_id, full_key = _create_agent_with_key()

    created = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "turn one"},
        },
    )
    task_id = created.json()["task_id"]

    # Append is only allowed once the task leaves RUNNING.
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).update(
            {"status": TaskStatus.COMPLETED}
        )
        db.commit()
    finally:
        db.close()

    up = client.post(
        "/v1/chat/files",
        headers=_bearer(full_key),
        files=[("files", ("turn2.txt", b"more", "text/plain"))],
    )
    file_id = up.json()["files"][0]["file_id"]

    appended = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {
                "role": "user",
                "content": "look at this too",
                "files": [file_id],
            },
        },
    )
    assert appended.status_code == 202, appended.text

    payload = mock_start_task.call_args.kwargs["payload"]
    assert payload.transcript_message == "look at this too"
    assert file_id in payload.execution_message
    assert any(a.get("file_id") == file_id for a in payload.attachments)

    from xagent.web.models.uploaded_file import UploadedFile

    db = _direct_db_session()
    try:
        rec = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
        assert rec is not None
        assert rec.task_id == task_id
    finally:
        db.close()


def test_upload_files_requires_api_key(mock_start_task):
    """POST /v1/chat/files without a key -> 401 invalid_api_key envelope."""
    resp = client.post(
        "/v1/chat/files",
        files=[("files", ("x.txt", b"data", "text/plain"))],
    )
    assert resp.status_code == 401, resp.text
    assert resp.json()["error"]["code"] == "invalid_api_key"


def test_upload_oversized_maps_to_413(mock_start_task):
    """A 413 from the shared uploader surfaces as a v1-enveloped 413."""
    _agent_id, full_key = _create_agent_with_key()

    with patch(
        "xagent.web.api.files.store_uploaded_files",
        new=AsyncMock(side_effect=HTTPException(status_code=413, detail="too big")),
    ):
        resp = client.post(
            "/v1/chat/files",
            headers=_bearer(full_key),
            files=[("files", ("big.txt", b"data", "text/plain"))],
        )
    assert resp.status_code == 413, resp.text
    body = resp.json()
    assert body["error"]["code"] == "invalid_input"
    assert "detail" not in body  # not FastAPI's default envelope


def test_upload_storage_unavailable_maps_to_503(mock_start_task):
    """A 503 from the shared uploader surfaces as a retryable v1 503."""
    _agent_id, full_key = _create_agent_with_key()

    with patch(
        "xagent.web.api.files.store_uploaded_files",
        new=AsyncMock(side_effect=HTTPException(status_code=503, detail="down")),
    ):
        resp = client.post(
            "/v1/chat/files",
            headers=_bearer(full_key),
            files=[("files", ("f.txt", b"data", "text/plain"))],
        )
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["error"]["code"] == "internal_error"
    assert "detail" not in body


def test_cross_user_file_not_attachable(mock_start_task):
    """User B's agent key cannot attach a file uploaded by user A."""
    # User A (admin) uploads a file under their own agent key.
    _agent_a, key_a = _create_agent_with_key()
    up = client.post(
        "/v1/chat/files",
        headers=_bearer(key_a),
        files=[("files", ("secret.txt", b"private", "text/plain"))],
    )
    file_id_a = up.json()["files"][0]["file_id"]

    # User B has a separate account, agent, and key.
    bob_headers = _register_second_user()
    agent_b, key_b = _create_agent_with_key_for(bob_headers)

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(key_b),
        json={
            "agent_id": agent_b,
            "message": {
                "role": "user",
                "content": "read A's file",
                "files": [file_id_a],
            },
        },
    )
    # user_id filter makes A's file invisible to B -> unresolvable -> 400.
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "invalid_input"

    # A's file must remain unbound (B's failed attempt cannot touch it).
    from xagent.web.models.uploaded_file import UploadedFile

    db = _direct_db_session()
    try:
        rec = db.query(UploadedFile).filter(UploadedFile.file_id == file_id_a).first()
        assert rec is not None
        assert rec.task_id is None
    finally:
        db.close()


def test_create_task_persists_connector_runtime_snapshot_and_context(
    mock_start_task,
):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(agent_id)

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first user message"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"account_id": "6185"},
                }
            ],
        },
    )

    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.connector_runtime_selected_refs == [
            {"connector_type": "mcp", "connector_id": server_id}
        ]
        context_row = (
            db.query(TaskConnectorRuntimeContext)
            .filter(TaskConnectorRuntimeContext.task_id == task_id)
            .one()
        )
        assert context_row.connector_type == "mcp"
        assert context_row.connector_id == server_id
        assert context_row.context == {"account_id": "6185"}
    finally:
        db.close()

    assert mock_start_task.call_count == 1


def test_create_task_rejects_runtime_context_for_unselected_connector(
    mock_start_task,
):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(agent_id, selected=False)

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first user message"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"account_id": "6185"},
                }
            ],
        },
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_runtime_context"
    assert resp.json()["error"]["details"]["reason"] == "connector_not_selected"
    assert mock_start_task.call_count == 0


def test_create_task_rejects_missing_required_runtime_context(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    _install_runtime_mcp_connector(agent_id)

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first user message"},
        },
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "missing_runtime_context"
    assert mock_start_task.call_count == 0


def test_create_task_rejects_missing_required_auth_selector(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(
        agent_id, required=False, auth_selector_required=True
    )

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first user message"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    }
                }
            ],
        },
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "runtime_secret_unavailable"
    assert resp.json()["error"]["details"]["reason"] == "not_provided"
    assert mock_start_task.call_count == 0


def test_create_task_rejects_missing_required_secret_without_resolver(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(
        agent_id, required=False, secret_required=True
    )

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first user message"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    }
                }
            ],
        },
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "runtime_secret_unavailable"
    assert resp.json()["error"]["details"]["reason"] == "not_provided"
    assert mock_start_task.call_count == 0


def test_connector_runtime_view_loads_task_context(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(agent_id)

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first user message"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"account_id": "6185"},
                }
            ],
        },
    )

    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]
    turn_id = mock_start_task.call_args.kwargs["payload"].turn_id
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        view = load_connector_runtime_view(
            db=db,
            task_id=task_id,
            turn_id=turn_id,
            user_id=int(task.user_id),
        )
    finally:
        db.close()

    assert view == {
        f"mcp:{server_id}": {
            "context": {"account_id": "6185"},
            "secrets": {},
            "auth_selector": {},
        }
    }


def test_connector_runtime_view_reports_store_lost_for_missing_ephemeral(
    mock_start_task,
):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(
        agent_id, required=False, secret_required=True
    )

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first user message"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "secrets": {"authorization": "Bearer tenant-token"},
                }
            ],
        },
    )

    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]
    turn_id = mock_start_task.call_args.kwargs["payload"].turn_id
    drop_ephemeral_runtime_values_for_testing(turn_id)
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        with pytest.raises(Exception) as exc_info:
            load_connector_runtime_view(
                db=db,
                task_id=task_id,
                turn_id=turn_id,
                user_id=int(task.user_id),
            )
    finally:
        db.close()

    assert getattr(exc_info.value, "code", None) == "runtime_secret_unavailable"
    assert getattr(exc_info.value, "details", {}).get("reason") == "store_lost"


def test_connector_runtime_resolver_can_supply_required_secret(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(
        agent_id, required=False, secret_required=True
    )

    def _resolver(request):
        assert request.connector_ref.connector_id == server_id
        assert request.values.secrets == {}
        return ConnectorRuntimeValues(
            context={},
            secrets={"authorization": "Bearer hook-token"},
            auth_selector={},
        )

    set_connector_runtime_resolver_for_testing(_resolver)
    try:
        resp = client.post(
            "/v1/chat/tasks",
            headers=_bearer(full_key),
            json={
                "agent_id": agent_id,
                "message": {"role": "user", "content": "first user message"},
            },
        )
    finally:
        set_connector_runtime_resolver_for_testing(None)

    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]
    turn_id = mock_start_task.call_args.kwargs["payload"].turn_id
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        set_connector_runtime_resolver_for_testing(_resolver)
        try:
            view = load_connector_runtime_view(
                db=db,
                task_id=task_id,
                turn_id=turn_id,
                user_id=int(task.user_id),
            )
        finally:
            set_connector_runtime_resolver_for_testing(None)
    finally:
        db.close()

    assert view[f"mcp:{server_id}"]["secrets"] == {"authorization": "Bearer hook-token"}


def test_connector_runtime_resolver_can_supply_required_auth_selector(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(
        agent_id, required=False, auth_selector_required=True
    )

    def _resolver(request):
        assert request.connector_ref.connector_id == server_id
        assert request.values.auth_selector == {}
        return ConnectorRuntimeValues(
            context={},
            secrets={},
            auth_selector={"resource_owner_key": "xagent:user:owner"},
        )

    set_connector_runtime_resolver_for_testing(_resolver)
    try:
        resp = client.post(
            "/v1/chat/tasks",
            headers=_bearer(full_key),
            json={
                "agent_id": agent_id,
                "message": {"role": "user", "content": "first user message"},
            },
        )
    finally:
        set_connector_runtime_resolver_for_testing(None)

    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]
    turn_id = mock_start_task.call_args.kwargs["payload"].turn_id
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        set_connector_runtime_resolver_for_testing(_resolver)
        try:
            view = load_connector_runtime_view(
                db=db,
                task_id=task_id,
                turn_id=turn_id,
                user_id=int(task.user_id),
            )
        finally:
            set_connector_runtime_resolver_for_testing(None)
    finally:
        db.close()

    assert view[f"mcp:{server_id}"]["auth_selector"] == {
        "resource_owner_key": "xagent:user:owner"
    }


def test_connector_runtime_view_reports_not_provided_when_resolver_omits_secret(
    mock_start_task,
):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(
        agent_id, required=False, secret_required=True
    )

    def _resolver(_request):
        return None

    set_connector_runtime_resolver_for_testing(_resolver)
    try:
        resp = client.post(
            "/v1/chat/tasks",
            headers=_bearer(full_key),
            json={
                "agent_id": agent_id,
                "message": {"role": "user", "content": "first user message"},
            },
        )
    finally:
        set_connector_runtime_resolver_for_testing(None)

    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]
    turn_id = mock_start_task.call_args.kwargs["payload"].turn_id
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        set_connector_runtime_resolver_for_testing(_resolver)
        try:
            with pytest.raises(Exception) as exc_info:
                load_connector_runtime_view(
                    db=db,
                    task_id=task_id,
                    turn_id=turn_id,
                    user_id=int(task.user_id),
                )
        finally:
            set_connector_runtime_resolver_for_testing(None)
    finally:
        db.close()

    assert getattr(exc_info.value, "code", None) == "runtime_secret_unavailable"
    assert getattr(exc_info.value, "details", {}).get("reason") == "not_provided"
    assert getattr(exc_info.value, "details", {}).get("connector_ref") == {
        "connector_type": "mcp",
        "connector_id": server_id,
    }


def test_create_task_marks_failed_when_runtime_secret_store_fails(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(
        agent_id, required=False, secret_required=True
    )

    with patch(
        "xagent.web.api.v1.tasks.store_ephemeral_runtime_values",
        side_effect=RuntimeError("store failed for Bearer tenant-token"),
    ):
        resp = client.post(
            "/v1/chat/tasks",
            headers=_bearer(full_key),
            json={
                "agent_id": agent_id,
                "message": {
                    "role": "user",
                    "content": "runtime secret store create failure",
                },
                "connector_runtime_context": [
                    {
                        "connector_ref": {
                            "connector_type": "mcp",
                            "connector_id": server_id,
                        },
                        "secrets": {"authorization": "Bearer tenant-token"},
                    }
                ],
            },
        )

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["message"] == "Connector runtime setup failed."
    assert "tenant-token" not in resp.text
    assert "store failed" not in resp.text
    assert mock_start_task.call_count == 0

    db = _direct_db_session()
    try:
        task = (
            db.query(Task)
            .filter(Task.agent_id == agent_id)
            .filter(Task.input == "runtime secret store create failure")
            .one()
        )
        assert task.status == TaskStatus.FAILED
        assert task.error_message == "Connector runtime setup failed."
    finally:
        db.close()


def test_runtime_setup_failed_mark_swallows_secondary_rollback_failure(caplog):
    class RollbackFailingDB:
        rollback_calls = 0

        def rollback(self):
            self.rollback_calls += 1
            if self.rollback_calls > 1:
                raise RuntimeError("rollback failed")

        def query(self, _model):
            raise RuntimeError("query failed")

    db = RollbackFailingDB()

    with caplog.at_level("WARNING", logger="xagent.web.api.v1.tasks"):
        v1_tasks._mark_task_failed_after_runtime_setup_error(db, 123)

    assert db.rollback_calls == 2
    assert "Failed to roll back task 123 session" in caplog.text
    assert "Failed to mark task 123 failed" in caplog.text


def test_create_task_cleans_runtime_secret_when_schedule_fails(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(
        agent_id, required=False, secret_required=True
    )
    turn_ids: list[str] = []

    def recording_store(turn_id, values_by_ref):
        turn_ids.append(turn_id)
        store_ephemeral_runtime_values(turn_id, values_by_ref)

    with (
        patch(
            "xagent.web.api.v1.tasks.store_ephemeral_runtime_values",
            new=recording_store,
        ),
        patch(
            "xagent.web.services.task_orchestrator._schedule_bg",
            side_effect=RuntimeError("schedule failed for Bearer create-token"),
        ),
    ):
        resp = client.post(
            "/v1/chat/tasks",
            headers=_bearer(full_key),
            json={
                "agent_id": agent_id,
                "message": {
                    "role": "user",
                    "content": "runtime secret schedule failure",
                },
                "connector_runtime_context": [
                    {
                        "connector_ref": {
                            "connector_type": "mcp",
                            "connector_id": server_id,
                        },
                        "secrets": {"authorization": "Bearer create-token"},
                    }
                ],
            },
        )

    assert resp.status_code == 500
    assert turn_ids
    assert get_ephemeral_runtime_values(turn_ids[0]) is None
    assert "create-token" not in resp.text
    assert "schedule failed" not in resp.text
    assert mock_start_task.call_count == 0


def test_connector_runtime_resolver_can_override_values(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(agent_id)

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first user message"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"account_id": "6185"},
                }
            ],
        },
    )

    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]
    turn_id = mock_start_task.call_args.kwargs["payload"].turn_id
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()

        def _resolver(_request):
            return ConnectorRuntimeValues(
                context={"account_id": "hooked"},
                secrets={},
                auth_selector={},
            )

        set_connector_runtime_resolver_for_testing(_resolver)
        try:
            view = load_connector_runtime_view(
                db=db,
                task_id=task_id,
                turn_id=turn_id,
                user_id=int(task.user_id),
            )
        finally:
            set_connector_runtime_resolver_for_testing(None)
    finally:
        db.close()

    assert view[f"mcp:{server_id}"]["context"] == {"account_id": "hooked"}


@pytest.mark.asyncio
async def test_web_tool_config_applies_runtime_mcp_authorization_header(
    mock_start_task,
):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(
        agent_id,
        required=False,
        secret_required=True,
        delegated_authorization_binding=True,
    )

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first user message"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "secrets": {"authorization": "Bearer tenant-token"},
                }
            ],
        },
    )

    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]
    turn_id = mock_start_task.call_args.kwargs["payload"].turn_id
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        tool_config = WebToolConfig(
            db=db,
            request=None,
            user=SimpleNamespace(id=int(task.user_id), is_admin=False),
            user_id=int(task.user_id),
            is_admin=False,
            task_id=f"web_task_{task_id}",
            include_mcp_tools=True,
            tool_selection_spec=ToolSelectionSpec.from_raw(
                tool_categories=["mcp:ShiftCare"]
            ),
            connector_runtime_turn_id=turn_id,
        )

        configs = await tool_config.get_mcp_server_configs()
        refresh = configs[0]["config"].get("_connector_runtime_refresh")
        assert callable(refresh)

        def _resolver(_request):
            return ConnectorRuntimeValues(
                context={},
                secrets={"authorization": "Bearer refreshed-token"},
                auth_selector={},
            )

        set_connector_runtime_resolver_for_testing(_resolver)
        try:
            refreshed_connection = refresh()
        finally:
            set_connector_runtime_resolver_for_testing(None)
    finally:
        db.close()

    assert len(configs) == 1
    assert configs[0]["id"] == server_id
    assert configs[0]["config"]["headers"]["Authorization"] == "Bearer tenant-token"
    assert refreshed_connection["headers"]["Authorization"] == "Bearer refreshed-token"
    assert configs[0]["connector_runtime"]["secrets"] == {}


@pytest.mark.asyncio
async def test_web_tool_config_resolver_owned_remote_keeps_adapter_runtime_data(
    mock_start_task,
):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(agent_id, required=False)

    db = _direct_db_session()
    try:
        db.add(
            PublicMCPApp(
                app_id="shiftcare-runtime",
                name="ShiftCare",
                description="ShiftCare runtime app",
                transport="streamable_http",
                provider_name="shiftcare",
                launch_config={},
            )
        )
        db.commit()
    finally:
        db.close()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first user message"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"account_id": "tenant-account"},
                }
            ],
        },
    )

    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]
    turn_id = mock_start_task.call_args.kwargs["payload"].turn_id
    requests: list[TokenRequest] = []

    async def resolver(request: TokenRequest) -> ResolvedToken:
        requests.append(request)
        generation = "generation-1" if request.refresh is None else "generation-2"
        return ResolvedToken(access_token=generation, generation=generation)

    set_oauth_token_resolver_hook(resolver)
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        tool_config = WebToolConfig(
            db=db,
            request=None,
            user=SimpleNamespace(id=int(task.user_id), is_admin=False),
            user_id=int(task.user_id),
            is_admin=False,
            task_id=f"web_task_{task_id}",
            include_mcp_tools=True,
            tool_selection_spec=ToolSelectionSpec.from_raw(
                tool_categories=["mcp:ShiftCare"]
            ),
            connector_runtime_turn_id=turn_id,
        )
        configs = await tool_config.get_mcp_server_configs()
        refresh = configs[0]["config"]["_oauth_token_resolver_refresh"]
        refreshed = await refresh(
            MCPAuthorizationChallenge(
                resource_metadata_url=None,
                scope="shiftcare.read",
                params={},
            )
        )
    finally:
        set_oauth_token_resolver_hook(None)
        db.close()

    assert requests[0].provider == "shiftcare"
    assert configs[0]["connector_runtime"] == {
        "context": {"account_id": "tenant-account"},
        "secrets": {},
        "auth_selector": {},
    }
    assert configs[0]["runtime_bindings"][0]["target"]["target_type"] == "mcp_meta"
    assert refreshed["headers"]["Authorization"] == "Bearer generation-2"


def test_connector_runtime_secrets_do_not_enter_task_transcript(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(
        agent_id,
        required=False,
        secret_required=True,
        delegated_authorization_binding=True,
    )

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first user message"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "secrets": {"authorization": "Bearer transcript-token"},
                }
            ],
        },
    )

    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]
    from xagent.web.models.chat_message import TaskChatMessage

    db = _direct_db_session()
    try:
        msgs = (
            db.query(TaskChatMessage)
            .filter(TaskChatMessage.task_id == task_id)
            .order_by(TaskChatMessage.id)
            .all()
        )
    finally:
        db.close()

    assert [msg.content for msg in msgs] == ["first user message"]
    assert "transcript-token" not in repr([msg.content for msg in msgs])


def test_create_task_missing_authorization_returns_401(mock_start_task):
    """No Authorization header -> 401 invalid_api_key envelope."""
    agent_id, _key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "hi"},
        },
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "invalid_api_key"
    # No DB side effects
    assert mock_start_task.call_count == 0


def test_create_task_agent_id_mismatch_returns_404(mock_start_task):
    """body.agent_id != authed agent.id -> 404 agent_not_found."""
    _agent_id, full_key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": 999999,  # not the bound agent
            "message": {"role": "user", "content": "hi"},
        },
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "agent_not_found"
    assert mock_start_task.call_count == 0


def test_create_task_empty_message_returns_422(mock_start_task):
    """Empty message.content fails Pydantic min_length=1.

    The /v1/* path rewrites the FastAPI default
    ``{"detail": [...]}`` shape into the SDK envelope so clients can
    pin against ``body.error.code == 'invalid_input'`` for 422 just
    like they do for the other error codes.
    """
    agent_id, full_key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": ""},
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "invalid_input"
    assert "detail" not in body  # legacy FastAPI shape must not leak
    assert mock_start_task.call_count == 0


def test_create_task_wrong_role_returns_422(mock_start_task):
    """role != 'user' fails Pydantic Literal check -> 422 with the
    SDK envelope shape, not FastAPI's default ``{"detail": [...]}``."""
    agent_id, full_key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "assistant", "content": "hi"},
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "invalid_input"
    assert "detail" not in body
    assert mock_start_task.call_count == 0


def test_create_task_revoked_key_returns_401(mock_start_task):
    """Revoked key can't create tasks -> 401 invalid_api_key."""
    agent_id, full_key = _create_agent_with_key()
    # Revoke the key via the admin endpoint
    admin = _admin_headers()
    revoke = client.delete(f"/api/agents/{agent_id}/api-key", headers=admin)
    assert revoke.status_code == 200
    assert revoke.json()["revoked"] is True

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "hi"},
        },
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"
    assert mock_start_task.call_count == 0


def test_create_task_cross_user_agent_returns_404(mock_start_task):
    """Bob's key cannot target Alice's agent_id -> 404 agent_not_found.

    Defense: the key is bound to agent X. Putting agent_id=Y in the
    body where Y != X always returns 404 regardless of whether Y
    exists, owned by a different user, etc.
    """
    # Admin (alice) creates agent A and a key for it.
    alice_agent_id, _alice_key = _create_agent_with_key()

    # Register bob and create agent B + key, then have bob attempt to
    # POST against alice's agent_id using bob's own key.
    from ..conftest import _register_second_user

    bob_headers = _register_second_user()
    bob_agent = client.post(
        "/api/agents",
        headers=bob_headers,
        json={
            "name": "bob agent",
            "description": "test",
            "instructions": "test",
            "execution_mode": "balanced",
        },
    ).json()
    bob_agent_id = bob_agent["id"]
    bob_key = client.post(
        f"/api/agents/{bob_agent_id}/api-key", headers=bob_headers
    ).json()["full_key"]

    # Bob's key + Alice's agent_id in body -> 404
    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(bob_key),
        json={
            "agent_id": alice_agent_id,
            "message": {"role": "user", "content": "hi"},
        },
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent_not_found"
    assert mock_start_task.call_count == 0


# ===== Shared helper for E tests: create a task via POST then return its id =====


def _create_task(full_key: str, agent_id: int, content: str = "hello") -> int:
    """Drive POST /v1/chat/tasks and return the resulting task_id."""
    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": content}},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["task_id"]


# ===== POST /v1/chat/tasks/{task_id}/messages =====


def _force_task_status(task_id: int, status: TaskStatus) -> None:
    """Bypass the bg coroutine and flip a task to a desired status.

    Tests in this file mock out the bg scheduling so a freshly-created
    task stays at PENDING forever. The orchestrator's ``append_turn``
    only accepts terminal statuses (COMPLETED / FAILED) -- which is
    correct production behavior -- so tests that exercise the
    append-happy path need to push the task to COMPLETED first.
    """
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = status
        db.commit()
    finally:
        db.close()


def test_append_message_happy_path(mock_start_task):
    """Returns 202 + accepted_at, persists new user message, kicks off bg."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, content="first turn")
    # Mark the previous turn as COMPLETED so append_turn's atomic claim
    # passes (PENDING is rejected as busy because it means the create's
    # bg run hasn't finished yet).
    _force_task_status(task_id, TaskStatus.COMPLETED)
    mock_start_task.reset_mock()  # discard the create-task call so we count just the append

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "second turn"},
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["agent_id"] == agent_id
    # The atomic claim in the endpoint already flipped the row to
    # RUNNING; the response must mirror that so back-to-back POST+GET
    # don't show contradictory statuses to SDK clients.
    assert body["status"] == "running"
    assert "accepted_at" in body

    # Two user messages now exist for this task; task.input is the latest
    from xagent.web.models.chat_message import TaskChatMessage

    db = _direct_db_session()
    try:
        msgs = (
            db.query(TaskChatMessage)
            .filter(TaskChatMessage.task_id == task_id)
            .order_by(TaskChatMessage.id)
            .all()
        )
        assert len(msgs) == 2
        assert [m.content for m in msgs] == ["first turn", "second turn"]
        task = db.query(Task).filter(Task.id == task_id).first()
        assert task is not None
        assert task.input == "second turn"
    finally:
        db.close()

    assert mock_start_task.call_count == 1


def test_append_message_rejects_changed_connector_runtime_context(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(agent_id)
    create_resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first turn"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"account_id": "6185"},
                }
            ],
        },
    )
    assert create_resp.status_code == 202, create_resp.text
    task_id = create_resp.json()["task_id"]
    _force_task_status(task_id, TaskStatus.COMPLETED)
    mock_start_task.reset_mock()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "second turn"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"account_id": "9999"},
                }
            ],
        },
    )

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "runtime_context_immutable"
    assert mock_start_task.call_count == 0


def test_append_message_accepts_same_connector_runtime_context(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(agent_id)
    create_resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first turn"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"account_id": "6185"},
                }
            ],
        },
    )
    assert create_resp.status_code == 202, create_resp.text
    task_id = create_resp.json()["task_id"]
    _force_task_status(task_id, TaskStatus.COMPLETED)
    mock_start_task.reset_mock()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "second turn"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"account_id": "6185"},
                }
            ],
        },
    )

    assert resp.status_code == 202, resp.text
    assert mock_start_task.call_count == 1


def test_append_message_ignores_disabled_historical_connector_not_in_payload(
    mock_start_task,
):
    agent_id, full_key = _create_agent_with_key()
    stale_server_id = _install_runtime_mcp_connector(
        agent_id, name="StaleCare", required=False, selected=False
    )
    active_server_id = _install_runtime_mcp_connector(
        agent_id,
        name="ActiveCare",
        required=False,
        secret_required=True,
        selected=False,
    )

    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        agent.tool_categories = ["mcp"]
        db.commit()
    finally:
        db.close()

    create_resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first turn"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": active_server_id,
                    },
                    "secrets": {"authorization": "Bearer initial-token"},
                }
            ],
        },
    )
    assert create_resp.status_code == 202, create_resp.text
    task_id = create_resp.json()["task_id"]
    _force_task_status(task_id, TaskStatus.COMPLETED)
    mock_start_task.reset_mock()

    db = _direct_db_session()
    try:
        junction = (
            db.query(UserMCPServer)
            .filter(UserMCPServer.mcpserver_id == stale_server_id)
            .one()
        )
        junction.is_active = False
        db.commit()
    finally:
        db.close()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "second turn"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": active_server_id,
                    },
                    "secrets": {"authorization": "Bearer append-token"},
                }
            ],
        },
    )

    assert resp.status_code == 202, resp.text
    assert mock_start_task.call_count == 1


def test_append_message_rejects_disabled_historical_connector_in_payload(
    mock_start_task,
):
    agent_id, full_key = _create_agent_with_key()
    stale_server_id = _install_runtime_mcp_connector(
        agent_id, name="StaleCare", required=False, selected=False
    )
    active_server_id = _install_runtime_mcp_connector(
        agent_id,
        name="ActiveCare",
        required=False,
        secret_required=True,
        selected=False,
    )

    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        agent.tool_categories = ["mcp"]
        db.commit()
    finally:
        db.close()

    create_resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first turn"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": active_server_id,
                    },
                    "secrets": {"authorization": "Bearer initial-token"},
                }
            ],
        },
    )
    assert create_resp.status_code == 202, create_resp.text
    task_id = create_resp.json()["task_id"]
    _force_task_status(task_id, TaskStatus.COMPLETED)
    mock_start_task.reset_mock()

    db = _direct_db_session()
    try:
        junction = (
            db.query(UserMCPServer)
            .filter(UserMCPServer.mcpserver_id == stale_server_id)
            .one()
        )
        junction.is_active = False
        db.commit()
    finally:
        db.close()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "second turn"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": stale_server_id,
                    },
                    "context": {"account_id": "6185"},
                }
            ],
        },
    )

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "connector_not_found"
    assert mock_start_task.call_count == 0


def test_append_message_keeps_task_state_when_runtime_secret_store_fails(
    mock_start_task,
):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(
        agent_id, required=False, secret_required=True
    )
    create_resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first turn"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "secrets": {"authorization": "Bearer initial-token"},
                }
            ],
        },
    )
    assert create_resp.status_code == 202, create_resp.text
    pop_ephemeral_runtime_values(mock_start_task.call_args.kwargs["payload"].turn_id)
    task_id = create_resp.json()["task_id"]
    _force_task_status(task_id, TaskStatus.COMPLETED)
    mock_start_task.reset_mock()

    with patch(
        "xagent.web.api.v1.tasks.store_ephemeral_runtime_values",
        side_effect=RuntimeError("store failed for Bearer append-token"),
    ):
        resp = client.post(
            f"/v1/chat/tasks/{task_id}/messages",
            headers=_bearer(full_key),
            json={
                "agent_id": agent_id,
                "message": {"role": "user", "content": "second turn"},
                "connector_runtime_context": [
                    {
                        "connector_ref": {
                            "connector_type": "mcp",
                            "connector_id": server_id,
                        },
                        "secrets": {"authorization": "Bearer append-token"},
                    }
                ],
            },
        )

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["message"] == "Connector runtime setup failed."
    assert "append-token" not in resp.text
    assert "store failed" not in resp.text
    assert mock_start_task.call_count == 0

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.status == TaskStatus.COMPLETED
        assert task.input == "first turn"
        assert task.error_message is None
    finally:
        db.close()


def test_append_message_cleans_runtime_secret_when_task_is_busy(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    server_id = _install_runtime_mcp_connector(
        agent_id, required=False, secret_required=True
    )
    create_resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first turn"},
            "connector_runtime_context": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "secrets": {"authorization": "Bearer initial-token"},
                }
            ],
        },
    )
    assert create_resp.status_code == 202, create_resp.text
    pop_ephemeral_runtime_values(mock_start_task.call_args.kwargs["payload"].turn_id)
    task_id = create_resp.json()["task_id"]
    mock_start_task.reset_mock()

    turn_ids: list[str] = []

    def recording_store(turn_id, values_by_ref):
        turn_ids.append(turn_id)
        store_ephemeral_runtime_values(turn_id, values_by_ref)

    with patch(
        "xagent.web.api.v1.tasks.store_ephemeral_runtime_values",
        new=recording_store,
    ):
        resp = client.post(
            f"/v1/chat/tasks/{task_id}/messages",
            headers=_bearer(full_key),
            json={
                "agent_id": agent_id,
                "message": {"role": "user", "content": "second turn"},
                "connector_runtime_context": [
                    {
                        "connector_ref": {
                            "connector_type": "mcp",
                            "connector_id": server_id,
                        },
                        "secrets": {"authorization": "Bearer append-token"},
                    }
                ],
            },
        )

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "task_busy"
    assert turn_ids
    assert get_ephemeral_runtime_values(turn_ids[0]) is None
    assert "append-token" not in resp.text
    assert mock_start_task.call_count == 0


def test_append_message_to_running_task_returns_409(mock_start_task):
    """Appending to a RUNNING task is rejected as task_busy."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    mock_start_task.reset_mock()

    # Flip status to RUNNING directly so we don't have to actually run
    # the agent service in tests.
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = TaskStatus.RUNNING
        db.commit()
    finally:
        db.close()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "hello"}},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "task_busy"
    # No new background kickoff happened
    assert mock_start_task.call_count == 0


def test_append_message_claims_slot_atomically(mock_start_task):
    """Successful append flips task.status to RUNNING in the same
    transaction as the input write, so a concurrent POST can't pass
    the busy check and both kick off background tasks.

    We verify the post-state directly: after one successful append,
    task.status == RUNNING and a second POST to the same task gets
    409 even though the bg coroutine hasn't run yet (mocked out).
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, content="t1")
    # Push to COMPLETED so the first append is allowed (PENDING is busy).
    _force_task_status(task_id, TaskStatus.COMPLETED)
    mock_start_task.reset_mock()

    # First append succeeds and atomically claims the slot.
    r1 = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "t2"}},
    )
    assert r1.status_code == 202

    # task.status was flipped to RUNNING inside the endpoint, even
    # though the (mocked) bg coroutine never ran. This is the
    # mechanism that defeats the TOCTOU race.
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        assert task is not None
        assert task.status == TaskStatus.RUNNING
    finally:
        db.close()

    # Second append (the would-be losing concurrent request) hits the
    # claim filter and 409s.
    r2 = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "t3"}},
    )
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "task_busy"
    # Only one bg kickoff total (from the winning first append).
    assert mock_start_task.call_count == 1


def test_create_then_append_race_returns_409(mock_start_task):
    """Regression: create-then-immediate-append must 409, not race.

    The old append_turn used ``status != RUNNING`` which let PENDING
    slip through. A client could POST /v1/chat/tasks, get a PENDING
    task back, and immediately POST /messages; both the create's bg
    coroutine and the append's bg coroutine would then race to run
    the same task. The orchestrator's terminal-state-only filter
    closes this: PENDING (= "first turn scheduled but not started")
    is treated as busy.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, content="first")
    # NOT calling _force_task_status here — task stays in PENDING
    # exactly as it would right after the SDK's create response is
    # returned but before the bg coroutine ran.
    mock_start_task.reset_mock()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "second"},
        },
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "task_busy"
    # No second bg kickoff should have happened
    assert mock_start_task.call_count == 0


def test_append_message_bg_inflight_does_not_corrupt_task_state(mock_start_task):
    """Regression: when ``append_turn`` refuses because a previous bg
    coroutine is still in flight, the DB row must NOT have been mutated.

    The bug scenario: previous turn flipped status to COMPLETED but the
    bg coroutine is still in tail cleanup (``_sync_sdk_columns`` hasn't
    returned), so ``background_task_manager.running_tasks[task_id]`` is
    still a not-done asyncio.Task. A new append should be refused as
    busy, and the DB row should still report COMPLETED + the original
    input — not RUNNING + new input.

    If the inflight check happened *after* the atomic UPDATE (the old
    ordering), the row would be RUNNING + new_input even on 409
    rejection. The old runner's _sync_sdk_columns would then see
    RUNNING and flip the row to FAILED with a placeholder error
    message, corrupting an otherwise successful past turn.
    """
    import asyncio

    from xagent.web.api.websocket import background_task_manager

    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, content="first turn")
    _force_task_status(task_id, TaskStatus.COMPLETED)
    mock_start_task.reset_mock()

    # Snapshot DB state right before the append attempt.
    db_before = _direct_db_session()
    try:
        task_before = db_before.query(Task).filter(Task.id == task_id).first()
        original_status = task_before.status
        original_input = task_before.input
    finally:
        db_before.close()

    # Plant a not-done asyncio.Task in the bg manager registry to
    # simulate "previous runner is still cleaning up".
    loop = asyncio.new_event_loop()
    try:

        async def _never_done() -> None:
            await asyncio.sleep(3600)

        fake_inflight = loop.create_task(_never_done())
        background_task_manager.running_tasks[task_id] = fake_inflight

        try:
            resp = client.post(
                f"/v1/chat/tasks/{task_id}/messages",
                headers=_bearer(full_key),
                json={
                    "agent_id": agent_id,
                    "message": {"role": "user", "content": "second"},
                },
            )
            assert resp.status_code == 409
            assert resp.json()["error"]["code"] == "task_busy"
            assert mock_start_task.call_count == 0

            # The critical assertion: DB row was NOT mutated by the
            # refused append. status stays terminal, input unchanged.
            db_after = _direct_db_session()
            try:
                task_after = db_after.query(Task).filter(Task.id == task_id).first()
                assert task_after.status == original_status, (
                    f"task.status was corrupted on refused append: "
                    f"{original_status} -> {task_after.status}"
                )
                assert task_after.input == original_input, (
                    f"task.input was overwritten on refused append: "
                    f"{original_input!r} -> {task_after.input!r}"
                )
            finally:
                db_after.close()
        finally:
            # Clean up the fake registry entry so other tests aren't
            # affected.
            background_task_manager.running_tasks.pop(task_id, None)
            fake_inflight.cancel()
    finally:
        loop.close()


def test_append_message_to_missing_task_returns_404(mock_start_task):
    """Appending to a task that doesn't exist -> 404 task_not_found."""
    agent_id, full_key = _create_agent_with_key()
    resp = client.post(
        "/v1/chat/tasks/9999999/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "hi"}},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"
    assert mock_start_task.call_count == 0


def test_append_message_to_other_agents_task_returns_404(mock_start_task):
    """Bob can't append to Alice's task even if he knows the id."""
    alice_agent_id, alice_key = _create_agent_with_key()
    alice_task_id = _create_task(alice_key, alice_agent_id)
    mock_start_task.reset_mock()

    from ..conftest import _register_second_user

    bob_headers = _register_second_user()
    bob_agent = client.post(
        "/api/agents",
        headers=bob_headers,
        json={
            "name": "bob agent",
            "description": "test",
            "instructions": "test",
            "execution_mode": "balanced",
        },
    ).json()
    bob_agent_id = bob_agent["id"]
    bob_key = client.post(
        f"/api/agents/{bob_agent_id}/api-key", headers=bob_headers
    ).json()["full_key"]

    resp = client.post(
        f"/v1/chat/tasks/{alice_task_id}/messages",
        headers=_bearer(bob_key),
        json={"agent_id": bob_agent_id, "message": {"role": "user", "content": "hi"}},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"
    assert mock_start_task.call_count == 0


def test_append_message_body_agent_id_mismatch_returns_404(mock_start_task):
    """body.agent_id != authed agent.id -> 404 agent_not_found.

    Distinct from cross-agent task ownership (which is task_not_found) --
    here the task IS the caller's, but the body claims a different
    agent_id; consistent with POST /v1/chat/tasks behavior.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    mock_start_task.reset_mock()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": 999999, "message": {"role": "user", "content": "hi"}},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent_not_found"
    assert mock_start_task.call_count == 0


# ===== GET /v1/chat/tasks/{task_id} =====


def test_get_task_running_right_after_create(mock_start_task):
    """A fresh SDK task is visible as ``status='running'`` immediately
    after POST returns: the atomic claim commits the status flip before
    202, so an immediate GET sees RUNNING + input set + output/error
    null + completed_at null.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, content="seed input")

    resp = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["agent_id"] == agent_id
    assert body["status"] == "running"
    assert body["input"] == "seed input"
    assert body["output"] is None
    assert body["error"] is None
    assert "created_at" in body
    assert body["completed_at"] is None


def test_get_task_completed_returns_output(mock_start_task):
    """Completed task: output populated, completed_at set."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    # Flip status + write output to simulate completed background turn
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = TaskStatus.COMPLETED
        task.output = "final answer"
        db.commit()
    finally:
        db.close()

    resp = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["output"] == "final answer"
    assert body["completed_at"] is not None


def test_get_task_failed_returns_error(mock_start_task):
    """Failed task: error populated, completed_at set."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = TaskStatus.FAILED
        task.error_message = "agent crashed"
        db.commit()
    finally:
        db.close()

    resp = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["error"] == "agent crashed"
    assert body["output"] is None
    assert body["completed_at"] is not None


def test_get_missing_task_returns_404(mock_start_task):
    _agent_id, full_key = _create_agent_with_key()
    resp = client.get("/v1/chat/tasks/9999999", headers=_bearer(full_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


def test_get_other_agents_task_returns_404(mock_start_task):
    """Cross-agent task access -> 404 (not leaking existence)."""
    alice_agent_id, alice_key = _create_agent_with_key()
    alice_task_id = _create_task(alice_key, alice_agent_id)

    from ..conftest import _register_second_user

    bob_headers = _register_second_user()
    bob_agent = client.post(
        "/api/agents",
        headers=bob_headers,
        json={
            "name": "bob agent",
            "description": "test",
            "instructions": "test",
            "execution_mode": "balanced",
        },
    ).json()
    bob_agent_id = bob_agent["id"]
    bob_key = client.post(
        f"/api/agents/{bob_agent_id}/api-key", headers=bob_headers
    ).json()["full_key"]

    resp = client.get(f"/v1/chat/tasks/{alice_task_id}", headers=_bearer(bob_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


# ===== F: GET /v1/chat/tasks/{task_id}/steps =====


def _insert_trace_event(
    *,
    task_id: int,
    event_type: str,
    event_id: str,
    timestamp: datetime,
    data: dict,
    step_id: str | None = None,
    build_id: str | None = None,
) -> None:
    """Insert one TraceEvent row directly via the test DB.

    Bypasses the production trace handler (which runs through asyncio
    + thread pool) so tests can assert on the GET /steps surface
    without spinning up the agent runtime.
    """
    db = _direct_db_session()
    try:
        ev = TraceEvent(
            task_id=task_id,
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            step_id=step_id,
            build_id=build_id,
            data=data,
        )
        db.add(ev)
        db.commit()
    finally:
        db.close()


def test_get_steps_returns_mapped_steps_in_order(mock_start_task):
    """Insert react_action + tool_execution + ai_message + filtered
    llm_call events, GET /steps, assert 3 public steps in started_at
    order with correct types and statuses.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Public step 1: thinking phase=action (react_action_start/end)
    _insert_trace_event(
        task_id=task_id,
        event_type="react_action_start",
        event_id="evt-1",
        timestamp=base.replace(second=1),
        step_id="step-A",
        data={},
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="react_action_end",
        event_id="evt-2",
        timestamp=base.replace(second=2),
        step_id="step-A",
        data={},
    )

    # Filtered: llm_call_start / end -- must not appear
    _insert_trace_event(
        task_id=task_id,
        event_type="llm_call_start",
        event_id="evt-3",
        timestamp=base.replace(second=3),
        step_id="step-A",
        data={},
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="llm_call_end",
        event_id="evt-4",
        timestamp=base.replace(second=4),
        step_id="step-A",
        data={},
    )

    # Public step 2: tool_call (execute_python)
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_start",
        event_id="evt-5",
        timestamp=base.replace(second=5),
        step_id="step-A",
        data={
            "tool_name": "execute_python",
            "tool_args": {"code": "print(1)"},
            "tool_execution_id": "tx-1",
        },
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_end",
        event_id="evt-6",
        timestamp=base.replace(second=6),
        step_id="step-A",
        data={
            "tool_name": "execute_python",
            "tool_args": {"code": "print(1)"},
            "tool_execution_id": "tx-1",
            "result": {"output": "1\n"},
            "success": True,
        },
    )

    # Public step 3: message role=assistant (ai_message)
    _insert_trace_event(
        task_id=task_id,
        event_type="ai_message",
        event_id="evt-7",
        timestamp=base.replace(second=7),
        data={"content": "Here's the result"},
    )

    resp = client.get(f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["agent_id"] == agent_id
    steps = body["steps"]
    assert len(steps) == 3

    # Assert ordering and types
    assert steps[0]["type"] == "thinking"
    assert steps[0]["data"]["phase"] == "action"
    assert steps[0]["status"] == "completed"

    assert steps[1]["type"] == "tool_call"
    assert steps[1]["data"]["name"] == "execute_python"
    assert steps[1]["data"]["args"] == {"code": "print(1)"}
    assert steps[1]["data"]["result"] == {"output": "1\n"}

    assert steps[2]["type"] == "message"
    assert steps[2]["data"] == {"role": "assistant", "content": "Here's the result"}


def test_get_steps_task_not_found_returns_404(mock_start_task):
    """Non-existent task_id -> 404 task_not_found."""
    _agent_id, full_key = _create_agent_with_key()
    resp = client.get("/v1/chat/tasks/9999999/steps", headers=_bearer(full_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


def test_get_steps_other_agents_task_returns_404(mock_start_task):
    """Cross-agent steps access -> 404 (not leaking existence)."""
    alice_agent_id, alice_key = _create_agent_with_key()
    alice_task_id = _create_task(alice_key, alice_agent_id)

    from ..conftest import _register_second_user

    bob_headers = _register_second_user()
    bob_agent = client.post(
        "/api/agents",
        headers=bob_headers,
        json={
            "name": "bob agent steps",
            "description": "test",
            "instructions": "test",
            "execution_mode": "balanced",
        },
    ).json()
    bob_agent_id = bob_agent["id"]
    bob_key = client.post(
        f"/api/agents/{bob_agent_id}/api-key", headers=bob_headers
    ).json()["full_key"]

    resp = client.get(f"/v1/chat/tasks/{alice_task_id}/steps", headers=_bearer(bob_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


def test_get_steps_empty_task_returns_empty_array(mock_start_task):
    """Task with no trace events yet -> 200 + empty steps array."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    resp = client.get(f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key))
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["agent_id"] == agent_id
    assert body["steps"] == []


def test_get_steps_ignores_worker_build_trace_events(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_start",
        event_id="worker-trace-1",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        step_id="worker-step",
        build_id="agent_123_abcd1234",
        data={
            "tool_name": "worker_tool",
            "tool_execution_id": "worker-call-1",
            "worker_task_id": "agent_123_abcd1234",
        },
    )

    resp = client.get(f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key))

    assert resp.status_code == 200, resp.text
    assert resp.json()["steps"] == []


def test_get_steps_cache_reuses_mapping_until_trace_event_changes(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    _insert_trace_event(
        task_id=task_id,
        event_type="ai_message",
        event_id="evt-cache-1",
        timestamp=base,
        data={"content": "cached"},
    )

    set_cache_backend_for_testing(InMemoryTTLCache())
    try:
        from xagent.web.api.v1 import _step_mapping

        with patch(
            "xagent.web.api.v1.tasks.map_trace_events_to_public_steps",
            wraps=_step_mapping.map_trace_events_to_public_steps,
        ) as mapper:
            first = client.get(
                f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
            )
            second = client.get(
                f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
            )

            assert first.status_code == 200, first.text
            assert second.status_code == 200, second.text
            assert first.json() == second.json()
            assert mapper.call_count == 1

            _insert_trace_event(
                task_id=task_id,
                event_type="ai_message",
                event_id="evt-cache-2",
                timestamp=base.replace(second=1),
                data={"content": "new"},
            )
            third = client.get(
                f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
            )

            assert third.status_code == 200, third.text
            assert mapper.call_count == 2
            assert len(third.json()["steps"]) == 2
    finally:
        set_cache_backend_for_testing(None)


def test_get_steps_redacts_runtime_secrets_before_cache_write(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_start",
        event_id="evt-secret-start",
        timestamp=base,
        step_id="secret-step",
        data={
            "tool_name": "custom_api_call",
            "tool_execution_id": "secret-call",
            "tool_args": {
                "headers": {
                    "Authorization": "Bearer runtime-token",
                    "X-Account": "6185",
                },
                "connector_runtime": {
                    "secrets": {"authorization": "Bearer nested-token"},
                    "auth_selector": {"resource_owner_key": "xagent:user:owner"},
                },
            },
        },
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_end",
        event_id="evt-secret-end",
        timestamp=base.replace(second=1),
        step_id="secret-step",
        data={
            "tool_name": "custom_api_call",
            "tool_execution_id": "secret-call",
            "result": {
                "headers": {"Authorization": "Bearer echoed-token"},
                "safe": "ok",
            },
            "success": True,
        },
    )

    set_cache_backend_for_testing(InMemoryTTLCache())
    try:
        from xagent.web.api.v1 import _step_mapping

        with patch(
            "xagent.web.api.v1.tasks.map_trace_events_to_public_steps",
            wraps=_step_mapping.map_trace_events_to_public_steps,
        ) as mapper:
            first = client.get(
                f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
            )
            cached = cache_get(task_steps_key(task_id))
            second = client.get(
                f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
            )
    finally:
        set_cache_backend_for_testing(None)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json() == second.json()
    assert mapper.call_count == 1
    assert isinstance(cached, dict)
    assert "runtime-token" not in repr(cached)
    assert "nested-token" not in repr(cached)
    assert "xagent:user:owner" not in repr(cached)
    assert "echoed-token" not in repr(cached)
    response_text = first.text + second.text
    assert "runtime-token" not in response_text
    assert "nested-token" not in response_text
    assert "xagent:user:owner" not in response_text
    assert "echoed-token" not in response_text
    step = first.json()["steps"][0]
    assert (
        step["data"]["args"]["headers"]["Authorization"] == "[REDACTED_RUNTIME_SECRET]"
    )
    assert step["data"]["args"]["headers"]["X-Account"] == "6185"
    assert step["data"]["result"]["headers"]["Authorization"] == (
        "[REDACTED_RUNTIME_SECRET]"
    )


# ===== source filtering: SDK API surface only sees source="sdk" tasks =====


def _insert_internal_task(agent_id: int) -> int:
    """Manually INSERT a task under ``agent_id`` with source != "sdk".

    Tests for the SDK source filter need a task that lives under the
    same agent but was created by the Web UI / internal paths, not
    via POST /v1/chat/tasks. Since the SDK create endpoint always
    writes source="sdk", we bypass it and craft the row directly.
    """
    from xagent.web.models.agent import Agent
    from xagent.web.models.task import Task, TaskStatus
    from xagent.web.models.user import User

    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        assert agent is not None
        # Reuse the agent's owner so user_id stays consistent.
        user = db.query(User).filter(User.id == agent.user_id).first()
        assert user is not None
        task = Task(
            user_id=user.id,
            title="internal task",
            description="created via web ui, not sdk",
            status=TaskStatus.COMPLETED,
            agent_id=agent.id,
            input="internal user message",
            source="internal",
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return int(task.id)
    finally:
        db.close()


def test_get_task_returns_404_for_non_sdk_source(mock_start_task):
    """A task created by the Web UI / internal path (source != "sdk")
    under the same agent must NOT be readable through GET /v1/chat/tasks/{id}.

    Without the source filter, an SDK API key could enumerate / read
    the user's own Web UI conversations whenever they happen to live
    under the same agent.
    """
    agent_id, full_key = _create_agent_with_key()
    internal_task_id = _insert_internal_task(agent_id)
    mock_start_task.reset_mock()

    resp = client.get(f"/v1/chat/tasks/{internal_task_id}", headers=_bearer(full_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


def test_append_message_returns_404_for_non_sdk_source(mock_start_task):
    """POST /v1/chat/tasks/{id}/messages on a non-SDK task must 404
    with task_not_found — the SDK key shouldn't be able to mutate
    Web UI conversations even if it knows the task id."""
    agent_id, full_key = _create_agent_with_key()
    internal_task_id = _insert_internal_task(agent_id)
    mock_start_task.reset_mock()

    resp = client.post(
        f"/v1/chat/tasks/{internal_task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "hi"}},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"
    assert mock_start_task.call_count == 0


def test_get_steps_returns_404_for_non_sdk_source(mock_start_task):
    """GET /v1/chat/tasks/{id}/steps on a non-SDK task must 404 so
    the SDK can't enumerate Web UI step traces under the same agent."""
    agent_id, full_key = _create_agent_with_key()
    internal_task_id = _insert_internal_task(agent_id)

    resp = client.get(
        f"/v1/chat/tasks/{internal_task_id}/steps", headers=_bearer(full_key)
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


# ===== Latest-turn snapshot invariant on SDK append =====


def test_append_message_clears_stale_output_for_sdk_caller(mock_start_task):
    """An SDK append on a previously completed task immediately clears
    the stored ``output`` and ``error_message`` so a GET right after
    the append sees a clean latest-turn snapshot. Without this clearing,
    the response would mix the new turn's status / input with the
    previous turn's output — an internally contradictory snapshot for
    SDK consumers polling the task.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, content="first")

    # Plant a completed-state row with prior output / error_message
    # populated, as if the first turn had finished successfully.
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = TaskStatus.COMPLETED
        task.output = "first answer"
        task.error_message = "stale error from prior failure"
        db.commit()
    finally:
        db.close()

    mock_start_task.reset_mock()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "second"}},
    )
    assert resp.status_code == 202, resp.text
    assert mock_start_task.call_count == 1

    # After the response returns, an immediate GET must see:
    #   - status = running (atomic transition committed)
    #   - input = the new turn's message
    #   - output = NULL (stale prior-turn output cleared)
    #   - error_message = NULL (stale prior error cleared)
    db = _direct_db_session()
    try:
        task_after = db.query(Task).filter(Task.id == task_id).first()
        assert task_after.status == TaskStatus.RUNNING
        assert task_after.input == "second"
        assert task_after.output is None
        assert task_after.error_message is None
    finally:
        db.close()
