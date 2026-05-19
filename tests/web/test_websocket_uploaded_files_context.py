from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.agent.trace import get_display_user_message
from xagent.web.api import websocket as websocket_api
from xagent.web.api.chat import _build_task_agent_config
from xagent.web.api.websocket import (
    _append_uploaded_files_context_to_message,
    _build_uploaded_files_context,
    _display_file_refs_from_file_info,
    _display_message_for_user,
    _normalize_file_outputs,
    _register_uploaded_files_for_agent,
    _selected_file_refs_from_task,
    execute_task_background,
    handle_file_upload_for_task,
)
from xagent.web.models import Base
from xagent.web.models import database as database_models
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def _create_user(db, user_id: int, username: str) -> User:
    user = User(id=user_id, username=username, password_hash="hash")
    db.add(user)
    db.flush()
    return user


def _create_task(
    db,
    *,
    task_id: int,
    user_id: int,
    selected_file_ids: list[str] | None = None,
    status: TaskStatus = TaskStatus.PENDING,
) -> Task:
    task = Task(
        id=task_id,
        user_id=user_id,
        title=f"task-{task_id}",
        description="task",
        status=status,
        agent_config=(
            {"selected_file_ids": selected_file_ids}
            if selected_file_ids is not None
            else None
        ),
    )
    db.add(task)
    db.flush()
    return task


def _create_uploaded_file(
    db,
    tmp_path,
    *,
    file_id: str,
    user_id: int,
    task_id: int | None,
    filename: str,
) -> UploadedFile:
    path = tmp_path / f"{file_id}-{filename}"
    path.write_text("file content")
    file_record = UploadedFile(
        file_id=file_id,
        user_id=user_id,
        task_id=task_id,
        filename=filename,
        storage_path=str(path),
        mime_type="text/plain",
        file_size=len("file content"),
    )
    db.add(file_record)
    db.flush()
    return file_record


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_status",
    [TaskStatus.COMPLETED, TaskStatus.FAILED],
)
async def test_execute_task_background_reuses_task_id_for_terminal_tasks(
    db_session,
    monkeypatch,
    terminal_status,
):
    user = _create_user(db_session, 1, "owner")
    _create_task(
        db_session,
        task_id=10,
        user_id=1,
        status=terminal_status,
    )
    captured: dict[str, object] = {}

    class BackgroundTaskManager:
        async def wait_for_previous(self, task_id):
            captured["waited_for"] = task_id

        def cleanup_task(self, task_id):
            captured["cleaned_up"] = task_id

    class BroadcastManager:
        async def broadcast_to_task(self, event, task_id):
            captured["broadcast_task_id"] = task_id

    class AgentService:
        def set_conversation_history(self, history):
            captured["conversation_history"] = history

        def set_execution_context_messages(self, messages):
            captured["execution_context_messages"] = messages

        def set_recovered_skill_context(self, skill_context):
            captured["skill_context"] = skill_context

    class AgentManager:
        async def get_agent_for_task(self, task_id, db, user=None):
            captured["agent_db"] = db
            return AgentService()

        async def execute_task(
            self,
            *,
            agent_service,
            task,
            context,
            task_id,
            tracking_task_id,
            db_session,
        ):
            captured["agent_task"] = task
            captured["agent_task_id"] = task_id
            captured["tracking_task_id"] = tracking_task_id
            return {"success": True, "output": "ok", "file_outputs": []}

    def fake_get_db():
        yield db_session

    def fake_release_current_runner_task_lease(db, task_id, *, status):
        return True

    monkeypatch.setattr(
        websocket_api,
        "background_task_manager",
        BackgroundTaskManager(),
    )
    monkeypatch.setattr(websocket_api, "manager", BroadcastManager())
    monkeypatch.setattr(
        websocket_api,
        "release_current_runner_task_lease",
        fake_release_current_runner_task_lease,
    )
    monkeypatch.setattr(database_models, "get_db", fake_get_db)
    monkeypatch.setattr(
        "xagent.web.services.chat_history_service.persist_assistant_message",
        lambda *args, **kwargs: None,
    )

    await execute_task_background(
        task_id=10,
        user_message="重试",
        context={},
        agent_manager=AgentManager(),
        user_id=int(user.id),
        llm_user_message="重试",
    )

    assert captured["agent_task_id"] == "10"
    assert captured["tracking_task_id"] == "10"
    assert captured["agent_task"] == "重试"


def test_build_uploaded_files_context_includes_agent_builder_kb_instruction():
    context = _build_uploaded_files_context(
        [
            {
                "file_id": "file-123",
                "name": "faq.docx",
                "original_name": "FAQ.docx",
            }
        ],
        is_agent_builder=True,
    )

    assert "FAQ.docx: file_id=file-123" in context
    assert "## FILE REFERENCES" in context
    assert "Treat file_id as the canonical file handle" in context
    assert "call prepare_html_asset(file_id, html_path, alias) first" in context
    assert "create_knowledge_base_from_file" in context
    assert 'file_ids = ["file-123"]' in context
    assert "Do NOT ask the user to upload again" in context


def test_append_uploaded_files_context_to_message_is_idempotent():
    context = _build_uploaded_files_context(
        [{"file_id": "file-123", "name": "faq.docx"}],
        is_agent_builder=False,
    )

    message = _append_uploaded_files_context_to_message("Upload File", context)
    assert message.startswith("Upload File\n\n## UPLOADED FILES")
    assert "Do not guess storage paths" in message
    assert "Use the returned html_src inside HTML" in message
    assert _append_uploaded_files_context_to_message(message, context) == message


def test_build_task_agent_config_ignores_client_selected_file_ids():
    assert _build_task_agent_config(
        {"selected_file_ids": ["forged"], "tools": ["search"]},
        [],
    ) == {"tools": ["search"]}
    assert _build_task_agent_config({"selected_file_ids": ["forged"]}, []) is None
    assert _build_task_agent_config(
        {"selected_file_ids": ["forged"], "tools": ["search"]},
        ["valid-file"],
    ) == {"tools": ["search"], "selected_file_ids": ["valid-file"]}


def test_create_task_file_selection_requires_unbound_files(db_session, tmp_path):
    _create_user(db_session, 1, "owner")
    _create_task(db_session, task_id=10, user_id=1)
    bound_file = _create_uploaded_file(
        db_session,
        tmp_path,
        file_id="bound-file",
        user_id=1,
        task_id=10,
        filename="bound.txt",
    )

    selected_file_ids = []
    uploaded_file = (
        db_session.query(UploadedFile)
        .filter(
            UploadedFile.file_id == "bound-file",
            UploadedFile.user_id == 1,
            UploadedFile.task_id.is_(None),
        )
        .first()
    )
    if uploaded_file is not None:
        selected_file_ids.append(str(uploaded_file.file_id))

    assert selected_file_ids == []
    db_session.refresh(bound_file)
    assert bound_file.task_id == 10


def test_selected_file_refs_from_task_revalidates_owner_and_task_binding(
    db_session,
    tmp_path,
):
    _create_user(db_session, 1, "owner")
    _create_user(db_session, 2, "other")
    task = _create_task(
        db_session,
        task_id=10,
        user_id=1,
        selected_file_ids=[
            "task-file",
            "unbound-file",
            "other-user-file",
            "other-task-file",
            "missing-file",
            "task-file",
        ],
    )
    _create_task(db_session, task_id=11, user_id=1)
    _create_uploaded_file(
        db_session,
        tmp_path,
        file_id="task-file",
        user_id=1,
        task_id=10,
        filename="task.txt",
    )
    _create_uploaded_file(
        db_session,
        tmp_path,
        file_id="unbound-file",
        user_id=1,
        task_id=None,
        filename="unbound.txt",
    )
    _create_uploaded_file(
        db_session,
        tmp_path,
        file_id="other-user-file",
        user_id=2,
        task_id=None,
        filename="other-user.txt",
    )
    _create_uploaded_file(
        db_session,
        tmp_path,
        file_id="other-task-file",
        user_id=1,
        task_id=11,
        filename="other-task.txt",
    )

    assert _selected_file_refs_from_task(task, db_session) == [
        {
            "file_id": "task-file",
            "name": "task.txt",
            "size": len("file content"),
            "type": "text/plain",
        },
        {
            "file_id": "unbound-file",
            "name": "unbound.txt",
            "size": len("file content"),
            "type": "text/plain",
        },
    ]


def test_selected_file_refs_from_task_ignores_missing_config(db_session):
    _create_user(db_session, 1, "owner")
    task = _create_task(db_session, task_id=10, user_id=1)

    assert _selected_file_refs_from_task(task, db_session) == []


def test_normalize_file_outputs_rejects_foreign_storage_path_record(
    db_session,
    tmp_path,
    monkeypatch,
):
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    _create_user(db_session, 1, "owner")
    _create_user(db_session, 2, "other")
    _create_task(db_session, task_id=20, user_id=1)
    _create_task(db_session, task_id=10, user_id=2)
    foreign_path = uploads_dir / "user_2" / "web_task_10" / "output" / "secret.txt"
    foreign_path.parent.mkdir(parents=True)
    foreign_path.write_text("secret")
    db_session.add(
        UploadedFile(
            file_id="foreign-output",
            user_id=2,
            task_id=10,
            filename="secret.txt",
            storage_path=str(foreign_path),
            mime_type="text/plain",
            file_size=len("secret"),
        )
    )
    db_session.flush()

    normalized_outputs, path_to_file_id = _normalize_file_outputs(
        db_session,
        task_id=20,
        task_user_id=1,
        file_outputs=[{"path": str(foreign_path)}],
    )

    assert normalized_outputs == []
    assert path_to_file_id == {}


def test_normalize_file_outputs_rejects_foreign_untracked_storage_path(
    db_session,
    tmp_path,
    monkeypatch,
):
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    _create_user(db_session, 1, "owner")
    _create_task(db_session, task_id=20, user_id=1)
    foreign_path = uploads_dir / "user_2" / "web_task_10" / "output" / "secret.txt"
    foreign_path.parent.mkdir(parents=True)
    foreign_path.write_text("secret")

    normalized_outputs, path_to_file_id = _normalize_file_outputs(
        db_session,
        task_id=20,
        task_user_id=1,
        file_outputs=[{"path": str(foreign_path)}],
    )

    assert normalized_outputs == []
    assert path_to_file_id == {}
    assert db_session.query(UploadedFile).count() == 0


def test_normalize_file_outputs_registers_current_task_output_path(
    db_session,
    tmp_path,
    monkeypatch,
):
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    _create_user(db_session, 1, "owner")
    _create_task(db_session, task_id=20, user_id=1)
    output_path = uploads_dir / "user_1" / "web_task_20" / "output" / "report.txt"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("report")

    normalized_outputs, path_to_file_id = _normalize_file_outputs(
        db_session,
        task_id=20,
        task_user_id=1,
        file_outputs=[{"path": str(output_path), "filename": "report.txt"}],
    )

    assert len(normalized_outputs) == 1
    assert normalized_outputs[0]["filename"] == "report.txt"
    assert path_to_file_id[str(output_path)] == normalized_outputs[0]["file_id"]
    file_record = db_session.query(UploadedFile).one()
    assert file_record.user_id == 1
    assert file_record.task_id == 20
    assert file_record.storage_path == str(output_path)


@pytest.mark.asyncio
async def test_handle_file_upload_for_task_rejects_unowned_and_wrong_task_files(
    db_session,
    tmp_path,
    monkeypatch,
):
    _create_user(db_session, 1, "owner")
    _create_user(db_session, 2, "other")
    _create_task(db_session, task_id=10, user_id=1)
    _create_task(db_session, task_id=11, user_id=1)
    valid_file = _create_uploaded_file(
        db_session,
        tmp_path,
        file_id="valid-file",
        user_id=1,
        task_id=None,
        filename="valid.txt",
    )
    _create_uploaded_file(
        db_session,
        tmp_path,
        file_id="other-user-file",
        user_id=2,
        task_id=None,
        filename="other-user.txt",
    )
    _create_uploaded_file(
        db_session,
        tmp_path,
        file_id="other-task-file",
        user_id=1,
        task_id=11,
        filename="other-task.txt",
    )

    import xagent.web.api.chat as chat_api

    monkeypatch.setattr(
        chat_api,
        "get_agent_manager",
        lambda: pytest.fail("file staging must not create an AgentService"),
    )

    result = await handle_file_upload_for_task(
        10,
        [
            {"file_id": "other-user-file"},
            {"file_id": "other-task-file"},
            {"file_id": "valid-file"},
        ],
        db_session,
        SimpleNamespace(id=1, is_admin=False),
        task_owner_id=1,
    )

    assert [item["file_id"] for item in result["file_info_list"]] == ["valid-file"]
    db_session.refresh(valid_file)
    assert valid_file.task_id == 10


def test_register_uploaded_files_for_agent_uses_execution_db_session(
    db_session,
    tmp_path,
):
    upload = _create_uploaded_file(
        db_session,
        tmp_path,
        file_id="valid-file",
        user_id=1,
        task_id=10,
        filename="valid file.txt",
    )

    class Workspace:
        def __init__(self):
            self.input_dir = str(tmp_path / "workspace" / "input")
            self.registered_files = []

        def register_file(self, path, file_id, db_session=None):
            self.registered_files.append((path, file_id, db_session))

    workspace = Workspace()
    file_info = {
        "file_id": "valid-file",
        "name": "valid_file.txt",
        "path": str(upload.storage_path),
        "workspace_path": None,
    }

    _register_uploaded_files_for_agent(
        SimpleNamespace(workspace=workspace),
        [file_info],
        db_session,
    )

    assert [item[1] for item in workspace.registered_files] == ["valid-file"]
    assert workspace.registered_files[0][2] is db_session
    assert file_info["workspace_path"]


def test_get_display_user_message_reads_agent_context_state():
    context = SimpleNamespace(
        state={
            "display_user_message": "Summarize this document",
        }
    )

    assert (
        get_display_user_message(
            context,
            "Summarize this document\n\n## UPLOADED FILES\nfile_id=file-123",
        )
        == "Summarize this document"
    )


def test_get_display_user_message_prefers_latest_message_metadata():
    context = SimpleNamespace(
        metadata={
            "display_user_message": "First turn",
        },
        messages=[
            SimpleNamespace(
                role="user",
                content="First turn",
                metadata={"display_message": "First turn"},
            ),
            SimpleNamespace(
                role="user",
                content="Second turn\n\n## UPLOADED FILES\nfile_id=file-123",
                metadata={"display_message": "Second turn"},
            ),
        ],
    )

    assert get_display_user_message(context, "fallback") == "Second turn"


def test_get_display_user_message_does_not_reuse_stale_context_display():
    context = SimpleNamespace(
        metadata={
            "display_user_message": "First turn",
        },
        messages=[
            SimpleNamespace(
                role="user",
                content="First turn",
                metadata={"display_message": "First turn"},
            ),
            SimpleNamespace(
                role="user",
                content="Second turn",
                metadata={},
            ),
        ],
    )

    assert get_display_user_message(context, "fallback") == "Second turn"


def test_get_display_user_message_respects_empty_display_metadata():
    context = SimpleNamespace(
        messages=[
            SimpleNamespace(
                role="user",
                content="Internal prompt\n\n## UPLOADED FILES\nfile_id=file-123",
                metadata={"display_message": ""},
            ),
        ],
    )

    assert get_display_user_message(context, "fallback") == ""


def test_display_message_for_file_only_turn_uses_placeholder():
    assert _display_message_for_user("", has_files=True) == "Uploaded file(s)"
    assert (
        _display_message_for_user("Summarize this document", has_files=True)
        == "Summarize this document"
    )


def test_display_file_refs_from_file_info_omits_runtime_paths():
    refs = _display_file_refs_from_file_info(
        [
            {
                "file_id": 123,
                "name": "notes.txt",
                "size": 42,
                "type": "text/plain",
                "path": "/internal/uploads/notes.txt",
                "workspace_path": "/workspace/input/notes.txt",
            }
        ]
    )

    assert refs == [
        {
            "file_id": "123",
            "name": "notes.txt",
            "size": 42,
            "type": "text/plain",
        }
    ]
    assert "path" not in refs[0]
    assert "workspace_path" not in refs[0]
