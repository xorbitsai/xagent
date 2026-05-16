from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.agent.trace import get_display_user_message
from xagent.web.api.chat import _build_task_agent_config
from xagent.web.api.websocket import (
    _append_uploaded_files_context_to_message,
    _build_uploaded_files_context,
    _display_message_for_user,
    _selected_file_refs_from_task,
    handle_file_upload_for_task,
)
from xagent.web.models import Base
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
) -> Task:
    task = Task(
        id=task_id,
        user_id=user_id,
        title=f"task-{task_id}",
        description="task",
        status=TaskStatus.PENDING,
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
    assert "call prepare_html_asset(file_id, alias) first" in context
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

    class Workspace:
        def __init__(self):
            self.input_dir = str(tmp_path / "workspace" / "input")
            self.registered_files = []

        def register_file(self, path, file_id, db_session=None):
            self.registered_files.append((path, file_id, db_session))

    workspace = Workspace()

    class Manager:
        async def get_agent_for_task(self, task_id, db, user=None):
            return SimpleNamespace(workspace=workspace)

    import xagent.web.api.chat as chat_api

    monkeypatch.setattr(chat_api, "get_agent_manager", lambda: Manager())

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
    assert [item[1] for item in workspace.registered_files] == ["valid-file"]
    db_session.refresh(valid_file)
    assert valid_file.task_id == 10


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


def test_display_message_for_file_only_turn_uses_placeholder():
    assert _display_message_for_user("", has_files=True) == "Uploaded file(s)"
    assert (
        _display_message_for_user("Summarize this document", has_files=True)
        == "Summarize this document"
    )
