from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services.task_execution_context_service import (
    load_task_execution_context_messages,
    load_task_execution_recovery_state,
    summarize_execution_failure_event,
    summarize_tool_event,
)


def _create_db_session():
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return SessionLocal()


def _create_task(db_session):
    user = User(username="tester", password_hash="hashed_password", is_admin=False)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    task = Task(
        user_id=int(user.id),
        title="Execution context task",
        description="Task execution context",
        status=TaskStatus.COMPLETED,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)
    return task


def test_load_task_execution_context_messages_summarizes_list_files_results():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        db_session.add(
            TraceEvent(
                task_id=int(task.id),
                build_id=None,
                event_id="tool-event-1",
                event_type="tool_execution_end",
                timestamp=datetime.now(timezone.utc),
                step_id="step_1",
                parent_event_id=None,
                data={
                    "tool_name": "list_files",
                    "success": True,
                    "result": {
                        "input": [
                            {"filename": "foo.docx"},
                            {"filename": "bar.pdf"},
                        ],
                        "output": [],
                        "temp": [],
                        "workspace": [],
                    },
                },
            )
        )
        db_session.commit()

        messages = load_task_execution_context_messages(db_session, int(task.id))

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "Tool list_files previously returned" in messages[0]["content"]
        assert '"input"' in messages[0]["content"]
        assert "foo.docx" in messages[0]["content"]
        assert "bar.pdf" in messages[0]["content"]
    finally:
        db_session.close()


def test_summarize_tool_event_uses_generic_fallback_for_unknown_tools():
    summary = summarize_tool_event(
        {
            "tool_name": "web_search",
            "success": True,
            "result": {"top_result": "https://example.com", "title": "Example"},
        }
    )

    assert summary is not None
    assert summary.startswith("- Tool web_search previously returned: ")
    assert "top_result" in summary


def test_summarize_tool_event_omits_binary_payloads():
    summary = summarize_tool_event(
        {
            "tool_name": "read_file",
            "success": True,
            "tool_params": {"file_path": "logo.png"},
            "result": "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + ("\x80" * 100),
        }
    )

    assert summary is not None
    assert "non-text payload omitted" in summary
    assert "logo.png" in summary
    assert "PNG" not in summary
    assert "IHDR" not in summary


def test_summarize_tool_event_compacts_nested_large_strings():
    summary = summarize_tool_event(
        {
            "tool_name": "write_file",
            "success": True,
            "result": {
                "success": True,
                "file_path": "/tmp/report.html",
                "content": "<html>" + ("a" * 300) + "</html>",
            },
        }
    )

    assert summary is not None
    assert "/tmp/report.html" in summary
    assert "a" * 200 not in summary


def test_summarize_tool_event_preserves_file_ref_before_long_video_payload():
    file_id = "c6553861-5bdd-4628-9b15-1310e34fe499"
    markdown_link = f"[generated_video_253a6da9.mp4](file:{file_id})"

    summary = summarize_tool_event(
        {
            "tool_name": "generate_video",
            "success": True,
            "result": {
                "success": True,
                "video_url": "https://provider.example/video?signature=" + ("x" * 500),
                "file_ref": {
                    "file_id": file_id,
                    "filename": "generated_video_253a6da9.mp4",
                    "markdown_link": markdown_link,
                },
            },
        }
    )

    assert summary is not None
    assert summary.startswith("- Tool generate_video previously returned: files: ")
    assert markdown_link in summary
    assert file_id in summary


def test_summarize_tool_event_builds_link_from_file_id_and_filename():
    summary = summarize_tool_event(
        {
            "tool_name": "write_file",
            "success": True,
            "result": {
                "file_refs": [{"file_id": "report-id", "filename": "report.pdf"}]
            },
        }
    )

    assert summary is not None
    assert "[report.pdf](file:report-id)" in summary


def test_summarize_execution_failure_event_includes_step_anchor():
    summary = summarize_execution_failure_event(
        {
            "status": "failed",
            "result": {
                "success": False,
                "error": "Gemini SDK API error: connection reset by peer",
                "failure_reason": "step_failed",
                "failed_step_id": "write_html",
            },
        }
    )

    assert summary is not None
    assert "step=write_html" in summary
    assert "reason=step_failed" in summary
    assert "connection reset" in summary


def test_load_task_execution_context_messages_includes_latest_failure_anchor():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        db_session.add(
            TraceEvent(
                task_id=int(task.id),
                build_id=None,
                event_id="failure-event-1",
                event_type="dag_execute_end",
                timestamp=datetime.now(timezone.utc),
                step_id=str(task.id),
                parent_event_id=None,
                data={
                    "status": "failed",
                    "result": {
                        "success": False,
                        "error": "network reset",
                        "failure_reason": "step_failed",
                        "failed_step_id": "render_poster",
                    },
                },
            )
        )
        db_session.commit()

        messages = load_task_execution_context_messages(db_session, int(task.id))

        assert len(messages) == 1
        assert "Previous execution failed" in messages[0]["content"]
        assert "step=render_poster" in messages[0]["content"]
    finally:
        db_session.close()


@pytest.mark.asyncio
async def test_load_task_execution_recovery_state_recovers_skill_context(monkeypatch):
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        db_session.add(
            TraceEvent(
                task_id=int(task.id),
                build_id=None,
                event_id="skill-event-1",
                event_type="skill_select_end",
                timestamp=datetime.now(timezone.utc),
                step_id=None,
                parent_event_id=None,
                data={
                    "selected": True,
                    "skill_name": "translator",
                },
            )
        )
        db_session.commit()

        async def fake_load_skill_context_by_name(skill_name: str):
            assert skill_name == "translator"
            return "## Available Skill: translator\n\nUse translation workflow."

        monkeypatch.setattr(
            "xagent.web.services.task_execution_context_service._load_skill_context_by_name",
            fake_load_skill_context_by_name,
        )

        recovery_state = await load_task_execution_recovery_state(
            db_session, int(task.id)
        )

        assert recovery_state["skill_context"] == (
            "## Available Skill: translator\n\nUse translation workflow."
        )
        assert recovery_state["messages"] == []
    finally:
        db_session.close()
