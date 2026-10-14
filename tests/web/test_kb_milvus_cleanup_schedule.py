"""The Milvus cleanup beat tick and its failure containment."""

from __future__ import annotations

import pytest


def test_beat_schedule_includes_the_cleanup_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_KB_MILVUS_CLEANUP_INTERVAL_SECONDS", "777")
    from xagent.web.jobs.celery_app import create_celery_app

    app = create_celery_app()
    entry = app.conf.beat_schedule["purge-leftover-milvus-rows"]

    assert entry["task"] == "xagent.web.jobs.trigger_tasks.purge_leftover_milvus_rows"
    assert entry["schedule"] == 777.0
    assert "xagent.web.jobs.trigger_tasks.purge_leftover_milvus_rows" in app.tasks


def test_the_task_reports_a_failed_purge_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.kb import milvus_cleanup
    from xagent.web.jobs import trigger_tasks

    def explode() -> dict[str, int]:
        raise RuntimeError("milvus is down")

    monkeypatch.setattr(milvus_cleanup, "purge_leftover_milvus_rows", explode)

    result = trigger_tasks.purge_leftover_milvus_rows()

    assert result["status"] == "failed"
    assert "milvus is down" in result["error"]


def test_the_task_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from xagent.core.tools.core.RAG_tools.kb import milvus_cleanup
    from xagent.web.jobs import trigger_tasks

    monkeypatch.setattr(milvus_cleanup, "purge_leftover_milvus_rows", lambda: {"c": 2})

    result = trigger_tasks.purge_leftover_milvus_rows()

    assert result["status"] == "ok"
    assert result["deleted"] == {"c": 2}
