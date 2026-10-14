"""The Milvus cleanup beat tick, its routing, and its failure containment."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_TASK = "xagent.web.jobs.trigger_tasks.purge_leftover_milvus_rows"


def _shipped_worker_queues() -> set[str]:
    compose = (Path(__file__).resolve().parents[2] / "docker-compose.yml").read_text()
    match = re.search(r'"-Q",\s*"([^"]+)"', compose)
    assert match, "the shipped worker command has no -Q queue list"
    return set(match.group(1).split(","))


def test_beat_schedule_includes_the_cleanup_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_KB_MILVUS_CLEANUP_INTERVAL_SECONDS", "777")
    from xagent.web.jobs.celery_app import create_celery_app

    app = create_celery_app()
    entry = app.conf.beat_schedule["purge-leftover-milvus-rows"]

    assert entry["task"] == _TASK
    assert entry["schedule"] == 777.0
    # An unconsumed enqueue must expire on the configured interval, not pile up.
    assert entry["options"]["expires"] == entry["schedule"]
    assert _TASK in app.tasks


def test_the_cleanup_task_is_routed_to_a_consumed_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_KB_MILVUS_CLEANUP_INTERVAL_SECONDS", "777")
    from xagent.web.jobs.celery_app import create_celery_app

    app = create_celery_app()
    route = app.conf.task_routes[_TASK]

    assert route == {"queue": "kb"}
    # The shipped worker consumes only the queues in its -Q list.
    assert route["queue"] in _shipped_worker_queues()


def test_the_cleanup_task_does_not_use_late_acks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_KB_MILVUS_CLEANUP_INTERVAL_SECONDS", "777")
    from xagent.web.jobs.celery_app import create_celery_app

    app = create_celery_app()

    assert app.tasks[_TASK].acks_late is False


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
