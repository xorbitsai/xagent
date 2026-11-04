"""Automatic resume of SDK tasks, as their caller sees it.

An SDK caller only reads a task's status, through ``GET /v1/chat/tasks/{id}``
or the ``/events`` stream. PAUSED is not a final status, so an interrupted
SDK run is resumed like any other task's while automatic resume is on, and
the caller ends on the resumed run's result. Once automatic resume stops, the
task stays PAUSED in storage but ``/v1`` reports it ``failed``, so a caller
does not wait on it for good. With automatic resume off a settlement
interruption fails the task, as it always did.
"""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest

from tests.e2e.shared_execution_harness import (
    SETTLE_TIMEOUT,
    answered_model_calls,
    auto_resume_commands,
    create_sdk_task,
    kill_host,
    model_calls,
    model_faults,
    recovery_state,
    set_model_fault,
    task_state,
    wait_for,
)

pytestmark = pytest.mark.e2e

# Provider attempts per model call (E2E_MODEL_RETRY_ATTEMPTS).
ATTEMPTS = 3
# Longer than any resume this module could schedule (see
# test_shared_provider_outage.py).
LLM_QUIET_SECONDS = 5
ANSWER = "Shared E2E answer"
FINAL_STATUSES = {"completed", "failed"}
TWO_RESUMES = [
    "interrupted",
    "scheduled",
    "auto_resumed",
    "interrupted",
    "scheduled",
    "auto_resumed",
]
# The /v1 text for a run whose automatic resume ran out of attempts; a
# literal on purpose, so a change to the published text fails here.
EXHAUSTED_ERROR = (
    "Task failed after a system interruption; automatic resume stopped after "
    "repeated attempts (reason: llm_unavailable)."
)


@pytest.fixture(autouse=True)
def _fast_auto_resume(monkeypatch):
    # Read by the hosts the shared_app fixture spawns after this.
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_POLL_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_SHORT_INITIAL_BACKOFF_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_LLM_INITIAL_BACKOFF_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_LLM_MAX_BACKOFF_SECONDS", "2")
    monkeypatch.setenv("E2E_MODEL_RETRY_ATTEMPTS", str(ATTEMPTS))
    monkeypatch.setenv("XAGENT_LLM_RETRY_DEADLINE_SECONDS", "30")
    for name in (
        "XAGENT_TASK_AUTO_RESUME_ENABLED",
        "XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED",
        "XAGENT_TASK_AUTO_RESUME_MAX_TOTAL_PER_RUN",
        "XAGENT_TASK_AUTO_RESUME_LLM_MAX_ELAPSED_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


# Per-test settings: requested before ``shared_app`` so its hosts see them.


@pytest.fixture
def two_resumes_per_run(monkeypatch):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_MAX_TOTAL_PER_RUN", "2")


@pytest.fixture
def auto_resume_off(monkeypatch):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")


def get_task(app, task_id: int, headers) -> dict:
    response = app.client.get(f"/v1/chat/tasks/{task_id}", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def poll_until_final(app, task_id: int, headers) -> tuple[dict, list[str]]:
    """Poll as an SDK caller does: the final view and every status seen."""
    seen: list[str] = []
    deadline = time.monotonic() + SETTLE_TIMEOUT
    while time.monotonic() < deadline:
        view = get_task(app, task_id, headers)
        if not seen or seen[-1] != view["status"]:
            seen.append(view["status"])
        if view["status"] in FINAL_STATUSES:
            return view, seen
        time.sleep(0.1)
    pytest.fail(f"SDK task stayed {seen}\n{app.diagnostics()}")


def _frames(lines) -> list[tuple[str, dict]]:
    """``(event, data)`` of SSE lines; comments (pings) are skipped."""
    frames, event = [], None
    for line in lines:
        if line.startswith("event: "):
            event = line.removeprefix("event: ")
        elif line.startswith("data: ") and event is not None:
            frames.append((event, json.loads(line.removeprefix("data: "))))
            event = None
    return frames


class LiveEvents:
    """The task's ``/events`` stream, read on a thread until it closes."""

    def __init__(self, app, task_id: int, headers):
        self.lines: list[str] = []
        self.error: BaseException | None = None
        self._client = httpx.Client(
            base_url=app.client.base_url, timeout=httpx.Timeout(10, read=60)
        )
        self._attached = threading.Event()
        self._thread = threading.Thread(
            target=self._read, args=(task_id, headers), daemon=True
        )
        self._thread.start()
        assert self._attached.wait(10), "events stream did not attach"

    def _read(self, task_id, headers):
        try:
            with self._client.stream(
                "GET", f"/v1/chat/tasks/{task_id}/events", headers=headers
            ) as response:
                assert response.status_code == 200
                self._attached.set()
                for line in response.iter_lines():
                    self.lines.append(line)
        except BaseException as exc:  # noqa: BLE001 - reported by close()
            self.error = exc
            self._attached.set()

    def close(self, app, timeout: float = SETTLE_TIMEOUT) -> list[tuple[str, dict]]:
        """Wait for the server to end the stream; its frames."""
        self._thread.join(timeout)
        self._client.close()
        if self._thread.is_alive() or self.error is not None:
            pytest.fail(
                f"events stream: {self.error!r} {self.lines}\n{app.diagnostics()}"
            )
        return _frames(self.lines)


def _statuses(frames) -> list[str]:
    return [data["status"] for event, data in frames if event == "task.status"]


# --------------------------------------------------------------------------
# Lease expiry: a killed worker's run is resumed for its SDK caller
# --------------------------------------------------------------------------


def test_sdk_task_resumes_after_its_worker_is_killed(shared_app):
    app = shared_app
    task_id, run_id, headers = create_sdk_task(app, "e2e:gate")
    wait_for(lambda: (app.root / "model-entered").exists(), app, "model call")
    live = LiveEvents(app, task_id, headers)

    kill_host(app, app.processes[1])
    # The resumed call may answer; the killed one never returns.
    (app.root / "model-release").touch()
    app.start("worker")

    view, seen = poll_until_final(app, task_id, headers)
    # The caller never read the run as failed: it kept polling through the
    # pause and got the resumed run's answer.
    assert "failed" not in seen, seen
    assert view["status"] == "completed", view
    assert ANSWER in (view["output"] or ""), view
    assert view["error"] is None, view
    # The stream opened while the run was live stayed open through the
    # pause and closed on the resumed run's result.
    frames = live.close(app)
    assert frames[-1][0] == "task.completed", frames
    assert frames[-1][1]["status"] == "completed", frames
    assert "failed" not in _statuses(frames), frames

    app.wait_task(task_id, run_id=run_id, timeout=SETTLE_TIMEOUT)
    # The killed call and the resumed one: nothing ran twice.
    assert len(model_calls(app)) == 2
    row, events = recovery_state(task_id)
    assert row is not None
    assert (row["reason"], row["run_id"], row["total_resumes"]) == (
        "lease_expired",
        run_id,
        1,
    )
    assert events == ["interrupted", "scheduled", "auto_resumed"]
    assert len(auto_resume_commands(task_id)) == 1


# --------------------------------------------------------------------------
# A provider outage pauses the SDK run; it finishes once the provider is back
# --------------------------------------------------------------------------


def test_sdk_task_resumes_after_a_provider_outage(shared_app):
    app = shared_app
    # Two whole model calls fail; the third call's first attempt answers.
    set_model_fault(app, "unavailable", count=2 * ATTEMPTS)
    task_id, run_id, headers = create_sdk_task(app, "e2e outage")

    view, seen = poll_until_final(app, task_id, headers)
    # It may read paused while it waits for a resume, never failed.
    assert "failed" not in seen, seen
    assert view["status"] == "completed", view
    assert ANSWER in (view["output"] or ""), view
    app.wait_task(task_id, run_id=run_id, timeout=SETTLE_TIMEOUT)
    assert len(answered_model_calls(app)) == 1
    row, events = recovery_state(task_id)
    assert row is not None
    assert (row["reason"], row["run_id"], row["total_resumes"], row["state"]) == (
        "llm_unavailable",
        run_id,
        2,
        "dispatched",
    )
    assert events == TWO_RESUMES
    assert len(auto_resume_commands(task_id)) == 2


# --------------------------------------------------------------------------
# Automatic resume gives up: the paused task reads failed on /v1
# --------------------------------------------------------------------------


def test_sdk_task_reads_failed_once_automatic_resume_stops(
    two_resumes_per_run, shared_app
):
    app = shared_app
    set_model_fault(app, "unavailable", count=None)
    task_id, run_id, headers = create_sdk_task(app, "e2e outage")

    view, seen = poll_until_final(app, task_id, headers)
    assert "completed" not in seen, seen
    row, events = recovery_state(task_id)
    assert (row["state"], row["reason"], row["total_resumes"]) == (
        "exhausted",
        "llm_unavailable",
        2,
    )
    assert events == TWO_RESUMES + ["interrupted", "exhausted"]
    # Stored, the task is still paused on its run: the failure is derived.
    state = task_state(task_id)
    assert (state["status"], state["control_state"], state["run_id"]) == (
        "paused",
        "paused",
        run_id,
    )
    assert view["status"] == "failed", view
    assert view["control_state"] == "failed", view
    assert view["error"] == EXHAUSTED_ERROR, view
    assert view["output"] is None, view
    assert view["completed_at"] is not None, view

    # A stream attached now reports the failure at once, with no pause first.
    frames = LiveEvents(app, task_id, headers).close(app, timeout=10)
    assert _statuses(frames) == ["failed"], frames
    assert frames[-1][0] == "task.completed", frames
    assert frames[-1][1]["status"] == "failed", frames
    assert frames[-1][1]["error"] == EXHAUSTED_ERROR, frames

    # Nothing resumes it on its own, and the derived view holds.
    time.sleep(LLM_QUIET_SECONDS)
    assert len(auto_resume_commands(task_id)) == 2
    assert len(model_faults(app)) == 3 * ATTEMPTS
    assert get_task(app, task_id, headers)["status"] == "failed"


# --------------------------------------------------------------------------
# With automatic resume off an SDK outage fails the task, as before
# --------------------------------------------------------------------------


def test_sdk_task_fails_on_a_provider_outage_with_auto_resume_off(
    auto_resume_off, shared_app
):
    app = shared_app
    set_model_fault(app, "unavailable", count=None)
    task_id, run_id, headers = create_sdk_task(app, "e2e outage")

    view, seen = poll_until_final(app, task_id, headers)
    assert "paused" not in seen, seen
    assert view["status"] == "failed", view
    # The run's stored error (the runner's legacy failure text), not a
    # derived automatic-resume one.
    settled = app.wait_task(task_id, status="failed", run_id=run_id)
    assert settled["error"], settled
    assert view["error"] == settled["error"], (view, settled)
    assert not view["error"].startswith("Task failed after a system interruption")
    # One model call, retried by tier 1 only, and never resumed.
    assert len(model_faults(app, "unavailable")) == ATTEMPTS
    row, events = recovery_state(task_id)
    assert row is not None
    assert (row["state"], row["reason"]) == ("manual", "llm_unavailable")
    assert events == ["interrupted"]
    assert auto_resume_commands(task_id) == []

    time.sleep(LLM_QUIET_SECONDS)
    assert task_state(task_id)["status"] == "failed"
    assert len(model_faults(app)) == ATTEMPTS
    assert recovery_state(task_id)[1] == ["interrupted"]
    assert auto_resume_commands(task_id) == []
