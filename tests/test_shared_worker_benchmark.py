"""Benchmark prewarm guards; no service, database, or paid model calls."""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from scripts import shared_worker_benchmark as benchmark
from scripts import shared_worker_report as report
from scripts import shared_worker_workload as workload
from scripts.shared_worker_benchmark import (
    parse_args,
    validate_prewarm,
    warmed_execution_pids,
)


def test_naive_database_timestamp_is_utc():
    naive = datetime(2026, 1, 1)
    assert workload.timestamp(naive) == naive.replace(tzinfo=timezone.utc).timestamp()


@pytest.mark.parametrize("tasks", [[], [{"id": 2, "status": "completed"}]])
def test_missing_or_wrong_persisted_task_is_not_drained(tasks):
    assert not workload.all_terminal({"tasks": tasks}, [1])


def test_drained_failure_is_not_success():
    assert workload.all_terminal({"tasks": [{"id": 1, "status": "failed"}]}, [1])
    with pytest.raises(RuntimeError, match="Continuous"):
        benchmark.validate_probes(
            {
                "rows": [{"id": 1, "http": 202, "ordinal": 0}],
                "tasks": [{"id": 1, "status": "failed"}],
            },
            1,
        )


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "timeout", "rejected", "ordinal"]
)
def test_continuous_probe_validation(fault):
    data = {
        "rows": [{"id": 1, "http": 202, "ordinal": 0}],
        "tasks": [{"id": 1, "status": "completed"}],
    }
    if fault == "missing":
        data["tasks"] = []
    elif fault == "duplicate":
        data["rows"] *= 2
    elif fault == "timeout":
        data["timeout"] = True
    elif fault == "rejected":
        data["rows"][0]["http"] = 429
    else:
        data["rows"][0]["ordinal"] = 2
    with pytest.raises(RuntimeError, match="Continuous"):
        benchmark.validate_probes(data, 1)
    benchmark.validate_probes({"rows": [], "tasks": []}, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["timeout", "non_json", "rejected", "missing"])
async def test_submit_retains_failures_without_sensitive_bodies(kind):
    def handler(request):
        if kind == "timeout":
            raise httpx.ReadTimeout("private token", request=request)
        if kind == "non_json":
            return httpx.Response(503, text="private token")
        if kind == "rejected":
            return httpx.Response(429, json={"detail": "private token"})
        return httpx.Response(202, json={})

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        row = await workload.submit(client, (1, {}), "private prompt")
    assert row["unknown_creation"]
    assert row["api_ms"] >= 0
    assert row["error"]
    assert "private" not in json.dumps(row)


@pytest.mark.asyncio
async def test_workload_http_contract_and_complete_round(monkeypatch):
    captured, task_ids = [], []
    real_client = httpx.AsyncClient
    original_sleep = asyncio.sleep

    async def fast_sleep(_):
        await original_sleep(0)

    def handler(request):
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "private-login"})
        if request.url.path == "/api/agents":
            body = json.loads(request.content)
            assert set(body["models"].values()) == {7}
            return httpx.Response(200, json={"id": 2 if body["tool_categories"] else 1})
        if request.url.path.endswith("/publish"):
            return httpx.Response(200, json={})
        if request.url.path.endswith("/api-key"):
            return httpx.Response(200, json={"full_key": "private-api"})
        assert request.url.path == "/v1/chat/tasks"
        task_ids.append(len(task_ids) + 1)
        return httpx.Response(202, json={"task_id": task_ids[-1]})

    def state(ids):
        return {
            "tasks": [
                {"id": key, "status": "completed", "updated": 1000} for key in ids
            ],
            "events": [
                {
                    "id": key,
                    "type": "tool_execution_end",
                    "count": 10,
                    "success_count": 10,
                }
                for key in ids
            ],
        }

    monkeypatch.setenv("XAGENT_BENCH_USERNAME", "test")
    monkeypatch.setenv("XAGENT_BENCH_PASSWORD", "private-password")
    monkeypatch.setattr(workload, "snapshot", state)
    monkeypatch.setattr(workload, "emit", captured.append)
    monkeypatch.setattr(workload.asyncio, "sleep", fast_sleep)
    monkeypatch.setattr(
        workload.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    await workload.run(
        SimpleNamespace(
            base_url="http://test",
            model_id=7,
            background=2,
            probes=2,
            probe_interval=2,
            spacing=0,
            task_timeout=5,
        )
    )
    assert (
        len(task_ids) == 15
    )  # connectivity + two tools + ten initial + two continuous
    assert benchmark.validate_prewarm(captured, 2) == {2, 3}
    benchmark.validate_probes(
        next(row for row in captured if row["type"] == "probes"), 2
    )
    assert "private" not in json.dumps(captured)


@pytest.fixture
def case_folder(tmp_path):
    submitted = [
        {"id": key, "http": 202, "api_ms": 10, "sent": 100.0} for key in range(2, 15)
    ]
    tasks = [
        {"id": row["id"], "status": "completed", "updated": 102.0} for row in submitted
    ]
    manifest = {
        "complete": True,
        "workers": 1,
        "background_tasks": 1,
        "continuous_probes": 2,
        "client_exit_codes": [0],
        "forced_shutdown_pids": [],
        "hosts": [{"pid": 1, "role": "web"}, {"pid": 2, "role": "worker"}],
        "prewarm_requested": False,
        "workload_started_at": 90,
        "probe_interval_s": 2,
        "model_id": 7,
        "prompt_sha256": "prompt",
        "harness_sha256": {},
        "task_timeout_s": 600,
    }
    records = [
        {"type": "warm", "tasks": [{"id": 1, "status": "completed"}]},
        {"type": "admission", "background": submitted[:1], "probes": submitted[1:11]},
        {
            "type": "completed",
            "tasks": tasks[:11],
            "events": [
                {
                    "id": 2,
                    "type": "tool_execution_end",
                    "count": 10,
                    "success_count": 10,
                }
            ],
        },
        {
            "type": "probes",
            "rows": [
                dict(row, ordinal=index, active_background=1)
                for index, row in enumerate(submitted[11:])
            ],
            "tasks": tasks[11:],
        },
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "workload.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records)
    )
    (tmp_path / "model-entries-2.json").write_text(
        json.dumps([{"task_id": key, "time": 101.0} for key in range(1, 15)])
    )
    return tmp_path


def test_report_uses_model_clock_and_excludes_warmup(case_folder):
    result = report.read_case(case_folder)
    assert result["valid"]
    assert result["background"]["first_model_s"]["p95"] == 1.0
    assert result["background"]["total_s"]["p95"] == 2.0
    assert result["background_by_worker"] == {2: 1}
    assert len(result["raw"]["continuous"]) == 2


@pytest.mark.parametrize(
    "fault",
    [
        "missing_model",
        "negative_clock",
        "duplicate_worker",
        "tool_failure",
        "failed_task",
        "missing_probe",
        "forced_stop",
    ],
)
def test_report_rejects_incomplete_evidence(case_folder, fault):
    entries_path = case_folder / "model-entries-2.json"
    if fault in ("missing_model", "negative_clock"):
        entries = json.loads(entries_path.read_text())
        if fault == "missing_model":
            entries = []
        else:
            entries[1]["time"] = 99.0
        entries_path.write_text(json.dumps(entries))
    elif fault == "duplicate_worker":
        (case_folder / "model-entries-3.json").write_text(entries_path.read_text())
    elif fault == "forced_stop":
        path = case_folder / "manifest.json"
        value = json.loads(path.read_text())
        value["forced_shutdown_pids"] = [2]
        path.write_text(json.dumps(value))
    else:
        path = case_folder / "workload.jsonl"
        values = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == "tool_failure":
            values[2]["events"][0]["success_count"] = 9
        elif fault == "failed_task":
            values[2]["tasks"][0]["status"] = "failed"
        else:
            values[3]["rows"].pop()
        path.write_text("\n".join(json.dumps(row) for row in values))
    assert not report.read_case(case_folder)["valid"]


def test_full_load_comparison_does_not_include_idle_tail(case_folder):
    first = report.read_case(case_folder)
    second = report.read_case(case_folder)
    second["raw"]["continuous"][1]["active_background"] = 0
    comparison = report.compare_cases([first, second])
    assert comparison["matched_full_load_ordinals"] == [0]
    assert comparison["matched_full_load"][0]["first_model_s"]["n"] == 1
    second["manifest"]["prewarm_requested"] = True
    assert not report.compare_cases([first, second])["valid"]


def test_missing_percentile_is_not_zero():
    assert report.stats([None]) == {"n": 0}


def test_owned_port_and_output_guards(tmp_path):
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(RuntimeError, match="occupied"):
            benchmark.assert_free(listener.getsockname()[1])
    output = tmp_path / "data.json"
    benchmark.save(output, {"old": True})
    with pytest.raises(FileExistsError):
        benchmark.save(output, {"new": True})
    assert json.loads(output.read_text()) == {"old": True}


def test_isolated_child_pool_configuration():
    args = SimpleNamespace(redis_port=6385)
    base = {"XAGENT_BENCH_PASSWORD": "private", "XAGENT_BENCH_USERNAME": "user"}
    one = benchmark.host_environment(base, args, "worker", 1)
    four = benchmark.host_environment(base, args, "worker", 4)
    assert "XAGENT_BENCH_PASSWORD" not in one
    assert int(one["XAGENT_DB_POOL_SIZE"]) == 4 * int(four["XAGENT_DB_POOL_SIZE"])
    assert int(one["XAGENT_TRACE_DB_MAX_INFLIGHT"]) == 4 * int(
        four["XAGENT_TRACE_DB_MAX_INFLIGHT"]
    )


def test_private_environment_does_not_inherit_shell_secrets(tmp_path, monkeypatch):
    import dotenv

    worker = tmp_path / "src/xagent/web/worker.py"
    worker.parent.mkdir(parents=True)
    worker.touch()
    config = {
        "DATABASE_URL": "postgresql://test@localhost/test",
        "ENCRYPTION_KEY": "test-key",
        "XAGENT_JWT_SECRET": "test-jwt",
        "XAGENT_STORAGE_ROOT": str(tmp_path),
        "XAGENT_BENCH_USERNAME": "test",
        "XAGENT_BENCH_PASSWORD": "test-password",
    }
    monkeypatch.setenv("OPENAI_API_KEY", "shell-secret-must-not-propagate")
    monkeypatch.setenv("PYTHONPATH", "unrelated-checkout")
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **kw: dict(config))
    monkeypatch.setattr(
        benchmark.inspect, "getsource", lambda _: "PYTHON_DOTENV_DISABLED"
    )
    args = SimpleNamespace(
        confirm_test_environment=True,
        repo=tmp_path,
        port=8002,
        redis_port=6385,
        server_env=tmp_path / "private.env",
    )
    env = benchmark.read_environment(args)
    assert "OPENAI_API_KEY" not in env
    assert env["PYTHON_DOTENV_DISABLED"] == "1"
    assert env["PYTHONPATH"].startswith(str(tmp_path / "src"))
    args.confirm_test_environment = False
    with pytest.raises(ValueError, match="confirm"):
        benchmark.read_environment(args)
    args.confirm_test_environment = True
    config.pop("DATABASE_URL")
    with pytest.raises(ValueError, match="missing required"):
        benchmark.read_environment(args)


@pytest.mark.asyncio
async def test_model_observer_preserves_calls_and_closes_stream(tmp_path, monkeypatch):
    import atexit

    from scripts import shared_worker_observer as observer
    from xagent.core.model.chat.basic import openai
    from xagent.web.services import task_lease_service

    calls, dumps = [], []

    class Adapter:
        async def chat(self, value):
            calls.append(value)
            return value

        async def stream_chat(self, value):
            try:
                yield value
                yield "second"
            finally:
                calls.append("closed")

    monkeypatch.setattr(openai, "OpenAICompatibleLLM", Adapter)
    monkeypatch.setattr(
        task_lease_service, "current_task_lease", lambda: SimpleNamespace(task_id=7)
    )
    monkeypatch.setattr(atexit, "register", dumps.append)
    observer.install(tmp_path)
    adapter = Adapter()
    assert await adapter.chat("private-payload") == "private-payload"
    stream = adapter.stream_chat("first")
    assert await anext(stream) == "first"
    await stream.aclose()
    assert calls == ["private-payload", "closed"]
    dumps[0]()
    entries = json.loads(next(tmp_path.glob("model-entries-*.json")).read_text())
    assert len(entries) == 1
    assert entries[0]["task_id"] == 7
    assert set(entries[0]) == {"task_id", "time"}


@pytest.fixture
def warm_rows():
    def requests(ids):
        return [{"id": task_id, "http": 202} for task_id in ids]

    def tasks(ids):
        return [{"id": task_id, "status": "completed"} for task_id in ids]

    return [
        {"type": "warm", "rows": requests([1]), "tasks": tasks([1])},
        {
            "type": "admission",
            "background_n": 8,
            "background": requests(range(2, 10)),
            "probes": requests(range(10, 20)),
        },
        {
            "type": "completed",
            "background_n": 8,
            "tasks": tasks(range(2, 20)),
            "events": [
                {"id": task_id, "type": "tool_execution_end", "count": 10}
                for task_id in range(2, 10)
            ],
        },
    ]


def test_warmup_enabled_by_default_and_explicit_cold_mode():
    required = [
        "--server-env",
        "/private/test.env",
        "--output",
        "/private/results",
        "--model-id",
        "1",
    ]
    assert parse_args(required).prewarm
    assert parse_args(required + ["--prewarm"]).prewarm
    assert not parse_args(required + ["--no-prewarm"]).prewarm


def test_only_completed_background_tool_tasks_count(warm_rows):
    assert validate_prewarm(warm_rows) == set(range(2, 10))


def test_configured_sixteen_task_warmup(warm_rows):
    extra_ids = range(20, 28)
    warm_rows[1]["background_n"] = warm_rows[2]["background_n"] = 16
    warm_rows[1]["background"].extend(
        {"id": task_id, "http": 202} for task_id in extra_ids
    )
    warm_rows[2]["tasks"].extend(
        {"id": task_id, "status": "completed"} for task_id in extra_ids
    )
    warm_rows[2]["events"].extend(
        {"id": task_id, "type": "tool_execution_end", "count": 10}
        for task_id in extra_ids
    )
    assert validate_prewarm(warm_rows, 16) == set(range(2, 10)) | set(extra_ids)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_missing_stage_is_rejected(warm_rows, index):
    del warm_rows[index]
    with pytest.raises(RuntimeError, match="exactly one"):
        validate_prewarm(warm_rows)


def test_repeated_stage_is_rejected(warm_rows):
    warm_rows.append(warm_rows[-1])
    with pytest.raises(RuntimeError, match="exactly one completed"):
        validate_prewarm(warm_rows)


def test_zero_exit_safety_stop_is_rejected(warm_rows):
    warm_rows.append({"type": "safety_stop"})
    with pytest.raises(RuntimeError, match="safety stop"):
        validate_prewarm(warm_rows)


@pytest.mark.parametrize("stage, key", [(0, "rows"), (1, "background"), (1, "probes")])
def test_rejected_submission_is_not_ignored(warm_rows, stage, key):
    warm_rows[stage][key][0].update(http=429, id=None)
    with pytest.raises(RuntimeError, match="rejected or missing"):
        validate_prewarm(warm_rows)


@pytest.mark.parametrize("stage", [0, 2])
def test_partial_database_snapshot_is_rejected(warm_rows, stage):
    warm_rows[stage]["tasks"].pop()
    with pytest.raises(RuntimeError, match="did not all complete"):
        validate_prewarm(warm_rows)


@pytest.mark.parametrize("status", ["failed", "cancelled", "paused", "running"])
def test_non_completed_task_is_rejected(warm_rows, status):
    warm_rows[2]["tasks"][0]["status"] = status
    with pytest.raises(RuntimeError, match="did not all complete"):
        validate_prewarm(warm_rows)


@pytest.mark.parametrize("stage", [0, 2])
def test_timeout_is_rejected_even_with_completed_rows(warm_rows, stage):
    warm_rows[stage]["timeout"] = True
    with pytest.raises(RuntimeError, match="did not all complete"):
        validate_prewarm(warm_rows)


def test_duplicate_submission_is_rejected(warm_rows):
    warm_rows[1]["background"][0]["id"] = 3
    with pytest.raises(RuntimeError, match="did not all complete"):
        validate_prewarm(warm_rows)


def test_wrong_task_snapshot_is_rejected(warm_rows):
    warm_rows[2]["tasks"][0]["id"] = 99
    with pytest.raises(RuntimeError, match="did not all complete"):
        validate_prewarm(warm_rows)


def test_reused_warm_task_id_is_rejected(warm_rows):
    warm_rows[0]["rows"][0]["id"] = 2
    warm_rows[0]["tasks"][0]["id"] = 2
    with pytest.raises(RuntimeError, match="reused"):
        validate_prewarm(warm_rows)


def test_unexpected_workload_is_rejected(warm_rows):
    warm_rows[1]["background_n"] = 7
    with pytest.raises(RuntimeError, match="configured tool tasks"):
        validate_prewarm(warm_rows)


def test_tool_path_must_be_exercised(warm_rows):
    warm_rows[2]["events"].pop()
    with pytest.raises(RuntimeError, match="tool path"):
        validate_prewarm(warm_rows)


def test_each_process_needs_its_own_completed_tool_task(tmp_path):
    hosts = [{"name": f"worker-{index}", "pid": index} for index in range(1, 5)]
    for host, task_id in zip(hosts, [2, 3, 10, 999]):
        (tmp_path / (host["name"] + ".log")).write_text(
            f"Background task execution started for task {task_id}\n"
        )
    # QA-only (10) and still-running/unrelated (999) do not establish coverage.
    assert warmed_execution_pids(hosts, tmp_path, set(range(2, 10))) == {1, 2}
    (tmp_path / "worker-3.log").write_text(
        "Background task execution started for task 4\n"
    )
    (tmp_path / "worker-4.log").write_text(
        "Background task execution started for task 5\n"
    )
    assert warmed_execution_pids(hosts, tmp_path, set(range(2, 10))) == {1, 2, 3, 4}


def test_combined_single_process_is_warmed(tmp_path):
    (tmp_path / "web.log").write_text("Background task execution started for task 2\n")
    assert warmed_execution_pids([{"name": "web", "pid": 1}], tmp_path, {2}) == {1}


@pytest.mark.asyncio
@pytest.mark.parametrize("spacing", [0, 0.2])
async def test_warmup_client_pacing_does_not_change_default_burst(monkeypatch, spacing):
    from scripts import shared_worker_workload as module

    sleeps, submitted = [], []
    original_sleep = asyncio.sleep

    async def sleep(delay):
        sleeps.append(delay)
        await original_sleep(0)

    async def submit(config, prompt):
        submitted.append((config, prompt))
        return {"id": len(submitted)}

    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    result = await module.submit_background(submit, "agent", "arithmetic", 16, spacing)
    assert sleeps == ([0.2] * 15 if spacing else [])
    assert submitted == [("agent", "arithmetic")] * 16
    assert result == [{"id": index} for index in range(1, 17)]
