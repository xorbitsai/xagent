"""Bounded, opt-in HTTP/WebSocket runtime baseline runner.

Exercises real task creation, execute_task over one WebSocket per task, event
consumption and persisted completion status, alongside independent login and
/health probes. This measures performance; it does not isolate runtime workers.

Safety and setup:
  Use a dedicated test deployment, database/storage root and account. Login
  replaces the account's refresh token. Tasks persist, consume model quota and
  may invoke tools with external side effects. Use controlled local model/tool
  services without external credentials and check account concurrency quotas.
  This script does not create accounts, configure models, load .env, start/stop
  the backend, retry requests, cancel server tasks, or delete data.
  Set XAGENT_BENCH_USERNAME and XAGENT_BENCH_PASSWORD via your local environment
  (use a secret manager or hidden prompt for the password).

Fixed workload:
  Save a task JSON file, replacing TEST_MODEL_ID with a configured test model:
    {
      "title": "runtime-baseline",
      "description": "Reply with exactly BASELINE_OK. Do not use tools.",
      "execution_mode": "flash",
      "llm_ids": ["TEST_MODEL_ID", "TEST_MODEL_ID", "TEST_MODEL_ID", "TEST_MODEL_ID"]
    }
  Pin all four model slots, or an agent_id with a recorded published config;
  avoid automatic routing. The short prompt validates the harness, not the
  multi-round workload. To reproduce that workload, configure a local model
  with fixed delays, response sizes and tool-call rounds, plus harmless local
  tools/MCP. Record their source revisions; this script does not emulate them.

Example (from the repository root, using the existing aiohttp dependency):
  python scripts/runtime_baseline.py --base-url http://127.0.0.1:8001
    --task-payload /absolute/path/to/task.json
    --output /absolute/path/to/new-baseline-directory
    --confirm-test-environment --stages 0,10,50 --duration 120
    --tasks-per-minute 12 --probe-rate 1 --task-timeout 300
    --prometheus-url http://127.0.0.1:9090
  Join the example lines into one command. The output directory must be new.
  Start small: --stages 0,1 --duration 10 --tasks-per-minute 6. Increase only
  after validating the workload. Stages are concurrency CAPS, not measured
  active-task counts: 62 arrivals/minute with 150-second tasks gives roughly
  155 steady-state tasks, not 630. Check actual running-task gauges.

Scheduling and failure handling:
  Each stage submits for --duration seconds, then drains client operations;
  probes continue through drain with separate connection capacity. Fixed-rate
  arrivals exceeding in-flight limits or generator scheduling capacity are
  recorded as shed, never queued indefinitely or replayed as a catch-up burst.
  --probe-inflight applies separately to login and health; size it for the
  intended rate and anticipated latency. Any probe shedding means the requested
  load was not met. Run the generator in a separate process, preferably host.
  The first task failure stops new tasks and skips later stages. Known uncertain
  IDs and unknown_creations are retained: a timed-out/failed POST can already
  have committed. Inspect server state before rerunning. Disconnecting a socket
  or cancelling this script is NOT server-side task cancellation.

Reports and interpretation:
  report.json contains config, payload SHA-256, stage timestamps, per-attempt
  timings/outcomes, task IDs, summaries, throughput and optional raw Prometheus
  series. report.md summarizes p95/p99 and outcomes. Credentials, tokens, task
  contents, response bodies and exception messages are not written by the
  runner. Prometheus labels can contain deployment identifiers; use a dedicated
  test monitoring endpoint. completed means the runner finished, not an SLO pass.
  Task latency includes creation, WebSocket execution and status confirmation;
  login/health include connection and response handling. Scheduler lag is
  separate. Task timeout covers the whole operation; HTTP timeout defaults to
  60 seconds. Throughput uses submission PLUS drain time. Failed/timed-out
  attempts remain in latency statistics (timeouts are censored lower bounds);
  shed attempts have no latency sample. Empty percentiles are unavailable, not
  zero. Collect hundreds of probes per stage; small-sample p99 is near the max.
  Warm up separately and repeat at least three times. Record server Git SHA,
  lockfile hash, DB version/pool limits, worker count, CPU/memory limits, model/
  tool revisions and delay/round/payload settings, generator configuration and
  OTel export/scrape intervals. Separate cold starts from steady-state runs.

Monitoring:
  Enable backend runtime OTel export first. --prometheus-url is a read-only
  Prometheus origin, NOT the OTLP endpoint. All xagent_* series are captured at
  15-second resolution after --metrics-settle (default 30 seconds); increase it
  for slower export/scrape intervals. metrics_ended_at includes late post-drain
  exports; align analysis with the separate stage start/end timestamps.
  Missing/error/empty data is unavailable, not zero lag. Preserve instance
  labels; do not average worker percentiles. Example PromQL (filter job/instance
  to the test deployment; assumes standard Collector suffix translation):
    histogram_quantile(0.99, sum by (le, job, instance) (
      rate(xagent_event_loop_lag_milliseconds_bucket[2m])))
    histogram_quantile(0.95, sum by (le, job, instance, operation) (
      rate(xagent_thread_pool_queue_wait_duration_milliseconds_bucket[2m])))
    histogram_quantile(0.95, sum by (le, job, instance) (
      rate(xagent_trace_database_commit_duration_milliseconds_bucket[2m])))
  Compare client tails with loop lag, pool queue/execution times, trace encoding/
  commit, WebSocket broadcast and running-task gauges in the same window. Use
  separate host/container tools for CPU/RSS/GIL; metrics are not a CPU profiler.

Harness verification:
  PYTHONPATH=src:. pytest tests/test_runtime_baseline.py -q
  Tests use an ephemeral local HTTP/WebSocket fixture, not the user's backend,
  database or model services. Fixture timings are not XAgent capacity evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp


def percentile(values, fraction):
    """Nearest-rank percentile; empty samples are unavailable, not zero."""
    return (
        sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]
        if values
        else None
    )


def summarize(records):
    result = {}
    for operation in sorted({r["operation"] for r in records}):
        samples = [r for r in records if r["operation"] == operation]
        measured = [r for r in samples if r["outcome"] != "shed"]
        durations = [r["duration_ms"] for r in measured]
        result[operation] = {
            "scheduled": len(samples),
            "measured": len(measured),
            "outcomes": dict(Counter(r["outcome"] for r in samples)),
            "p50_ms": percentile(durations, 0.50),
            "p95_ms": percentile(durations, 0.95),
            "p99_ms": percentile(durations, 0.99),
            "max_ms": max(durations, default=None),
            "scheduler_lag_p99_ms": percentile(
                [r["scheduler_lag_ms"] for r in samples], 0.99
            ),
        }
    return result


async def schedule(rate, duration, limit, operation, invoke, records, stop_when=None):
    """Fixed arrival times, bounded in-flight work, no unbounded waiting queue."""
    if rate == 0:
        return
    loop = asyncio.get_running_loop()
    start = loop.time()
    pending = set()

    async def run(due):
        lag = max(0, loop.time() - due) * 1000
        began = loop.time()
        record = {"operation": operation, "scheduler_lag_ms": lag}
        try:
            record.update(await invoke())
        except Exception as exc:
            # Exception messages/URLs and response bodies can contain tokens.
            record.update(outcome="error", error_type=type(exc).__name__)
        record["duration_ms"] = (loop.time() - began) * 1000
        record["finished_at"] = time.time()
        records.append(record)

    try:
        for index in range(math.ceil(rate * duration)):
            due = start + index / rate
            await asyncio.sleep(max(0, due - loop.time()))
            if stop_when is not None and stop_when():
                break
            lag = max(0, loop.time() - due) * 1000
            # Do not burst missed arrivals after a stalled load generator.
            if len(pending) >= limit or lag > 1000 / rate:
                records.append(
                    dict(
                        operation=operation,
                        outcome="shed",
                        scheduler_lag_ms=lag,
                        finished_at=time.time(),
                    )
                )
                continue
            task = asyncio.create_task(run(due))
            pending.add(task)
            task.add_done_callback(pending.discard)
        if pending:
            await asyncio.gather(*pending)
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


class Runner:
    def __init__(self, base_url, username, password, payload, task_timeout):
        self.base_url = base_url.rstrip("/")
        self.credentials = {"username": username, "password": password}
        self.payload = payload
        self.task_timeout = task_timeout
        self.token = None
        self.uncertain = set()
        self.task_ids = []
        self.unknown_creations = 0
        self.stop_new_tasks = False

    async def login(self, session):
        async with session.post(
            self.base_url + "/api/auth/login", json=self.credentials
        ) as response:
            if response.status != 200:
                return {"outcome": "http_error", "status": response.status}
            data = await response.json()
            if not data.get("success") or not data.get("access_token"):
                return {"outcome": "invalid_login_response"}
            self.token = data["access_token"]
            return {"outcome": "success", "status": response.status}

    async def health(self, session):
        async with session.get(self.base_url + "/health") as response:
            await response.read()
            return {
                "outcome": "success" if response.status == 200 else "http_error",
                "status": response.status,
            }

    async def task(self, session):
        record = {"outcome": "error"}
        task_id = None
        try:
            async with asyncio.timeout(self.task_timeout):
                async with session.post(
                    self.base_url + "/api/chat/task/create",
                    json=self.payload,
                    headers={"Authorization": f"Bearer {self.token}"},
                ) as response:
                    if response.status != 200:
                        self.stop_new_tasks = True
                        if response.status >= 500:
                            self.unknown_creations += 1
                        return {
                            "outcome": "create_http_error",
                            "status": response.status,
                        }
                    task_id = int((await response.json())["task_id"])
                self.task_ids.append(task_id)
                self.uncertain.add(task_id)
                record["task_id"] = task_id
                ws_url = (
                    "wss" if self.base_url.startswith("https:") else "ws"
                ) + self.base_url[self.base_url.index(":") :]
                async with session.ws_connect(
                    f"{ws_url}/ws/chat/{task_id}",
                    params={"token": self.token},
                    max_msg_size=16 * 1024 * 1024,
                ) as socket:
                    await socket.send_json({"type": "execute_task"})
                    events = 0
                    async for message in socket:
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue
                        data = json.loads(message.data)
                        events += 1
                        if data.get("type") == "task_completed":
                            # Several completion producers omit `success`.
                            # Confirm the authoritative persisted status instead.
                            async with session.get(
                                f"{self.base_url}/api/chat/task/{task_id}/status",
                                headers={"Authorization": f"Bearer {self.token}"},
                            ) as status_response:
                                status_response.raise_for_status()
                                status = (await status_response.json()).get("status")
                            if status in {"completed", "failed", "cancelled"}:
                                self.uncertain.discard(task_id)
                            record.update(
                                outcome="success"
                                if status == "completed"
                                else "unconfirmed_completion",
                                events=events,
                            )
                            if record["outcome"] != "success":
                                self.stop_new_tasks = True
                            return record
                        if data.get("type") in {"error", "task_error", "agent_error"}:
                            record.update(outcome="task_error", events=events)
                            self.stop_new_tasks = True
                            return record
                    record["outcome"] = "socket_closed"
                    self.stop_new_tasks = True
        except Exception as exc:
            self.stop_new_tasks = True
            if task_id is None:
                # A timed-out POST may already have committed on the server.
                self.unknown_creations += 1
            record.update(outcome="error", error_type=type(exc).__name__)
        except asyncio.CancelledError:
            if task_id is None:
                self.unknown_creations += 1
            raise
        return record


async def collect_prometheus(session, url, started, ended):
    """Save raw per-series data, including instance labels; no cross-instance merge."""
    try:
        async with session.get(
            url.rstrip("/") + "/api/v1/query_range",
            params={
                "query": '{__name__=~"xagent_.*"}',
                "start": started,
                "end": ended,
                "step": "15s",
            },
        ) as response:
            response.raise_for_status()
            data = await response.json()
            if data.get("status") != "success":
                return {"error": "query_failed"}
            return data
    except Exception as exc:
        return {"error": type(exc).__name__}


def markdown(report):
    lines = [
        "# Runtime baseline",
        "",
        f"Status: {report['status']}",
        "",
        "Latency includes failed/timed-out attempts; shed arrivals have no latency sample.",
        "Task throughput covers submission plus drain. Caps are not measured active-task counts.",
        "",
        "| Stage | Operation | Measured / scheduled | p95 ms | p99 ms | Outcomes |",
        "|---|---|---:|---:|---:|---|",
    ]
    for stage in report["stages"]:
        for operation, value in stage["summary"].items():
            lines.append(
                f"| {stage['cap']} | {operation} | {value['measured']} / {value['scheduled']} | {value['p95_ms']} | {value['p99_ms']} | {value['outcomes']} |"
            )
    lines += [
        "",
        "Unconfirmed tasks (inspect before another run): "
        + str(report["uncertain_task_ids"]),
        "Unknown creations: " + str(report.get("unknown_creations", 0)),
        "",
        "Completed means the runner finished, not that a performance target passed.",
        "",
    ]
    return "\n".join(lines)


async def run(args):
    username = os.environ.get("XAGENT_BENCH_USERNAME")
    password = os.environ.get("XAGENT_BENCH_PASSWORD")
    if not username or not password:
        raise ValueError(
            "Set XAGENT_BENCH_USERNAME and XAGENT_BENCH_PASSWORD for a dedicated test account"
        )
    raw = args.task_payload.read_bytes()
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or not payload.get("title")
        or not payload.get("description")
    ):
        raise ValueError("Task payload must include title and a fixed description")
    if not payload.get("llm_ids") and not payload.get("agent_id"):
        raise ValueError("Pin llm_ids or agent_id in the task payload")
    # A new directory prevents accidental overwrites and separates comparison runs.
    args.output.mkdir(parents=True, exist_ok=False)
    runner = Runner(args.base_url, username, password, payload, args.task_timeout)
    report = {
        "status": "running",
        "stages": [],
        "uncertain_task_ids": [],
        "task_ids": runner.task_ids,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    # Do not store target URLs, which may contain environment-specific details.
    report["config"].pop("base_url")
    report["config"].pop("prometheus_url")
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    try:
        # Separate connector capacity for probes and long-lived task sockets.
        async with (
            aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=0),
                trust_env=False,
            ) as tasks,
            aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=0),
                trust_env=False,
            ) as probes,
        ):
            if (await runner.login(probes))["outcome"] != "success":
                raise ValueError("Test-account login preflight failed")
            if (await runner.health(probes))["outcome"] != "success":
                raise ValueError("Health preflight failed")
            for cap in args.stages:
                records = []
                stage = {"cap": cap, "started_at": time.time(), "records": records}
                report["stages"].append(stage)
                stage_started = asyncio.get_running_loop().time()
                # Probe arrivals continue during drain, not only during submissions.
                work = asyncio.create_task(
                    schedule(
                        args.tasks_per_minute / 60 if cap else 0,
                        args.duration,
                        max(cap, 1),
                        "task",
                        lambda: runner.task(tasks),
                        records,
                        stop_when=lambda: runner.stop_new_tasks,
                    )
                )

                async def probe_loop(operation, callback):
                    await schedule(
                        args.probe_rate,
                        args.duration + args.task_timeout + 1,
                        args.probe_inflight,
                        operation,
                        callback,
                        records,
                        stop_when=lambda: work.done()
                        and asyncio.get_running_loop().time() - stage_started
                        >= args.duration,
                    )

                await asyncio.gather(
                    work,
                    probe_loop("login", lambda: runner.login(probes)),
                    probe_loop("health", lambda: runner.health(probes)),
                )
                stage["ended_at"] = time.time()
                stage["summary"] = summarize(records)
                elapsed = stage["ended_at"] - stage["started_at"]
                stage["successful_tasks_per_second"] = (
                    sum(
                        r["operation"] == "task" and r["outcome"] == "success"
                        for r in records
                    )
                    / elapsed
                )
                if args.prometheus_url:
                    await asyncio.sleep(args.metrics_settle)
                    stage["metrics_ended_at"] = time.time()
                    stage["prometheus"] = await collect_prometheus(
                        probes,
                        args.prometheus_url,
                        stage["started_at"],
                        stage["metrics_ended_at"],
                    )
                print(
                    f"stage cap={cap}: {len(records)} samples; {len(runner.uncertain)} unconfirmed tasks",
                    flush=True,
                )
                if runner.uncertain or runner.unknown_creations:
                    report["status"] = "aborted_unconfirmed_tasks"
                    break
                if runner.stop_new_tasks:
                    report["status"] = "aborted_task_failure"
                    break
            else:
                report["status"] = "completed"
    finally:
        report["uncertain_task_ids"] = sorted(runner.uncertain)
        report["unknown_creations"] = runner.unknown_creations
        if report["status"] == "running":
            report["status"] = "interrupted"
        for stage in report["stages"]:
            stage["summary"] = summarize(stage["records"])
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (args.output / "report.md").write_text(markdown(report), encoding="utf-8")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--task-payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm-test-environment", action="store_true", required=True
    )
    parser.add_argument(
        "--stages", type=lambda s: [int(x) for x in s.split(",")], default=[0, 10, 50]
    )
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--tasks-per-minute", type=float, default=12)
    parser.add_argument("--probe-rate", type=float, default=1)
    parser.add_argument("--probe-inflight", type=int, default=16)
    parser.add_argument("--task-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--prometheus-url")
    parser.add_argument(
        "--metrics-settle",
        type=float,
        default=30,
        help="Post-drain delay for OTel export/scrape before querying stage data",
    )
    args = parser.parse_args(argv)
    for name in (
        "duration",
        "tasks_per_minute",
        "probe_rate",
        "probe_inflight",
        "task_timeout",
        "request_timeout",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if not args.stages or any(x < 0 for x in args.stages):
        parser.error("stages must be nonnegative concurrency caps")
    if not math.isfinite(args.metrics_settle) or args.metrics_settle < 0:
        parser.error("metrics_settle must be finite and nonnegative")
    for url in (args.base_url, args.prometheus_url):
        if url is None:
            continue
        parts = urlsplit(url)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
            or parts.path not in {"", "/"}
        ):
            parser.error(
                "URLs must be HTTP(S) origins without credentials, paths, query or fragment"
            )
    return args


if __name__ == "__main__":
    result = asyncio.run(run(parse_args()))
    raise SystemExit(0 if result["status"] == "completed" else 1)
