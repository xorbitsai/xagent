"""Opt-in PostgreSQL shared-worker SDK benchmark (depends on PR #2411).

Run from the repository containing these scripts:
  python -m scripts.shared_worker_benchmark --repo /absolute/path/to/2411-checkout
    --server-env /absolute/path/to/private-benchmark.env --model-id 1
    --output /absolute/path/to/new-results --confirm-test-environment
Join the lines into one command. --case shared-1 or shared-4 selects one layout;
the default runs both sequentially. --background 8 --probes 2 is a small smoke.
--no-prewarm retains only one connectivity Q&A, NOT a fully cold HTTP process.

Requires PostgreSQL extra, Redis server on PATH, httpx and python-dotenv >= 1.2
(for PYTHON_DOTENV_DISABLED). The selected checkout must be installed into the
same interpreter's environment and have a migrated, dedicated test database,
private shared encryption/JWT keys, test account and configured OpenAI-compatible
model. model-id is the models table PK used by the agent builder, not an alias.
The private env file must supply DATABASE_URL, ENCRYPTION_KEY, XAGENT_JWT_SECRET,
XAGENT_STORAGE_ROOT, XAGENT_BENCH_USERNAME and XAGENT_BENCH_PASSWORD. Supply other
deployment settings explicitly there. The launcher never copies production data,
seeds users/models, changes model aliases, migrates schemas, or loads repo .env.

The env file is read without interpolation and never included in reports. Use
absolute paths and literal values. Use a disposable storage root with controlled
tools and no external tool credentials; arithmetic prompts are NOT a sandbox.
This creates/publishes agents, issues their API keys, changes the test user's
login token, consumes real model quota and retains tasks/traces. No server tasks
are retried/deleted/cancelled. On failure inspect persisted tasks before rerunning.
Owned host shutdown may leave recoverable tasks; do not reuse that DB blindly.
Server logs are private local artifacts and may contain server-generated data.

Only child hosts and an ephemeral child Redis are stopped. Existing listeners
cause refusal, not termination. No CPU quota is enforced. Each layout has 24
ordinary DB connections per engine family configured (8 web + 16 execution),
no overflow, and four execution trace slots; web has an additional trace pool.
Env settings, external model latency and host resource limits must be held fixed.
Warmup runs real tool tasks until every execution process has evidence of a
completed tool task, at most three bursts plus a paced fourth round. Warm samples
are excluded; coverage failures stop the case and remain on disk. No fair task
assignment is assumed. Measurement always uses zero background submission spacing.

report.json contains creation / first model-adapter entry / persisted completion
latencies, per-worker distribution, failed/unknown requests and missing evidence.
The model clock is not network send time or llm_call_start. Timestamps are from
the same host; this launcher is NOT a distributed clock-synchronized benchmark.
Continuous full-load comparisons match probe ordinals only when all background
tasks were RUNNING in every case. This is not identical wall-clock exposure.
Repeat runs, retain failures, and report sample counts; no single run proves an
SLO, cold-start fix or fair load balancing. Loop/GIL/CPU diagnosis remains a
separate OTel/profiler concern; this script does not enable worker loop sampling.

Tests: PYTHONPATH=src:. pytest tests/test_shared_worker_benchmark.py -q
The tests use fixtures, not a real model, and are not capacity evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from scripts.shared_worker_report import compare_cases, read_case
from scripts.shared_worker_workload import PROMPT

HARNESS = Path(__file__).resolve().parents[1]


def rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)


def assert_free(port):
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is occupied; refusing to touch it")


def wait_for(check, processes, description, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError(f"{description}: application host exited")
        try:
            if check():
                return
        except (OSError, httpx.HTTPError):
            pass
        time.sleep(0.25)
    raise TimeoutError(description)


def validate_prewarm(log_rows, background_tasks=8):
    """Require the full real workload, not merely the subset returned by SQL.

    The load client can emit a safety stop and still exit zero. Checking only
    its exit status (or ``all`` on an empty task list) would accept a failed
    warmup. Only completed tool-bearing background tasks count for coverage.
    """
    if any(row.get("type") == "safety_stop" for row in log_rows):
        raise RuntimeError("Prewarm client reported a safety stop")

    def one(kind):
        matches = [row for row in log_rows if row.get("type") == kind]
        if len(matches) != 1:
            raise RuntimeError(f"Prewarm requires exactly one {kind} record")
        return matches[0]

    def completed(submissions, result, count):
        if len(submissions) != count or any(
            row.get("http") != 202 or not row.get("id") for row in submissions
        ):
            raise RuntimeError("Prewarm requests were rejected or missing")
        ids = {row["id"] for row in submissions}
        tasks = result.get("tasks", [])
        if (
            len(ids) != count
            or len(tasks) != count
            or {row.get("id") for row in tasks} != ids
            or result.get("timeout")
            or any(row.get("status") != "completed" for row in tasks)
        ):
            raise RuntimeError("Prewarm tasks did not all complete")
        return ids

    warm, admission, done = one("warm"), one("admission"), one("completed")
    first_ids = completed(warm.get("rows", []), warm, 1)
    background, probes = admission.get("background", []), admission.get("probes", [])
    if (
        admission.get("background_n") != background_tasks
        or done.get("background_n") != background_tasks
        or len(background) != background_tasks
        or len(probes) != 10
    ):
        raise RuntimeError(
            "Prewarm workload does not match the configured tool tasks and ten probes"
        )
    all_ids = completed(background + probes, done, background_tasks + 10)
    if first_ids & all_ids:
        raise RuntimeError("Prewarm task IDs were reused")
    background_ids = {row["id"] for row in background}
    tool_ids = {
        row["id"]
        for row in done.get("events", [])
        if row.get("type") == "tool_execution_end" and row.get("count", 0) > 0
    }
    if not background_ids <= tool_ids:
        raise RuntimeError("Prewarm background tasks did not exercise the tool path")
    return background_ids


def warmed_execution_pids(execution_hosts, output, completed_ids):
    """Read per-process evidence; a warmed web/QA request is not sufficient."""
    warmed = set()
    for host_info in execution_hosts:
        started = {
            int(value)
            for value in re.findall(
                r"Background task execution started for task (\d+)",
                (output / (host_info["name"] + ".log")).read_text(),
            )
        }
        if started & completed_ids:
            warmed.add(host_info["pid"])
    return warmed


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", type=Path, default=HARNESS)
    parser.add_argument("--server-env", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", type=positive, required=True)
    parser.add_argument("--case", choices=("shared-1", "shared-4"))
    parser.add_argument("--background", type=positive, default=200)
    parser.add_argument("--probes", type=positive, default=60)
    parser.add_argument("--probe-interval", type=positive, default=2)
    parser.add_argument("--task-timeout", type=positive, default=600)
    parser.add_argument("--port", type=positive, default=8002)
    parser.add_argument("--redis-port", type=positive, default=6385)
    parser.add_argument(
        "--prewarm", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--confirm-test-environment", action="store_true")
    return parser.parse_args(argv)


def read_environment(args):
    from dotenv import dotenv_values, load_dotenv

    if not args.confirm_test_environment:
        raise ValueError("--confirm-test-environment is required")
    if "PYTHON_DOTENV_DISABLED" not in inspect.getsource(load_dotenv):
        raise RuntimeError("Install python-dotenv >= 1.2 in the benchmark interpreter")
    if not (args.repo / "src/xagent/web/worker.py").is_file():
        raise ValueError("--repo needs the shared worker entry point from #2411")
    if (
        not 0 < args.port < 65536
        or not 0 < args.redis_port < 65536
        or args.port == args.redis_port
    ):
        raise ValueError("Choose distinct valid API and Redis ports")
    required = (
        "DATABASE_URL",
        "ENCRYPTION_KEY",
        "XAGENT_JWT_SECRET",
        "XAGENT_STORAGE_ROOT",
        "XAGENT_BENCH_USERNAME",
        "XAGENT_BENCH_PASSWORD",
    )
    config = {
        key: value
        for key, value in dotenv_values(args.server_env, interpolate=False).items()
        if value is not None
    }
    if any(not config.get(key) for key in required):
        raise ValueError("The private env file is missing required benchmark settings")
    if not config["DATABASE_URL"].startswith(("postgresql://", "postgres://")):
        raise ValueError("Benchmark requires an explicit PostgreSQL database")
    if not Path(config["XAGENT_STORAGE_ROOT"]).is_absolute():
        raise ValueError("XAGENT_STORAGE_ROOT must be absolute")
    if config.get("XAGENT_TASK_WORKER_MAX_CONCURRENT_TASKS"):
        raise ValueError("Remove the reverted worker capacity setting")
    # Do not inherit arbitrary shell secrets, PYTHONPATH or a developer .env.
    result = {
        key: os.environ[key]
        for key in ("HOME", "USER", "PATH", "TMPDIR", "LANG")
        if key in os.environ
    }
    result.update(config)
    result["PYTHON_DOTENV_DISABLED"] = "1"
    result["PYTHONPATH"] = os.pathsep.join((str(args.repo / "src"), str(HARNESS)))
    return result


def host_environment(base, args, role, workers):
    result = dict(base)
    result.pop("XAGENT_BENCH_USERNAME", None)
    result.pop("XAGENT_BENCH_PASSWORD", None)
    result.update(
        XAGENT_SHARED_TASK_EXECUTION_ENABLED="true",
        XAGENT_TASK_EXECUTION_ROLE=role,
        XAGENT_REDIS_URL=f"redis://127.0.0.1:{args.redis_port}/0",
        XAGENT_CHANNEL_INGRESS_ENABLED="false",
        XAGENT_DB_POOL_SIZE=str(8 if role == "web" else 16 // workers),
        XAGENT_DB_MAX_OVERFLOW="0",
        XAGENT_TRACE_DB_MAX_INFLIGHT=str(4 if role == "web" else 4 // workers),
        XAGENT_ASYNC_TRACE_DB_ENABLED="true",
    )
    return result


def workload_command(args, background, probes, spacing=0):
    return [
        sys.executable,
        "-m",
        "scripts.shared_worker_workload",
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--model-id",
        str(args.model_id),
        "--background",
        str(background),
        "--probes",
        str(probes),
        "--probe-interval",
        str(args.probe_interval),
        "--task-timeout",
        str(args.task_timeout),
        "--spacing",
        str(spacing),
    ]


def stop_owned(processes, forced):
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    for process in processes:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            forced.append(process.pid)
            process.kill()
            process.wait(timeout=10)


def check_idle_database(base):
    from contextlib import closing

    import psycopg2

    with closing(
        psycopg2.connect(
            base["DATABASE_URL"],
            connect_timeout=5,
            options="-c default_transaction_read_only=on -c statement_timeout=5000",
        )
    ) as db:
        with db.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM tasks WHERE status::text NOT IN "
                "('COMPLETED','FAILED','CANCELLED','PAUSED')"
            )
            if cursor.fetchone()[0]:
                raise RuntimeError(
                    "Database has unsettled tasks; inspect them before benchmarking"
                )


def run_case(args, base, workers):
    assert_free(args.port)
    assert_free(args.redis_port)
    check_idle_database(base)
    output = args.output / f"shared-{workers}"
    output.mkdir(mode=0o700)
    hosts, clients, infrastructure, files = [], [], [], []
    manifest = {
        "workers": workers,
        "hosts": [],
        "background_tasks": args.background,
        "continuous_probes": args.probes,
        "probe_interval_s": args.probe_interval,
        "model_id": args.model_id,
        "prewarm_requested": args.prewarm,
        "measured_submit_spacing_s": 0,
        "complete": False,
        "forced_shutdown_pids": [],
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=args.repo, text=True
        ),
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=args.repo)
        ).hexdigest(),
        "harness_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((HARNESS / "scripts").glob("shared_worker_*.py"))
        },
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "logical_cpus": os.cpu_count(),
        "cpu_quota": None,
        "ordinary_pool_budget_per_engine_family": 24,
        "execution_trace_budget": 4,
        "web_trace_budget": 4,
        "db_overflow": 0,
        "task_timeout_s": args.task_timeout,
    }

    def launch(command, name, env, group):
        log = (output / (name + ".log")).open("x")
        files.append(log)
        process = subprocess.Popen(
            command, env=env, cwd=output, stdout=log, stderr=subprocess.STDOUT
        )
        group.append(process)
        return process

    def client(name, background, probes, spacing=0):
        path = output / (name + ".jsonl")
        log = path.open("x")
        files.append(log)
        process = subprocess.Popen(
            workload_command(args, background, probes, spacing),
            env=base,
            cwd=output,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        clients.append(process)
        wait_for(
            lambda: process.poll() is not None,
            hosts + infrastructure,
            name,
            timeout=2 * args.task_timeout + probes * args.probe_interval + 240,
        )
        records = rows(path)
        if process.returncode:
            raise RuntimeError(f"{name} failed; inspect retained request records")
        completed = validate_prewarm(records, background)
        # Tool-end events alone do not prove success or the intended call count.
        done = next(row for row in records if row["type"] == "completed")
        events = {
            row["id"]: row
            for row in done["events"]
            if row["type"] == "tool_execution_end"
        }
        if any(
            events[key]["count"] != 10 or events[key].get("success_count") != 10
            for key in completed
        ):
            raise RuntimeError("Tool workload deviated from ten successful calls")
        probe_records = [row for row in records if row.get("type") == "probes"]
        if len(probe_records) != 1:
            raise RuntimeError("Missing continuous probe completion record")
        validate_probes(probe_records[0], probes)
        return completed

    try:
        redis = launch(
            [
                shutil.which("redis-server") or "redis-server",
                "--bind",
                "127.0.0.1",
                "--port",
                str(args.redis_port),
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                str(output),
            ],
            "redis",
            base,
            infrastructure,
        )

        def redis_ready():
            with socket.create_connection(
                ("127.0.0.1", args.redis_port), timeout=1
            ) as connection:
                connection.sendall(b"*1\r\n$4\r\nPING\r\n")
                return connection.recv(32).startswith(b"+PONG")

        wait_for(redis_ready, [redis], "Redis")
        for name, role in [("web", "web")] + [
            (f"worker-{index + 1}", "worker") for index in range(workers)
        ]:
            command = [
                sys.executable,
                "-m",
                "scripts.shared_worker_observer",
                "xagent.web.worker" if role == "worker" else "xagent.web",
                str(output),
            ]
            if role == "web":
                command.extend(["--host", "127.0.0.1", "--port", str(args.port)])
            process = launch(
                command, name, host_environment(base, args, role, workers), hosts
            )
            manifest["hosts"].append({"name": name, "pid": process.pid, "role": role})
            if role == "worker":
                wait_for(
                    lambda: "Shared task worker ready"
                    in (output / (name + ".log")).read_text(),
                    hosts,
                    name,
                )
            else:
                wait_for(
                    lambda: httpx.get(
                        f"http://127.0.0.1:{args.port}/openapi.json", timeout=2
                    ).status_code
                    == 200,
                    hosts,
                    name,
                )
        execution_hosts = [
            host for host in manifest["hosts"] if host["role"] == "worker"
        ]
        if args.prewarm:
            warm = manifest["prewarm"] = {
                "expected_pids": sorted(h["pid"] for h in execution_hosts),
                "warmed_pids": [],
                "rounds": [],
                "complete": False,
                "started_at": time.time(),
            }
            completed_ids = set()
            try:
                for index in range(4):
                    spacing = 0.2 if index == 3 else 0
                    name = f"prewarm-{index + 1}"
                    detail = {
                        "file": name + ".jsonl",
                        "started_at": time.time(),
                        "submit_spacing_s": spacing,
                    }
                    warm["rounds"].append(detail)
                    completed_ids.update(client(name, 16, 0, spacing))
                    detail["finished_at"] = time.time()
                    warm["warmed_pids"] = sorted(
                        warmed_execution_pids(execution_hosts, output, completed_ids)
                    )
                    if warm["warmed_pids"] == warm["expected_pids"]:
                        warm["complete"] = True
                        break
                if not warm["complete"]:
                    raise RuntimeError(
                        "Not every execution host warmed; refusing measurement"
                    )
            finally:
                warm["finished_at"] = time.time()
        manifest["workload_started_at"] = time.time()
        client("workload", args.background, args.probes)
        manifest["complete"] = True
    except BaseException as exc:
        manifest["error"] = type(exc).__name__
        raise
    finally:
        # Stop only our children. Clients are stopped first, then API/execution,
        # with Redis last so graceful worker shutdown retains its event bridge.
        stop_owned(clients, manifest["forced_shutdown_pids"])
        stop_owned(hosts, manifest["forced_shutdown_pids"])
        stop_owned(infrastructure, manifest["forced_shutdown_pids"])
        manifest["client_exit_codes"] = [p.returncode for p in clients]
        manifest["host_exit_codes"] = [p.returncode for p in hosts]
        if manifest["forced_shutdown_pids"]:
            manifest["complete"] = False
        for handle in files:
            handle.close()
        save(output / "manifest.json", manifest)
    report = read_case(output)
    save(output / "report.json", report)
    return report


def validate_probes(record, count):
    submissions, tasks = record.get("rows", []), record.get("tasks", [])
    ids = {row.get("id") for row in submissions}
    if (
        len(submissions) != count
        or len(ids) != count
        or None in ids
        or any(row.get("http") != 202 for row in submissions)
        or len(tasks) != count
        or {row.get("id") for row in tasks} != ids
        or record.get("timeout")
        or any(row.get("status") != "completed" for row in tasks)
        or {row.get("ordinal") for row in submissions} != set(range(count))
    ):
        raise RuntimeError("Continuous probes were rejected, missing or incomplete")


def main():
    args = parse_args()
    args.repo, args.output = args.repo.resolve(), args.output.resolve()
    base = read_environment(args)
    if not shutil.which("redis-server"):
        raise SystemExit("redis-server must be on PATH")
    args.output.mkdir(mode=0o700)  # Refuse overwrite; parent must already exist.
    cases = []
    try:
        for workers in (1, 4):
            if args.case and args.case != f"shared-{workers}":
                continue
            report = run_case(args, base, workers)
            cases.append(report)
            if not report["valid"]:
                raise RuntimeError("Incomplete measurement; inspect the case report")
    except BaseException as exc:
        save(
            args.output / "failure.json",
            {"error": type(exc).__name__, "completed_cases": len(cases)},
        )
        raise SystemExit(
            "Benchmark stopped; artifacts retained. Inspect tasks before rerunning."
        ) from None
    save(args.output / "comparison.json", compare_cases(cases))
    print(f"Saved {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
