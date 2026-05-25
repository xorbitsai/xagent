#!/usr/bin/env python3
"""Run the local web backend together with Celery background-job workers."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values, load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_CELERY_BROKER_URL = "redis://localhost:6379/1"
DEFAULT_CELERY_RESULT_BACKEND = "redis://localhost:6379/2"
FRONTEND_ENV_PATH = ROOT_DIR / "frontend" / ".env.local"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start xagent backend, Celery worker, and scheduler for local development."
        )
    )
    parser.add_argument(
        "--no-scheduler",
        "--no-beat",
        action="store_true",
        help="Do not start Celery beat.",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Do not pass --reload to the FastAPI backend.",
    )
    args, backend_args = parser.parse_known_args()
    args.backend_args = backend_args
    return args


def _safe_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlsplit(url)
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _configure_env() -> dict[str, str]:
    load_dotenv(ROOT_DIR / ".env")
    env = os.environ.copy()

    src_path = str(ROOT_DIR / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else src_path
    )

    env.setdefault("XAGENT_REDIS_URL", DEFAULT_REDIS_URL)
    env["XAGENT_CELERY_ENABLED"] = "true"
    env.setdefault("XAGENT_CELERY_BROKER_URL", DEFAULT_CELERY_BROKER_URL)
    env.setdefault("XAGENT_CELERY_RESULT_BACKEND", DEFAULT_CELERY_RESULT_BACKEND)
    return env


def _has_backend_port_arg(backend_args: Sequence[str]) -> bool:
    return any(arg == "--port" or arg.startswith("--port=") for arg in backend_args)


def _frontend_api_port() -> int | None:
    if not FRONTEND_ENV_PATH.exists():
        return None

    api_url = dotenv_values(FRONTEND_ENV_PATH).get("NEXT_PUBLIC_API_URL")
    if not api_url:
        return None

    parsed = urlsplit(api_url)
    if parsed.scheme not in {"http", "https"}:
        return None

    try:
        return parsed.port
    except ValueError:
        return None


def _check_redis(env: dict[str, str]) -> None:
    try:
        import redis
    except ImportError as exc:
        raise SystemExit("redis package is not installed in this environment") from exc

    urls = [
        ("cache", env.get("XAGENT_REDIS_URL")),
        ("celery broker", env.get("XAGENT_CELERY_BROKER_URL")),
    ]
    result_backend = env.get("XAGENT_CELERY_RESULT_BACKEND")
    if result_backend:
        urls.append(("celery result backend", result_backend))

    for label, url in urls:
        if not url:
            raise SystemExit(f"{label} Redis URL is not configured")
        try:
            client = redis.Redis.from_url(
                url,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
        except Exception as exc:  # noqa: BLE001
            safe_url = _safe_url(url)
            raise SystemExit(
                f"Cannot connect to {label} Redis at {safe_url}: {exc}"
            ) from exc


def _start(
    *,
    name: str,
    command: Sequence[str],
    env: dict[str, str],
) -> subprocess.Popen[bytes]:
    print(f"Starting {name}: {' '.join(command)}", flush=True)
    return subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        env=env,
        start_new_session=os.name != "nt",
    )


def _send_signal(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    try:
        if os.name != "nt":
            os.killpg(process.pid, sig)
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        return


def _terminate(processes: list[tuple[str, subprocess.Popen[bytes]]]) -> None:
    for name, process in reversed(processes):
        if process.poll() is None:
            print(f"Stopping {name}", flush=True)
            _send_signal(process, signal.SIGTERM)

    deadline = time.time() + 10
    for name, process in reversed(processes):
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.2)
        if process.poll() is None:
            print(f"Killing {name}", flush=True)
            _send_signal(process, signal.SIGKILL)


def main() -> int:
    args = _parse_args()
    env = _configure_env()
    _check_redis(env)

    backend_args = list(args.backend_args)
    if backend_args and backend_args[0] == "--":
        backend_args = backend_args[1:]
    if not args.no_reload and "--reload" not in backend_args:
        backend_args.insert(0, "--reload")
    if not _has_backend_port_arg(backend_args):
        frontend_port = _frontend_api_port()
        if frontend_port is not None:
            print(
                "Using backend port "
                f"{frontend_port} from {FRONTEND_ENV_PATH.relative_to(ROOT_DIR)}",
                flush=True,
            )
            backend_args.extend(["--port", str(frontend_port)])

    processes: list[tuple[str, subprocess.Popen[bytes]]] = []

    def handle_signal(signum: int, _frame: object) -> None:
        print(f"Received signal {signum}", flush=True)
        _terminate(processes)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        processes.append(
            (
                "celery worker",
                _start(
                    name="celery worker",
                    command=[
                        sys.executable,
                        "-m",
                        "celery",
                        "-A",
                        "xagent.web.jobs.celery_app:celery_app",
                        "worker",
                        "-Q",
                        "kb,triggers,default",
                        "--loglevel=INFO",
                        "--pool=solo",
                    ],
                    env=env,
                ),
            )
        )
        if not args.no_scheduler:
            processes.append(
                (
                    "celery beat",
                    _start(
                        name="celery beat",
                        command=[
                            sys.executable,
                            "-m",
                            "celery",
                            "-A",
                            "xagent.web.jobs.celery_app:celery_app",
                            "beat",
                            "--loglevel=INFO",
                        ],
                        env=env,
                    ),
                )
            )

        processes.append(
            (
                "backend",
                _start(
                    name="backend",
                    command=[
                        sys.executable,
                        "-m",
                        "xagent.web.__main__",
                        *backend_args,
                    ],
                    env=env,
                ),
            )
        )

        while True:
            for name, process in processes:
                return_code = process.poll()
                if return_code is not None:
                    print(
                        f"{name} exited with status {return_code}",
                        flush=True,
                    )
                    return return_code
            time.sleep(0.5)
    finally:
        _terminate(processes)


if __name__ == "__main__":
    raise SystemExit(main())
