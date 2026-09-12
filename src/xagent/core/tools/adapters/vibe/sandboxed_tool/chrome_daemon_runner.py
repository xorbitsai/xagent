"""Sandbox-side controller for one pinned Chrome DevTools daemon.

This module intentionally uses only the Python standard library.  It is
executed *inside* a dedicated sandbox; the host-side client never starts an
MCP or browser process directly.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

CHROME_DEVTOOLS_PACKAGE = "chrome-devtools-mcp@1.6.0"
CHROME_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
DEFAULT_NPM_CACHE = "/opt/npm-cache"
_CLI_TIMEOUT_SECONDS = 30.0
_SOCKET_TIMEOUT_SECONDS = 60.0
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_kill_process = os.kill


class ChromeDaemonRunnerError(RuntimeError):
    """The sandbox-side daemon contract was not satisfied."""


def _decode_json(value: str, *, expected_type: type[Any]) -> Any:
    try:
        decoded = json.loads(base64.b64decode(value, validate=True).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChromeDaemonRunnerError("invalid base64 JSON payload") from exc
    if not isinstance(decoded, expected_type):
        raise ChromeDaemonRunnerError(
            f"payload must decode to {expected_type.__name__}"
        )
    return decoded


def _private_dir(path: Path) -> None:
    if path.is_symlink():
        raise ChromeDaemonRunnerError(f"private path must not be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ChromeDaemonRunnerError(
            f"private path has unsafe ownership or mode: {path}"
        )


def _session_environment(session_id: str) -> tuple[dict[str, str], Path, Path]:
    runtime_root = Path("/tmp/xagent-chrome-runtime")
    profile_root = Path(f"/tmp/xagent-chrome-profile-{session_id}")
    _private_dir(runtime_root)
    _private_dir(profile_root)

    env = dict(os.environ)
    env["XDG_RUNTIME_DIR"] = str(runtime_root)
    env["TMPDIR"] = str(profile_root)
    env["NPM_CONFIG_CACHE"] = env.get("NPM_CONFIG_CACHE") or DEFAULT_NPM_CACHE
    env["CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS"] = "1"
    return env, runtime_root, profile_root


def _runtime_paths(session_id: str, runtime_root: Path) -> tuple[Path, Path]:
    daemon_home = runtime_root / f"chrome-devtools-mcp-{session_id}"
    return daemon_home / "server.sock", daemon_home / "daemon.pid"


def _read_pid(pid_file: Path) -> int | None:
    try:
        raw = pid_file.read_text(encoding="utf-8").strip()
        pid = int(raw)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return pid if pid > 1 else None


def _pid_is_expected_daemon(pid: int) -> bool:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"chrome-devtools-mcp" in cmdline and b"daemon.js" in cmdline


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.05)
    return False


def _signal_process(pid: int, signal_number: int) -> bool:
    try:
        _kill_process(pid, signal_number)
    except ProcessLookupError:
        return False
    return True


def _terminate_expected_daemon(pid_file: Path) -> None:
    pid = _read_pid(pid_file)
    if pid is None:
        return
    if not _pid_is_expected_daemon(pid):
        raise ChromeDaemonRunnerError("refusing to signal an unverified daemon pid")
    if not _signal_process(pid, signal.SIGTERM):
        return
    if not _wait_for_exit(pid, 3.0):
        if not _signal_process(pid, signal.SIGKILL):
            return
        if not _wait_for_exit(pid, 2.0):
            raise ChromeDaemonRunnerError("daemon did not exit after SIGKILL")


def _socket_request(socket_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\0"
    received = bytearray()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(_SOCKET_TIMEOUT_SECONDS)
        client.connect(str(socket_path))
        client.sendall(payload)
        while True:
            chunk = client.recv(65536)
            if not chunk:
                raise ChromeDaemonRunnerError("daemon socket closed without a response")
            nul = chunk.find(b"\0")
            if nul >= 0:
                if len(received) + nul > _MAX_RESPONSE_BYTES:
                    raise ChromeDaemonRunnerError("daemon response exceeded size limit")
                received.extend(chunk[:nul])
                break
            if len(received) + len(chunk) > _MAX_RESPONSE_BYTES:
                raise ChromeDaemonRunnerError("daemon response exceeded size limit")
            received.extend(chunk)
    try:
        response = json.loads(received.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChromeDaemonRunnerError("daemon returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise ChromeDaemonRunnerError("daemon returned a non-object response")
    return response


def _status(socket_path: Path) -> dict[str, Any] | None:
    if not socket_path.exists():
        return None
    try:
        response = _socket_request(socket_path, {"method": "status"})
    except (OSError, ChromeDaemonRunnerError):
        return None
    if response.get("success") is not True:
        return None
    try:
        status = json.loads(response["result"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    return status if isinstance(status, dict) else None


def _validate_server_args(server_args: list[Any]) -> list[str]:
    if not all(isinstance(value, str) and value for value in server_args):
        raise ChromeDaemonRunnerError("server args must be non-empty strings")
    blocked = {
        "--session-id",
        "--sessionId",
        "--user-data-dir",
        "--userDataDir",
        "--auto-connect",
        "--autoConnect",
        "--browser-url",
        "--browserUrl",
        "--executable-path",
        "--executablePath",
        "--ws-endpoint",
        "--wsEndpoint",
        "--via-cli",
        "--viaCli",
    }
    if any(arg.split("=", 1)[0] in blocked for arg in server_args):
        raise ChromeDaemonRunnerError("server args override a managed isolation option")
    nested_blocked = {"--profile-directory", "--user-data-dir"}
    for index, arg in enumerate(server_args):
        nested: str | None = None
        if arg.startswith("--chrome-arg="):
            nested = arg.split("=", 1)[1]
        elif arg == "--chrome-arg" and index + 1 < len(server_args):
            nested = server_args[index + 1]
        if nested is not None and nested.split("=", 1)[0] in nested_blocked:
            raise ChromeDaemonRunnerError(
                "server args override the managed Chrome profile"
            )
    if "--isolated" not in server_args or "--headless" not in server_args:
        raise ChromeDaemonRunnerError(
            "Chrome daemon requires --isolated and --headless"
        )
    return list(server_args)


def _serialized_server_args(server_args: list[str]) -> list[str]:
    serialized: list[str] = []
    for arg in server_args:
        if arg.startswith("--chrome-arg="):
            serialized.extend(("--chrome-arg", arg.split("=", 1)[1]))
        else:
            serialized.append(arg)
    return serialized


def _status_matches_managed_daemon(
    status: dict[str, Any], server_args: list[str]
) -> bool:
    args = status.get("args")
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        return False
    expected = Counter(
        [
            *_serialized_server_args(server_args),
            "--viaCli",
            "--experimentalStructuredContent",
        ]
    )
    actual = Counter(args)
    forbidden_direct = {
        "--auto-connect",
        "--autoConnect",
        "--browser-url",
        "--browserUrl",
        "--executable-path",
        "--executablePath",
        "--user-data-dir",
        "--userDataDir",
        "--ws-endpoint",
        "--wsEndpoint",
    }
    if any(arg.split("=", 1)[0] in forbidden_direct for arg in args):
        return False
    for index, arg in enumerate(args):
        nested = None
        if arg.startswith("--chrome-arg="):
            nested = arg.split("=", 1)[1]
        elif arg == "--chrome-arg" and index + 1 < len(args):
            nested = args[index + 1]
        if nested is not None and nested.split("=", 1)[0] in {
            "--profile-directory",
            "--user-data-dir",
        }:
            return False
    return status.get("version") == CHROME_DEVTOOLS_PACKAGE.rsplit("@", 1)[1] and all(
        actual[arg] >= count for arg, count in expected.items()
    )


def _start(session_id: str, server_args: list[Any]) -> dict[str, Any]:
    validated_args = _validate_server_args(server_args)
    env, runtime_root, _profile_root = _session_environment(session_id)
    socket_path, pid_file = _runtime_paths(session_id, runtime_root)
    status = _status(socket_path)
    if status is not None and _status_matches_managed_daemon(status, validated_args):
        return status
    if status is not None:
        raise ChromeDaemonRunnerError(
            "existing daemon does not match the pinned launch contract"
        )

    if pid_file.exists():
        pid = _read_pid(pid_file)
        if pid is not None and _pid_is_expected_daemon(pid):
            _terminate_expected_daemon(pid_file)
        elif pid is not None and _pid_is_alive(pid):
            raise ChromeDaemonRunnerError(
                "refusing to replace an unverified live daemon pid"
            )
        pid_file.unlink(missing_ok=True)
    socket_path.unlink(missing_ok=True)

    args = [
        "npx",
        "-y",
        "--prefer-offline",
        "--package",
        CHROME_DEVTOOLS_PACKAGE,
        "chrome-devtools",
        "start",
        "--sessionId",
        session_id,
        *validated_args,
    ]
    completed = subprocess.run(
        args,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=_CLI_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise ChromeDaemonRunnerError(
            f"daemon start failed with exit code {completed.returncode}"
        )
    status = _status(socket_path)
    if status is None:
        raise ChromeDaemonRunnerError("daemon failed its post-start health check")
    if not _status_matches_managed_daemon(status, validated_args):
        raise ChromeDaemonRunnerError("daemon launch does not match the pinned spec")
    return status


def _tool(session_id: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    _env, runtime_root, _profile_root = _session_environment(session_id)
    socket_path, _pid_file = _runtime_paths(session_id, runtime_root)
    if _status(socket_path) is None:
        raise ChromeDaemonRunnerError("daemon is not healthy")
    response = _socket_request(
        socket_path,
        {"method": "invoke_tool", "tool": tool, "args": arguments},
    )
    if response.get("success") is not True:
        raise ChromeDaemonRunnerError("daemon rejected the tool call")
    try:
        result = json.loads(response["result"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ChromeDaemonRunnerError("daemon returned an invalid tool result") from exc
    if not isinstance(result, dict):
        raise ChromeDaemonRunnerError("daemon returned a non-object tool result")
    return result


def _stop(session_id: str) -> dict[str, bool]:
    _env, runtime_root, profile_root = _session_environment(session_id)
    socket_path, pid_file = _runtime_paths(session_id, runtime_root)
    try:
        if _status(socket_path) is not None:
            try:
                _socket_request(socket_path, {"method": "stop"})
            except (OSError, ChromeDaemonRunnerError):
                pass
        pid = _read_pid(pid_file)
        if pid is not None and not _wait_for_exit(pid, 10.0):
            _terminate_expected_daemon(pid_file)
    finally:
        try:
            shutil.rmtree(socket_path.parent)
        except OSError:
            pass
        try:
            shutil.rmtree(profile_root)
        except OSError:
            pass
    return {"stopped": True}


def _write_result(path: Path, result: dict[str, Any]) -> None:
    if not path.is_absolute() or path.parent != Path("/tmp"):
        raise ChromeDaemonRunnerError("result file must be directly under /tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    encoded = json.dumps(result, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        raise ChromeDaemonRunnerError("result exceeded size limit")
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("start", "status", "stop", "tool"))
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--server-args-b64")
    parser.add_argument("--tool")
    parser.add_argument("--arguments-b64")
    args = parser.parse_args()

    if CHROME_SESSION_ID_RE.fullmatch(args.session_id) is None:
        raise ChromeDaemonRunnerError("session id must be 32 lowercase hex characters")

    if args.operation == "start":
        if args.server_args_b64 is None:
            raise ChromeDaemonRunnerError("start requires server args")
        result = _start(
            args.session_id,
            _decode_json(args.server_args_b64, expected_type=list),
        )
    elif args.operation == "status":
        _env, runtime_root, _profile_root = _session_environment(args.session_id)
        socket_path, _pid_file = _runtime_paths(args.session_id, runtime_root)
        result = {"running": (status := _status(socket_path)) is not None}
        if status is not None:
            result["status"] = status
    elif args.operation == "tool":
        if args.tool is None or args.arguments_b64 is None:
            raise ChromeDaemonRunnerError("tool requires a name and arguments")
        arguments = _decode_json(args.arguments_b64, expected_type=dict)
        result = _tool(args.session_id, args.tool, arguments)
    else:
        result = _stop(args.session_id)

    _write_result(Path(args.result_file), result)


if __name__ == "__main__":
    main()
