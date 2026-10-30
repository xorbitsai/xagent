"""Post-deploy startability check for every catalog connector.

Usage: ``python -m xagent.web.connector_readiness [--exclude APP_ID]...
[--total-timeout SECONDS]``, run in the backend container with the running
service's environment, the same way as ``oauth_readiness``::

    docker compose exec -T backend /opt/xagent/deploy/entrypoint.sh \
        python -m xagent.web.connector_readiness

It reads every ``public_mcp_apps`` row of the deployment's database, hidden
rows included, and checks by ``launch_config`` shape whether the connector
can start here, without any credential:

* A stdio row (``command``) is spawned with a placeholder value for every
  ``env_mapping`` and ``required_env`` key, then asked for MCP ``initialize``
  and ``tools/list``; at least one tool must come back. The in-repo servers'
  tool lists are static, and the placeholder never leaves the process: the
  check stops after ``tools/list``.
* An ``npx``/``uvx`` row is reported ``SKIP sandboxed`` when the deployment
  runs a sandbox (``SANDBOX_ENABLED``), because tasks run those in the sandbox
  image, not the backend image. With the sandbox off it is spawned like any stdio row,
  with npm and uv forced offline so a cache miss fails before the handshake
  instead of reaching a registry.
* A remote row (``url``) gets one unauthenticated ``GET`` through the
  runtime's MCP HTTP client, so the probe sees the same proxy and egress
  policy a task does. Any HTTP response passes; a DNS, TLS, egress-policy or
  connect failure fails. No OAuth flow is started.

One line per connector, ``PASS``, ``SKIP sandboxed``, ``SKIP websocket`` or
``FAIL <class>`` with the class one of ``spawn`` (command not found, or a row
with neither a command nor a url), ``missing`` (the child exited before the
handshake: a Python import error, or an npm package absent from the offline
cache), ``handshake`` (no usable ``initialize``/``tools/list`` reply within the
timeout, or an error reply), ``no-tools`` and ``unreachable``. A failure line
carries the last line of the child's stderr or the exception, never the
environment. A database that cannot be read prints ``FAIL database: <class>``
once, and a run that checks no row prints ``FAIL catalog: no public_mcp_apps
rows to check``: neither is a green gate. The exit code is 1 when any
connector fails or either of those lines is printed.

Each connector is bounded by ``XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS`` (the
same bound task startup uses) and the whole run by ``--total-timeout``, which
defaults to the connector count times that value.
``XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS=0`` disables both the per-connector and
the default total bound; ``--total-timeout 0`` disables the total bound.

The child's environment is the placeholders, the offline switches and
``PATH``, plus the SDK's inherited allowlist (``HOME``, ``USER``, ...). A task
also forwards ``static_env`` entries, the proxy and CA-bundle variables and the
file allowlists (``web/tools/config.py``); building a connector's environment
outside a task is #3261's work. A connector that needs one of those before it
answers ``tools/list`` would fail here and not in a task; no registry row does.
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, TextIO, cast
from urllib.parse import urlsplit

import anyio
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED

from ..core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper import (
    should_sandbox_mcp_connection,
)
from ..core.tools.core.mcp.http import create_mcp_http_client

if TYPE_CHECKING:
    from ..core.tools.core.mcp.sessions import Connection

Status = Literal["PASS", "SKIP", "FAIL"]
Reason = Literal[
    "sandboxed", "websocket", "spawn", "missing", "handshake", "no-tools", "unreachable"
]
Origin = Literal["builtin", "custom"]

# Obviously fake, and the only value the spawned connector ever sees for a
# credential key. The stdio check stops after tools/list, so it never reaches
# a vendor.
PLACEHOLDER = "readiness-placeholder"
# Forces a cache miss to fail instead of reaching the registry; NPM_CONFIG_*
# is how Dockerfile.sandbox verifies its own warm-up.
OFFLINE_ENV = {"NPM_CONFIG_OFFLINE": "true", "UV_OFFLINE": "1"}
_STDERR_TAIL_LINES = 5
_DETAIL_MAX_CHARS = 300
# npm ends every failure with this pointer; the error itself is the line above.
_NPM_LOG_POINTER = "A complete log of this run can be found in"


@dataclass(frozen=True)
class ConnectorReadiness:
    """One check's result; ``app_id``, ``origin`` and ``hidden`` are filled in
    when the result is tied to a catalog row."""

    status: Status
    reason: Reason | None
    detail: str
    stderr_tail: str = ""
    app_id: str = ""
    origin: Origin = "custom"
    hidden: bool = False

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


SKIPPED_SANDBOXED = ConnectorReadiness(
    "SKIP", "sandboxed", "tasks run npx/uvx rows in the sandbox image, not this one"
)
NO_LAUNCH = ConnectorReadiness(
    "FAIL", "spawn", "launch_config has neither a command nor a url"
)


def sandbox_enabled_from_env() -> bool:
    """Whether a task attaches a sandbox, read the way
    ``sandbox_manager._create_sandbox_service`` reads it."""
    return os.getenv("SANDBOX_ENABLED", "false").lower() == "true"


def is_sandbox_command(command: str) -> bool:
    """Whether a task routes this stdio command into the sandbox (npx, uvx)."""
    connection = cast("Connection", {"transport": "stdio", "command": command})
    return should_sandbox_mcp_connection(connection)


def placeholder_env(launch_config: Mapping[str, Any]) -> dict[str, str]:
    """A placeholder for every key the connector expects a credential in."""
    keys: list[str] = []
    env_mapping = launch_config.get("env_mapping")
    if isinstance(env_mapping, Mapping):
        keys.extend(str(key) for key in env_mapping)
    required_env = launch_config.get("required_env")
    if isinstance(required_env, (list, tuple)):
        keys.extend(str(key) for key in required_env)
    return {key: PLACEHOLDER for key in keys}


def _first_line(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:_DETAIL_MAX_CHARS]


def _describe(exc: BaseException) -> str:
    message = _first_line(str(exc))
    name = type(exc).__name__
    return f"{name}: {message}" if message else name


def _leaves(exc: BaseException) -> list[BaseException]:
    """Flatten anyio task-group exception groups to their leaf exceptions."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for inner in exc.exceptions for leaf in _leaves(inner)]
    return [exc]


def _classify_stdio_failure(exc: BaseException) -> tuple[Reason, BaseException]:
    leaves = _leaves(exc)
    for leaf in leaves:
        if isinstance(leaf, TimeoutError):
            return "handshake", leaf
    for leaf in leaves:
        # TimeoutError and BrokenPipeError are OSError subclasses: the first
        # is a handshake failure, the second a child that exited early.
        if isinstance(leaf, OSError) and not isinstance(
            leaf, (TimeoutError, ConnectionError)
        ):
            return "spawn", leaf
    # The SDK answers every pending request with CONNECTION_CLOSED once the
    # child's stdout closes, so an exit before the reply surfaces here.
    for leaf in leaves:
        if _is_connection_closed(leaf):
            return "missing", leaf
    return "handshake", leaves[0]


def _is_connection_closed(exc: BaseException) -> bool:
    if isinstance(exc, McpError):
        return exc.error.code == CONNECTION_CLOSED
    return isinstance(
        exc,
        (
            anyio.ClosedResourceError,
            anyio.BrokenResourceError,
            anyio.EndOfStream,
            BrokenPipeError,
        ),
    )


def _stderr_tail(errlog: TextIO) -> str:
    errlog.seek(0)
    lines = [line.rstrip() for line in errlog if line.strip()]
    return "\n".join(lines[-_STDERR_TAIL_LINES:])


def _last_stderr_line(tail: str) -> str:
    """The child's last stderr line that says something, as the failure detail."""
    for line in reversed(tail.splitlines()):
        if _NPM_LOG_POINTER not in line:
            return _first_line(line)
    return _first_line(tail.splitlines()[-1]) if tail else ""


async def _list_tool_count(
    command: str,
    args: list[str],
    env: dict[str, str],
    errlog: TextIO,
    timeout: float,
) -> int:
    params = StdioServerParameters(command=command, args=args, env=env)
    async with (
        stdio_client(params, errlog=errlog) as (read, write),
        ClientSession(read, write) as session,
    ):
        # Only the handshake is bounded: the SDK's stdio teardown (its own 2 s
        # wait, then SIGTERM/SIGKILL of the process group) must never be
        # cancelled, and the count is returned before it runs.
        with anyio.fail_after(timeout if timeout > 0 else None):
            await session.initialize()
            result = await session.list_tools()
        return len(result.tools)


async def check_stdio(
    launch_config: Mapping[str, Any], *, timeout: float, offline: bool
) -> ConnectorReadiness:
    """Spawn the row's command and require a tools/list reply with a tool."""
    command = str(launch_config.get("command", ""))
    raw_args = launch_config.get("args")
    # Mirrors _oauth_launch_config_args (web/tools/config.py) and
    # parse_string_list (web/api/mcp.py); not imported, to keep startup fast.
    if isinstance(raw_args, list):
        args = [str(arg) for arg in raw_args]
    elif isinstance(raw_args, str):
        try:
            args = shlex.split(raw_args)
        except ValueError:
            args = raw_args.split()
    else:
        args = []
    env = placeholder_env(launch_config)
    if offline:
        env.update(OFFLINE_ENV)
    # Same PATH injection as sessions._create_stdio_session; the SDK adds the
    # rest of its inherited allowlist (HOME, USER, ...) underneath.
    env.setdefault("PATH", os.environ.get("PATH", ""))

    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as errlog:
        try:
            count = await _list_tool_count(command, args, env, errlog, timeout)
        # An ExceptionGroup of Exceptions is still caught here; a
        # BaseExceptionGroup carrying cancellation propagates.
        except Exception as exc:  # noqa: BLE001 - classified below
            reason, leaf = _classify_stdio_failure(exc)
            tail = _stderr_tail(errlog)
            if reason == "handshake" and isinstance(leaf, TimeoutError):
                detail = f"no initialize/tools/list reply within {timeout:g}s"
                if tail:
                    detail += f"; stderr: {_last_stderr_line(tail)}"
            elif tail:
                detail = _last_stderr_line(tail)
            else:
                detail = _describe(leaf)
            return ConnectorReadiness("FAIL", reason, detail, tail)
        tail = _stderr_tail(errlog)
    if count == 0:
        return ConnectorReadiness(
            "FAIL", "no-tools", "tools/list returned no tools", tail
        )
    return ConnectorReadiness(
        "PASS", None, f"{count} tool{'s' if count != 1 else ''}", tail
    )


async def check_remote(url: str, *, timeout: float) -> ConnectorReadiness:
    """One unauthenticated GET through the runtime's MCP HTTP client."""
    if urlsplit(url).scheme.lower() in ("ws", "wss"):
        return ConnectorReadiness("SKIP", "websocket", "websocket rows are not probed")
    # Timeout(None) disables httpx's own bound; a bare None would make the
    # factory substitute its 30 s default, so 0 would not mean unbounded here.
    http_timeout = httpx.Timeout(timeout if timeout > 0 else None)
    try:
        with anyio.fail_after(timeout if timeout > 0 else None):
            async with create_mcp_http_client(
                timeout=http_timeout, endpoint_url=url
            ) as client:
                # Never read the body: an SSE endpoint streams forever.
                async with client.stream("GET", url) as response:
                    status_code = response.status_code
    except Exception as exc:  # noqa: BLE001 - any failure is "unreachable"
        return ConnectorReadiness("FAIL", "unreachable", _describe(exc))
    return ConnectorReadiness("PASS", None, f"HTTP {status_code}")


def sandboxed_launch(launch_config: Any) -> bool:
    """Whether a task runs this launch_config in the sandbox (npx, uvx)."""
    if not isinstance(launch_config, Mapping):
        return False
    command = launch_config.get("command")
    # Unstripped, as the runtime passes it: a command with stray whitespace
    # is not sandboxed by a task and must not be reported as sandboxed here.
    return (
        isinstance(command, str)
        and bool(command.strip())
        and is_sandbox_command(command)
    )


async def check_launch(launch_config: Any, *, timeout: float) -> ConnectorReadiness:
    """Pick the check by the row's launch_config shape and run it."""
    if not isinstance(launch_config, Mapping):
        return NO_LAUNCH
    command = launch_config.get("command")
    url = launch_config.get("url")
    if isinstance(command, str) and command.strip():
        return await check_stdio(
            launch_config,
            timeout=timeout,
            offline=is_sandbox_command(command),
        )
    if isinstance(url, str) and url.strip():
        return await check_remote(url.strip(), timeout=timeout)
    return NO_LAUNCH


async def check_rows(
    rows: Iterable[Any],
    *,
    builtin_app_ids: set[str],
    sandbox_enabled: bool,
    timeout: float,
    total_timeout: float | None = None,
    exclude: set[str] | None = None,
) -> list[ConnectorReadiness]:
    """Check every row sequentially, in app_id order, within the total bound.

    Raises ``ValueError`` when ``exclude`` names an app_id that is not a row,
    so a misspelled exclusion fails instead of excluding nothing.
    """
    rows_by_id = {str(row.app_id): row for row in rows}
    exclude = exclude or set()
    unknown = sorted(exclude - set(rows_by_id))
    if unknown:
        raise ValueError(f"unknown connector to exclude: {', '.join(unknown)}")

    deadline = (
        time.monotonic() + total_timeout
        if total_timeout is not None and total_timeout > 0
        else None
    )
    results: list[ConnectorReadiness] = []
    for app_id in sorted(rows_by_id):
        if app_id in exclude:
            continue
        row = rows_by_id[app_id]
        row_fields: dict[str, Any] = {
            "app_id": str(row.app_id),
            "origin": "builtin" if app_id in builtin_app_ids else "custom",
            "hidden": not bool(getattr(row, "is_visible_in_connector", True)),
        }
        if sandbox_enabled and sandboxed_launch(row.launch_config):
            results.append(replace(SKIPPED_SANDBOXED, **row_fields))
            continue
        row_timeout = timeout
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                exhausted = ConnectorReadiness(
                    "FAIL",
                    "handshake",
                    f"not checked: total bound of {total_timeout:g}s exhausted",
                )
                results.append(replace(exhausted, **row_fields))
                continue
            row_timeout = min(timeout, remaining) if timeout > 0 else remaining
        outcome = await check_launch(row.launch_config, timeout=row_timeout)
        results.append(replace(outcome, **row_fields))
    return results


def format_result(result: ConnectorReadiness) -> str:
    head = (
        result.status if result.reason is None else f"{result.status} {result.reason}"
    )
    label = f"connector:{result.app_id} {result.origin}"
    if result.hidden:
        label += " hidden"
    return f"{head} {label}: {result.detail}"


def _load_app_rows() -> list[Any]:
    from sqlalchemy.orm import Session

    from .models.database import create_database_engine
    from .models.public_mcp import PublicMCPApp

    engine = create_database_engine(read_only=True)
    try:
        with Session(engine) as db:
            rows = db.query(PublicMCPApp).all()
            db.expunge_all()
            return rows
    finally:
        engine.dispose()


def _seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        seconds = math.nan
    if math.isnan(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("must be a non-negative number of seconds")
    return seconds


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    out = sys.stdout if out is None else out
    parser = argparse.ArgumentParser(
        prog="python -m xagent.web.connector_readiness",
        description="Check that every catalog connector starts in this "
        "container, with placeholder credentials. Prints no values.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        metavar="APP_ID",
        help="catalog app ID (e.g. slack) this deployment leaves unstartable on "
        "purpose; repeatable or comma-separated",
    )
    parser.add_argument(
        "--total-timeout",
        type=_seconds,
        metavar="SECONDS",
        help="bound for the whole run in seconds; connectors not reached in "
        "time fail as handshake. 0 disables it. Default: connector count times "
        "the per-connector timeout (XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS), or "
        "disabled when that is 0",
    )
    args = parser.parse_args(argv)

    from ..core.tools.core.mcp.http import MCPNetworkPolicy
    from ..core.utils.security import PrivateNetworkHostError

    try:
        MCPNetworkPolicy()
    except PrivateNetworkHostError as exc:
        print(f"FAIL configuration: {exc}", file=out)
        return 1

    from ..config import get_mcp_tool_init_timeout_seconds
    from .builtin_mcp_registry import get_builtin_public_mcp_app_rows
    from .oauth_readiness import _parse_exclude

    try:
        rows = _load_app_rows()
    except Exception as exc:
        # The class name only: a driver error can carry the database host.
        print(f"FAIL database: {type(exc).__name__}", file=out)
        return 1

    exclude = _parse_exclude(args.exclude)
    timeout = float(get_mcp_tool_init_timeout_seconds())
    total_timeout = args.total_timeout
    if total_timeout is None and timeout > 0:
        checked = [row for row in rows if str(row.app_id) not in exclude]
        total_timeout = timeout * len(checked)
    builtin_app_ids = {str(row["app_id"]) for row in get_builtin_public_mcp_app_rows()}
    try:
        results = anyio.run(
            lambda: check_rows(
                rows,
                builtin_app_ids=builtin_app_ids,
                sandbox_enabled=sandbox_enabled_from_env(),
                timeout=timeout,
                total_timeout=total_timeout,
                exclude=exclude,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not results:
        print("FAIL catalog: no public_mcp_apps rows to check", file=out)
        return 1

    failed = sum(result.failed for result in results)
    skipped = sum(result.status == "SKIP" for result in results)
    for result in results:
        print(format_result(result), file=out)
    print(f"{len(results)} checks, {failed} failed, {skipped} skipped", file=out)
    return 1 if failed else 0


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Same as the server's own entry point (xagent.web.__main__).
    load_dotenv()
    sys.exit(main())
