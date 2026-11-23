"""Cross-build cache of stdio MCP tool discovery.

An agent build learns a stdio server's tools by spawning the server and
running ``initialize`` + ``tools/list`` (``_load_direct_mcp_tools``). That
process serves discovery only: every tool call opens a session of its own.
With many stdio servers per user (built-in OAuth connectors run as stdio
children), those spawns dominate agent setup and compete for the host when
builds overlap.

This cache keeps the listing, not the process. A later build that launches
the server exactly the same way reuses it for a short while and spawns
nothing. What a build does with the listing (read-only allowlists, signed
uploads, adapters, approval gating) still runs on every build against the
live connection.

The key is a digest of what the child is launched with: ``command``,
``args``, ``cwd``, the encoding settings and ``env``, including the process
``PATH`` the launcher adds when the env names none. Credentials stay in the
key, so a listing is never shared across tokens or accounts. Only the task
file directories the host writes into the env for every task (the
``XAGENT_*_ALLOWED_DIRS`` and ``XAGENT_*_OUTPUT_DIR`` names it sets) enter it
by name: their values change on every task and do not change what a server
lists. Any other per-task value simply misses, which is today's behaviour.

The cache is opt-in per call (``discovery_cache=True`` on the loader): agent
builds use it, while connection tests and connector checks keep reaching the
live server. It is process-local, bounded, and holds only successful
listings with at least one tool.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, NamedTuple

from mcp.types import Tool as MCPTool

from ..... import config as _root_config
from ...core.mcp.tools import attach_raw_annotations, raw_annotations_for

logger = logging.getLogger(__name__)

# The env values the host sets per task: ``WebToolConfig`` writes the task's
# file directories into built-in stdio connectors' env under exactly these
# names. A name not listed here keys by value, so a name added there and not
# here only costs cache hits (a test keeps the two in step).
_TASK_FILE_DIRECTORY_ENV = frozenset(
    {
        "XAGENT_FACEBOOK_IMAGE_ALLOWED_DIRS",
        "XAGENT_GMAIL_FILE_ALLOWED_DIRS",
        "XAGENT_GOOGLE_DRIVE_FILE_ALLOWED_DIRS",
        "XAGENT_GOOGLE_DRIVE_OUTPUT_DIR",
        "XAGENT_JIRA_FILE_ALLOWED_DIRS",
        "XAGENT_LINKEDIN_IMAGE_ALLOWED_DIRS",
        "XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS",
        "XAGENT_ONEDRIVE_OUTPUT_DIR",
        "XAGENT_SHAREPOINT_FILE_ALLOWED_DIRS",
        "XAGENT_SHAREPOINT_OUTPUT_DIR",
        "XAGENT_SLACK_FILE_ALLOWED_DIRS",
    }
)
_TASK_FILE_DIRECTORY_PLACEHOLDER = "<task file directory>"

# What ``create_session`` hands a stdio child, besides ``env``.
_LAUNCH_FIELDS = ("command", "args", "cwd", "encoding", "encoding_error_handler")


class _Listing(NamedTuple):
    stored_at: float
    tools: tuple[MCPTool, ...]


_MAX_ENTRIES = 256
_clock = time.monotonic
_lock = threading.Lock()
_entries: OrderedDict[str, _Listing] = OrderedDict()


def _path_or_reject(value: Any) -> Any:
    """``json.dumps`` hook: a path is its string; any other object has no key."""
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(type(value).__name__)


def discovery_key(connection: Mapping[str, Any]) -> str | None:
    """The cache key for ``connection``'s discovery, or ``None`` if it has none.

    Only a stdio connection with a command has one. A launch the key cannot
    represent exactly has none either, so it is always discovered live: a
    value that is not plain data, or ``session_kwargs``, which can change the
    client's own ``initialize``.

    Like every function here, it never raises: the cache must never fail a
    load that would have succeeded live.
    """
    try:
        return _launch_digest(connection)
    except Exception:
        logger.debug("No MCP discovery cache key for this launch", exc_info=True)
        return None


def _launch_digest(connection: Mapping[str, Any]) -> str | None:
    if connection.get("transport") != "stdio":
        return None
    if connection.get("session_kwargs"):
        return None
    command = connection.get("command")
    if not isinstance(command, str) or not command:
        return None
    env = connection.get("env") or {}
    if not isinstance(env, Mapping):
        return None
    # Key on the env as the launcher sends it: it adds the process PATH when
    # the env has none, writing it into the caller's mapping when that is
    # non-empty, so the key must not change once that has happened.
    env = dict(env)
    if "PATH" not in env:
        env["PATH"] = os.environ.get("PATH", "")
    launch: dict[str, Any] = {field: connection.get(field) for field in _LAUNCH_FIELDS}
    launch["env"] = {
        str(name): (
            _TASK_FILE_DIRECTORY_PLACEHOLDER
            if str(name) in _TASK_FILE_DIRECTORY_ENV
            else value
        )
        for name, value in env.items()
    }
    try:
        payload = json.dumps(launch, sort_keys=True, default=_path_or_reject)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(payload.encode()).hexdigest()


def _copy_with_annotations(tool: MCPTool) -> MCPTool:
    # The wire annotations ride beside the model, not in it (see
    # ``attach_raw_annotations``), so they are copied explicitly.
    raw = raw_annotations_for(tool)
    return attach_raw_annotations(
        tool.model_copy(deep=True), deepcopy(raw) if raw is not None else None
    )


def cached_discovery(key: str) -> list[MCPTool] | None:
    """The listing remembered under ``key``, as fresh copies, or ``None``."""
    try:
        ttl = _root_config.get_mcp_tool_discovery_cache_seconds()
        if ttl <= 0:
            return None
        with _lock:
            listing = _entries.get(key)
            if listing is None:
                return None
            if _clock() - listing.stored_at > ttl:
                del _entries[key]
                return None
            _entries.move_to_end(key)
        return [_copy_with_annotations(tool) for tool in listing.tools]
    except Exception:
        logger.debug("MCP discovery cache read failed; listing live", exc_info=True)
        return None


def remember_discovery(key: str, tools: Sequence[MCPTool]) -> None:
    """Remember a successful listing under ``key``; an empty one is ignored.

    Expired listings are dropped on the way, so memory follows recent use.
    """
    try:
        ttl = _root_config.get_mcp_tool_discovery_cache_seconds()
        if not tools or ttl <= 0:
            return
        copies = tuple(_copy_with_annotations(tool) for tool in tools)
        now = _clock()
        with _lock:
            expired = [
                stale
                for stale, listing in _entries.items()
                if now - listing.stored_at > ttl
            ]
            for stale in expired:
                del _entries[stale]
            _entries[key] = _Listing(now, copies)
            _entries.move_to_end(key)
            while len(_entries) > _MAX_ENTRIES:
                _entries.popitem(last=False)
    except Exception:
        logger.debug("MCP discovery cache write failed", exc_info=True)


def clear_discovery_cache() -> None:
    """Forget every remembered listing."""
    with _lock:
        _entries.clear()
