"""Describe connector exposure from the tools actually supplied to a run."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def connector_tool_sources(tools: Sequence[Any]) -> dict[str, str]:
    """Read public origin metadata, never account configuration or credentials."""
    sources: dict[str, str] = {}
    for tool in tools:
        metadata = getattr(tool, "metadata", None)
        name = getattr(metadata, "name", None) or getattr(tool, "name", None)
        source = getattr(metadata, "source_server", None)
        if isinstance(name, str) and isinstance(source, str) and source:
            sources[name] = source
    return sources


def tool_access_context(sources: Mapping[str, str], tool_names: Sequence[str]) -> str:
    """Render current exposure, without claiming successful authorization."""
    exposed: dict[str, list[str]] = {}
    for name in tool_names:
        if name in sources:
            exposed.setdefault(sources[name], []).append(name)
    generic_http = "api_call" in tool_names and "api_call" not in sources
    if not exposed and not generic_http:
        return ""
    inventory = json.dumps(exposed, ensure_ascii=True, sort_keys=True)
    return (
        "Current connector-tool inventory (source -> exposed tool names): "
        f"{inventory}. "
        "This lists only this call's tools, not all connections in the user's "
        "account, and does not prove that authorization or a request will succeed. "
        "An empty inventory means no connector-backed tool is exposed in this call. "
        + (
            "api_call is a generic HTTP client, not a configured connection to "
            "the user's systems; it has no preconfigured endpoint or stored "
            "service credentials. "
            if generic_http
            else ""
        )
        + "Use supplied files or data, a relevant exposed connector, or an "
        "endpoint established by the user, configuration, documentation or tool "
        "results. A system's name or business purpose is not an endpoint. When "
        "the required source is missing, do not probe invented URLs. "
        + (
            "Use ask_user_question to request the missing connection or "
            "uploaded/pasted data. "
            if "ask_user_question" in tool_names
            else "Report the missing source without claiming completion. "
        )
        + "Do not claim a connection is absent from "
        "the account just because this task cannot access it."
    )
