"""Deferred connector tool schemas for the ReAct loop.

A run with many connected MCP / API connectors would otherwise send every
connector tool's full JSON schema on every model call. When a host opts in
(``defer_connector_schemas``), the connector tools stay in the run's tool list
-- so lookup, approval gating, write guards and tracing are untouched -- but
their schemas are withheld until the model asks for them with ``load_tools``.

The model still sees a compact index (the tool names, grouped by source, in
the system context). ``load_tools`` records the requested names in the run
context's ``loaded_tools`` component; from the next step on their schemas are
appended to the tool list. Because the component is part of the execution
context it survives checkpoints and approval pause/resume.

Connectors the user names in the latest message are pre-loaded, within a
token budget, so the common "do X in HubSpot" turn needs no extra model call.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel, Field

LOAD_TOOLS_NAME = "load_tools"
LOADED_TOOLS_COMPONENT = "loaded_tools"

# One load_tools call may name at most this many tools. A larger request is
# truncated and the model is told which names were not loaded.
MAX_TOOLS_PER_LOAD = 12

# How many extra protocol retries a run may spend auto-loading deferred tools a
# provider keeps refusing. A provider that reports only the first refused call
# of a response needs one per tool the model called in parallel.
MAX_AUTO_LOAD_RETRIES = 3

# Default ceiling, in estimated tokens, for the schemas pre-loaded because the
# user named their connector. A connector that would exceed what is left of the
# budget is not pre-loaded; the model loads the few tools it needs by name.
DEFAULT_PRELOAD_TOKEN_BUDGET = 20_000


class LoadToolsArgs(BaseModel):
    names: list[str] = Field(
        description=(
            "Tool names copied exactly from the connector-tool index, "
            "including capitalization."
        )
    )


class LoadToolsTool:
    """The ``load_tools`` tool, bound to one run's context.

    Built per run, like the ``load_skill`` tool, with a handler that closes
    over the run's context, so it executes through the ordinary tool path.
    """

    name = LOAD_TOOLS_NAME
    description = (
        "Load the full schemas of connector tools listed in the connector-tool "
        "index, so they can be called. Connector tools in the index cannot be "
        "called until loaded. Name only the tools you need (up to "
        f"{MAX_TOOLS_PER_LOAD} per call); the schemas become available in the "
        "next step, so call the loaded tools in a following step, not in the "
        "same response."
    )
    metadata = None

    def __init__(self, handler: Callable[[Any], str]) -> None:
        self._handler = handler

    def args_type(self) -> Any:
        return LoadToolsArgs

    async def execute(self, names: Any) -> str:
        return self._handler(names)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


DEFERRED_RULE = (
    "Connector tools in the inventory above are deferred: their schemas are "
    f"not sent with every call. To use one, first call {LOAD_TOOLS_NAME} "
    "with the names you need, copied exactly as listed (capitalization "
    "included); they can be called from the next step. "
    "A tool that is not loaded cannot be called. Load only what the request "
    "needs."
)

# Appended to the "Available tool names" sentence so it stays true, and stays
# the same text, whatever has been loaded.
AVAILABLE_NAMES_SUFFIX = (
    " Connector tools whose schemas are included in this call, whether "
    f"preloaded or loaded with {LOAD_TOOLS_NAME}, are also available."
)


def servers_mentioned(text: str, servers: Iterable[str]) -> list[str]:
    """Sources whose name appears as whole words in ``text``."""
    haystack = f" {_normalize(text)} "
    found: list[str] = []
    for server in servers:
        needle = _normalize(server)
        if needle and f" {needle} " in haystack:
            found.append(server)
    return found


def tool_names_mentioned(text: str, names: Iterable[str]) -> list[str]:
    """Exact deferred tool names quoted in ``text``."""
    return [
        name
        for name in names
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)
    ]
