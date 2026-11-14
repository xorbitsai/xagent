"""Recognize the in-repo connector servers' ``{"status": "error"}`` result.

The FastMCP servers under ``xagent.web.tools.mcp`` catch vendor errors and
return ``{"status": "error", "message": ...}`` as text, leaving the MCP
``isError`` flag unset. ``tool_result_succeeded`` reads only the flag and the
result's own top-level keys, so such a call counted as a success
(xorbitsai/xagent#3574).

Only a connection that launches one of those modules is trusted to use the
envelope. The decision is made on the launched module, not on the server's
name: a catalog id can name a vendor's remote server (HubSpot's does, while
the repo still carries a module of that name), and a catalog row's name need
not fold to its id ("Facebook Pages" is ``facebook``). A third-party server
can return a JSON object whose ``status`` is ``"error"`` as ordinary data, so
the shape alone is never enough.

``is_error`` is left as the server sent it. The connector probe and the smoke
checks read that flag as a protocol-level failure and judge the envelope on
their own, so the adapter marks the result ``success: False`` instead, which
``tool_result_succeeded`` already reads.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

IN_REPO_SERVER_PACKAGE = "xagent.web.tools.mcp"

# The FastMCP servers in ``IN_REPO_SERVER_PACKAGE``. Each reports a vendor
# error as the envelope (directly, or through ``meta_graph.error_response``)
# or raises, which sets ``isError``. LinkedIn's server is not here: it passes
# the vendor's response text through unchanged. A test pins this set to the
# package's FastMCP modules, so a new server has to be classified.
ERROR_ENVELOPE_SERVER_MODULES = frozenset(
    {
        "aws",
        "calendar",
        "chartmogul",
        "deputy",
        "employment_hero",
        "excel",
        "facebook",
        "freshdesk",
        "github",
        "gmail",
        "google_ads",
        "google_analytics",
        "google_docs",
        "google_drive",
        "google_search_console",
        "google_sheets",
        "google_slides",
        "hubspot",
        "instagram",
        "intercom",
        "jira",
        "jira_ticket",
        "linear",
        "magento",
        "meta_ads",
        "mixpanel",
        "myob",
        "onedrive",
        "outlook",
        "planner",
        "posthog",
        "powerpoint",
        "salesforce",
        "sharepoint",
        "shopify",
        "slack",
        "stripe",
        "teams",
        "whatsapp",
        "word",
        "xero",
        "zendesk",
        "zoom",
    }
)

# Error envelopes are a status and a short message. The bound keeps a large
# successful payload from being parsed on every call; an envelope past it is
# left as a success, as it was before.
ERROR_ENVELOPE_MAX_CHARS = 16_384


def speaks_error_envelope(connection: object) -> bool:
    """Whether ``connection`` runs an in-repo server that uses the envelope.

    The connection must be stdio and run ``-m <IN_REPO_SERVER_PACKAGE>.<module>``
    with ``<module>`` in ``ERROR_ENVELOPE_SERVER_MODULES``. Only the first
    ``-m`` counts, as it does for Python: what follows it is that module's own
    argv. Both launch shapes the catalog has used match, ``python -m ...`` and
    ``uv run python -m ...``; the command is not checked, because the module
    is what decides the result's shape.
    """
    if not isinstance(connection, Mapping) or connection.get("transport") != "stdio":
        return False
    args = connection.get("args")
    if not isinstance(args, Sequence) or isinstance(args, str):
        return False
    args = list(args)
    try:
        module_index = args.index("-m") + 1
    except ValueError:
        return False
    if module_index >= len(args):
        return False
    module = args[module_index]
    if not isinstance(module, str):
        return False
    package, _, name = module.rpartition(".")
    return package == IN_REPO_SERVER_PACKAGE and name in ERROR_ENVELOPE_SERVER_MODULES


def is_error_envelope(content: object) -> bool:
    """Whether a normalized MCP ``content`` list is one error envelope.

    It must be a single text item whose text, within
    ``ERROR_ENVELOPE_MAX_CHARS``, parses to a JSON object whose ``status`` is
    exactly ``"error"``. Other statuses the servers use (``indeterminate``,
    ``conflict``, ``partial``, ...) are outcomes the model has to read, not
    failures, and stay successful.
    """
    if not isinstance(content, list) or len(content) != 1:
        return False
    item = content[0]
    if not isinstance(item, Mapping) or item.get("type") != "text":
        return False
    text = item.get("text")
    if not isinstance(text, str) or len(text) > ERROR_ENVELOPE_MAX_CHARS:
        return False
    if not text.lstrip().startswith("{"):
        return False
    try:
        payload: Any = json.loads(text)
    except (ValueError, RecursionError):
        # Text nested past the parser's depth limit raises RecursionError.
        return False
    return isinstance(payload, dict) and payload.get("status") == "error"
