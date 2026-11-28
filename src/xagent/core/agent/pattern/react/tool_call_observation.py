"""Observe the structure of complete, provider-adapted ReAct tool batches."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from referencing import Registry

from ....model.chat.basic.base import ChatRequestIdentity
from ....model.chat.stream_progress import STREAM_ABORTED_KEY

# Matching is exact within each provider. The configured provider is already
# canonicalized by the model factory. Aliases serving other variants or custom
# deployments need their own explicit entry.
_PROFILE_BY_TARGET = {
    ("zhipu", "glm-5"): "glm-5",
    ("zhipuai-coding-plan", "glm-5"): "glm-5",
    ("openrouter", "z-ai/glm-5"): "glm-5",
    ("openrouter", "moonshotai/kimi-k2.5"): "kimi-k2.5",
}
_MAX_FINDINGS = 32
# Tool discovery can supply schemas with external references. Observation must
# never fetch them. A fresh offline registry still resolves local $defs.
_OFFLINE_SCHEMA_REGISTRY: Registry[Any] = Registry()


def tool_call_parts(call: Any) -> tuple[Any, Any, Any] | None:
    """Read the id, name and original arguments from an adapted function call."""
    if isinstance(call, dict):
        function = call.get("function")
        if isinstance(function, dict):
            return call.get("id"), function.get("name"), function.get("arguments", {})
        return (
            call.get("id"),
            call.get("name"),
            call.get("args", call.get("arguments", {})),
        )
    function = getattr(call, "function", None)
    if function is None:
        return None
    return (
        getattr(call, "id", None),
        getattr(function, "name", None),
        getattr(function, "arguments", {}),
    )


def _argument_finding(arguments: Any, schema: Any) -> str | None:
    if isinstance(arguments, str):
        # Providers may send blank arguments for parameterless tools (#1501).
        if not arguments.strip():
            arguments = {}
        else:
            try:
                arguments = json.loads(arguments)
            except (ValueError, RecursionError):
                return "invalid_json"
    if not isinstance(arguments, dict):
        return "arguments_not_object"
    if not isinstance(schema, (dict, bool)):
        return None
    try:
        validator = Draft202012Validator(schema, registry=_OFFLINE_SCHEMA_REGISTRY)
        if not validator.is_valid(arguments):
            return "schema_mismatch"
    except Exception:
        # Provider schemas can be malformed, cyclic, or refer to external
        # resources. Observation does not affect execution and should not
        # include exception details, which can contain parameter values.
        return "schema_unavailable"
    return None


def observe_tool_call_batch(
    llm: Any, response: Any, offered_tools: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return redacted trace metadata for an exact model/profile match."""
    if not isinstance(response, dict):
        return {}
    calls = response.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return {}
    if response.get("finish_reason") in ("length", "max_tokens") or (
        response.get(STREAM_ABORTED_KEY) and not response.get("stream_fallback")
    ):
        return {}
    identity = getattr(llm, "request_identity", None)
    if not isinstance(identity, ChatRequestIdentity):
        return {}
    profile = _PROFILE_BY_TARGET.get((identity.provider, identity.model_name))
    if profile is None:
        return {}

    schemas = {
        function["name"]: function.get("parameters")
        for tool in offered_tools
        if isinstance(function := tool.get("function"), dict)
        and isinstance(function.get("name"), str)
    }
    called_names: set[str] = set()
    findings: list[dict[str, Any]] = []
    finding_count = 0
    for index, call in enumerate(calls):
        _, name, arguments = tool_call_parts(call) or (None, None, None)
        if not isinstance(name, str) or not name:
            finding: str | None = "missing_tool_name"
        else:
            called_names.add(name)
            if name in schemas:
                finding = _argument_finding(arguments, schemas[name])
            else:
                finding = "tool_not_offered"
        if finding is not None:
            finding_count += 1
            if len(findings) < _MAX_FINDINGS:
                findings.append({"call_index": index, "reason": finding})

    return {
        "tool_call_observation": {
            "version": 1,
            "mode": "observe",
            "profile": profile,
            "identity": asdict(identity),
            "call_count": len(calls),
            "distinct_tool_count": len(called_names),
            "offered_tool_count": len(schemas),
            "called_offered_tool_count": len(called_names.intersection(schemas)),
            "finding_count": finding_count,
            "findings": findings,
        }
    }
