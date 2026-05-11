"""Execute a sandboxed tool from a stable Python entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, cast

import cloudpickle  # type: ignore[import-untyped]

# WARNING: This file runs as a standalone script in the sandbox, not a module.
# Absolute imports only — relative imports are unavailable (no package context).
from xagent.core.tools.adapters.vibe.sandboxed_tool.runner_utils import (
    ensure_user_bin_in_path,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for sandboxed tool execution."""
    parser = argparse.ArgumentParser(description="Run a sandboxed Xagent tool")
    parser.add_argument("--execution-spec-b64", required=True)
    parser.add_argument("--args-b64", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--init-params-b64")
    return parser.parse_args()


def _load_tool_class(import_path: str) -> type[Any]:
    """Import the class named by a module:path reference."""
    module_path, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return cast(type[Any], getattr(module, class_name))


def _load_args(args_b64: str) -> dict[str, Any]:
    """Decode JSON tool arguments from base64."""
    args_json = base64.b64decode(args_b64).decode("utf-8")
    return cast(dict[str, Any], json.loads(args_json))


def _load_init_params(init_params_b64: str | None) -> dict[str, Any]:
    """Decode optional cloudpickled init params from base64."""
    if not init_params_b64:
        return {}
    return cast(dict[str, Any], cloudpickle.loads(base64.b64decode(init_params_b64)))


def _load_execution_spec(parsed: argparse.Namespace) -> dict[str, Any]:
    """Decode the serialized execution spec from CLI args."""
    spec_json = base64.b64decode(parsed.execution_spec_b64).decode("utf-8")
    return cast(dict[str, Any], json.loads(spec_json))


def _run_tool(tool: Any, args: dict[str, Any]) -> Any:
    """Execute a tool instance through its JSON run interface."""
    if inspect.iscoroutinefunction(tool.run_json_async):
        return asyncio.run(tool.run_json_async(args))
    return tool.run_json_sync(args)


def _run_method(method: Any, args: dict[str, Any]) -> Any:
    """Execute a bound method using the decoded keyword arguments."""
    if inspect.iscoroutinefunction(method):
        return asyncio.run(method(**args))
    return method(**args)


def _validate_spec(spec: dict[str, Any]) -> None:
    """Validate execution spec format before use.

    Raises ValueError with a clear message when required keys are missing.
    """
    kind = spec.get("kind")
    if kind not in ("tool", "method"):
        raise ValueError(
            f"Unsupported execution kind: {kind!r}. Expected 'tool' or 'method'."
        )
    if "tool_class" not in spec:
        raise ValueError(
            f"Execution spec (kind={kind!r}) missing required key 'tool_class'."
        )
    if kind == "method" and "method_name" not in spec:
        raise ValueError(
            "Execution spec (kind='method') missing required key 'method_name'."
        )


def _execute_from_spec(
    spec: dict[str, Any], init_params: dict[str, Any], args: dict[str, Any]
) -> Any:
    """Reconstruct the target object from spec and execute it.

    Each call creates a fresh instance -- state is not preserved across
    invocations because the sandbox runner is a separate process.
    """
    _validate_spec(spec)

    tool_class = _load_tool_class(spec["tool_class"])
    instance = tool_class(**init_params)

    if spec["kind"] == "tool":
        return _run_tool(instance, args)

    if spec["kind"] == "method":
        method = getattr(instance, spec["method_name"])
        return _run_method(method, args)

    raise ValueError(f"Unsupported execution kind: {spec['kind']}")  # pragma: no cover


def main() -> None:
    """CLI entrypoint for sandboxed tool execution."""
    ensure_user_bin_in_path()
    try:
        parsed = _parse_args()
        execution_spec = _load_execution_spec(parsed)
        init_params = _load_init_params(parsed.init_params_b64)
        tool_args = _load_args(parsed.args_b64)
    except Exception as e:
        print(f"Sandbox config error: {e}", file=sys.stderr)
        raise

    result = _execute_from_spec(execution_spec, init_params, tool_args)

    result_path = Path(parsed.result_file)
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
