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

TOOL_RUNNER_PROBE_MARKER = "XAGENT_SANDBOX_TOOL_RUNNER_READY"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for sandboxed tool execution."""
    parser = argparse.ArgumentParser(description="Run a sandboxed Xagent tool")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="verify that the mounted runner and its imports are readable",
    )
    parser.add_argument("--execution-spec-b64")
    args_group = parser.add_mutually_exclusive_group()
    args_group.add_argument("--args-b64")
    args_group.add_argument("--args-file")
    parser.add_argument("--result-file")
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument("--init-params-b64")
    init_group.add_argument("--init-params-file")
    parsed = parser.parse_args()

    execution_args = (
        parsed.execution_spec_b64,
        parsed.args_b64,
        parsed.args_file,
        parsed.result_file,
        parsed.init_params_b64,
        parsed.init_params_file,
    )
    if parsed.probe:
        if any(value is not None for value in execution_args):
            parser.error("--probe cannot be combined with execution arguments")
        return parsed

    if parsed.execution_spec_b64 is None:
        parser.error("the following arguments are required: --execution-spec-b64")
    if parsed.args_b64 is None and parsed.args_file is None:
        parser.error("one of the arguments --args-b64 --args-file is required")
    if parsed.result_file is None:
        parser.error("the following arguments are required: --result-file")
    return parsed


def _load_tool_class(import_path: str) -> type[Any]:
    """Import the class named by a module:path reference."""
    module_path, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return cast(type[Any], getattr(module, class_name))


def _load_args(args_b64: str | None, args_file: str | None) -> dict[str, Any]:
    """Load JSON tool arguments from a private file or legacy base64."""
    if args_file:
        args_json = Path(args_file).read_text("utf-8")
    elif args_b64:
        args_json = base64.b64decode(args_b64).decode("utf-8")
    else:  # argparse enforces one source; keep the loader fail closed by itself.
        raise ValueError("tool arguments are missing")
    return cast(dict[str, Any], json.loads(args_json))


def _load_init_params(
    init_params_b64: str | None, init_params_file: str | None
) -> dict[str, Any]:
    """Decode optional cloudpickled init params from a private file or argv."""
    if init_params_file:
        init_params_b64 = Path(init_params_file).read_text("ascii")
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
        if parsed.probe:
            print(TOOL_RUNNER_PROBE_MARKER)
            return
        execution_spec = _load_execution_spec(parsed)
        init_params = _load_init_params(parsed.init_params_b64, parsed.init_params_file)
        tool_args = _load_args(parsed.args_b64, parsed.args_file)
    except Exception as e:
        print(f"Sandbox config error: {e}", file=sys.stderr)
        raise

    result = _execute_from_spec(execution_spec, init_params, tool_args)

    result_path = Path(cast(str, parsed.result_file))
    # ASCII escapes keep lone surrogates (undecodable filenames) encodable.
    result_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
