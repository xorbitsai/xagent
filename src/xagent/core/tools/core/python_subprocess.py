"""
Run execute_python_code in a child process with a wall-clock limit.

A worker thread running exec() cannot be stopped, so runaway code would hold
the agent turn forever. The child process returns the same result as the
in-process executor (preloaded modules, captured output, variable listing) and
can be killed, together with anything it spawned, once the limit expires. With
capture_output=False the child's own prints are discarded rather than reaching
the host's stdout.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

# Loads python_executor.py by file path so the child needs neither the xagent
# package nor its import-time side effects.
_BOOTSTRAP = """
import importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location("_xagent_python_executor", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open(sys.argv[2], encoding="utf-8") as f:
    request = json.load(f)
result = module.PythonExecutorCore(request["working_directory"]).execute_code(
    request["code"], request["capture_output"]
)
with open(sys.argv[3], "w", encoding="utf-8") as f:
    # ASCII escapes keep lone surrogates (undecodable filenames) encodable.
    json.dump(result, f)
# Exit without waiting on threads the code left running, which would hold the
# interpreter open until the limit. Files the code left open were already
# closed when its namespace was freed; only atexit hooks it registered are
# skipped.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
"""

_EXECUTOR_PATH = Path(__file__).with_name("python_executor.py")
_STDERR_TAIL_BYTES = 2000


def _kill_process_group(process: "subprocess.Popen[bytes]") -> None:
    """Kill the child and every process it started in its session."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
    process.wait()


def _timeout_result(timeout_seconds: float) -> Dict[str, Any]:
    return {
        "success": False,
        "output": "",
        "error": (
            f"Execution timed out after {timeout_seconds:g} seconds and the Python "
            "process was terminated. Split the work into smaller steps or avoid "
            "unbounded operations such as recursive scans of '/'."
        ),
    }


def run_python_code_in_subprocess(
    code: str,
    *,
    capture_output: bool = True,
    working_directory: Optional[str] = None,
    timeout_seconds: float,
) -> Dict[str, Any]:
    """
    Execute Python code in a child process and return the executor result.

    Args:
        code: Python code to execute
        capture_output: Whether to capture stdout/stderr
        working_directory: Directory to use as working directory
        timeout_seconds: Wall-clock limit before the process group is killed

    Returns:
        Dictionary with success status, output, and error information
    """
    temp_dir = tempfile.mkdtemp(prefix="xagent_python_exec_")
    try:
        request_path = os.path.join(temp_dir, "request.json")
        result_path = os.path.join(temp_dir, "result.json")
        stderr_path = os.path.join(temp_dir, "stderr.txt")
        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "code": code,
                    "capture_output": capture_output,
                    "working_directory": working_directory,
                },
                f,
            )

        # Output goes to files, not pipes: a pipe held open by a leftover
        # grandchild would keep the read blocked after the child exits.
        with (
            open(os.devnull, "wb") as stdout_file,
            open(stderr_path, "wb") as stderr_file,
        ):
            process = subprocess.Popen(
                # -P keeps a module in the working directory from shadowing
                # the bootstrap's own imports.
                [
                    sys.executable,
                    "-P",
                    "-c",
                    _BOOTSTRAP,
                    str(_EXECUTOR_PATH),
                    request_path,
                    result_path,
                ],
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=os.name == "posix",
            )
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                _kill_process_group(process)
                return _timeout_result(timeout_seconds)

        try:
            with open(result_path, encoding="utf-8") as f:
                result: Dict[str, Any] = json.load(f)
            return result
        except (OSError, ValueError):
            with open(stderr_path, "rb") as f:
                f.seek(max(0, os.path.getsize(stderr_path) - _STDERR_TAIL_BYTES))
                stderr_tail = f.read().decode("utf-8", errors="replace")
            return {
                "success": False,
                "output": "",
                "error": (
                    f"Python process exited with code {process.returncode} "
                    f"before returning a result.\n{stderr_tail}"
                ).strip(),
            }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
