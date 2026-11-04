"""Tests for running execute_python_code in a bounded child process."""

import os
import sys
import time
from pathlib import Path

import pytest

from xagent.core.tools.core.python_subprocess import run_python_code_in_subprocess

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="process-group kill is POSIX-only"
)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_until_gone(pid: int, deadline_seconds: float) -> bool:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        if not _process_alive(pid):
            return True
        time.sleep(0.05)
    return not _process_alive(pid)


def test_returns_captured_output():
    result = run_python_code_in_subprocess(
        "print('hello')\nprint(2 + 3)", timeout_seconds=60
    )

    assert result == {"success": True, "output": "hello\n5\n", "error": ""}


def test_preloaded_modules_and_variable_listing_survive():
    result = run_python_code_in_subprocess(
        "value = round(math.pi, 2)\nencoded = json.dumps([1])", timeout_seconds=60
    )

    assert result["success"] is True
    assert "Variables created:" in result["output"]
    assert "value = 3.14" in result["output"]
    assert "encoded = '[1]'" in result["output"]


def test_runtime_error_is_reported():
    result = run_python_code_in_subprocess("1 / 0", timeout_seconds=60)

    assert result["success"] is False
    assert "ZeroDivisionError" in result["error"]


def test_runs_in_working_directory_without_moving_host_cwd(tmp_path):
    host_cwd = os.getcwd()

    result = run_python_code_in_subprocess(
        "open('out.txt', 'w').write('x')",
        working_directory=str(tmp_path),
        timeout_seconds=60,
    )

    assert result["success"] is True
    assert (tmp_path / "out.txt").read_text() == "x"
    assert os.getcwd() == host_cwd


def test_working_directory_module_does_not_shadow_stdlib(tmp_path):
    (tmp_path / "json.py").write_text("raise RuntimeError('shadowed')\n")

    result = run_python_code_in_subprocess(
        "print('ok')", working_directory=str(tmp_path), timeout_seconds=60
    )

    assert result == {"success": True, "output": "ok\n", "error": ""}


def test_process_exit_without_result_is_reported():
    result = run_python_code_in_subprocess(
        "import sys\nprint('partial', file=sys.stderr)\nsys.exit(3)",
        timeout_seconds=60,
    )

    assert result["success"] is False
    assert "exited with code 3" in result["error"]


def test_runaway_code_times_out_within_limit():
    started = time.monotonic()

    result = run_python_code_in_subprocess("while True:\n    pass", timeout_seconds=1)

    assert time.monotonic() - started < 10
    assert result["success"] is False
    assert result["output"] == ""
    assert "timed out after 1 seconds" in result["error"]


def test_timeout_kills_child_processes(tmp_path):
    pid_file = tmp_path / "child.pid"
    # Written via rename so a kill mid-write never leaves an empty pid file.
    grandchild = (
        "import os, time; "
        f"open({str(pid_file) + '.tmp'!r}, 'w').write(str(os.getpid())); "
        f"os.replace({str(pid_file) + '.tmp'!r}, {str(pid_file)!r}); "
        "time.sleep(600)"
    )
    code = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        "while True:\n"
        "    time.sleep(0.1)\n"
    )

    # Leaves room for a cold pandas/matplotlib import on a loaded CI worker.
    result = run_python_code_in_subprocess(code, timeout_seconds=10)

    assert "timed out" in result["error"]
    child_pid = int(Path(pid_file).read_text())
    assert _wait_until_gone(child_pid, deadline_seconds=5)


def test_child_holding_output_open_does_not_block_return():
    code = (
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        "print(child.pid)\n"
    )
    started = time.monotonic()

    result = run_python_code_in_subprocess(code, timeout_seconds=60)

    try:
        assert time.monotonic() - started < 30
        assert result["success"] is True
    finally:
        os.kill(int(result["output"]), 9)


def test_does_not_touch_host_streams():
    stdout, stderr = sys.stdout, sys.stderr

    run_python_code_in_subprocess("print('x')", timeout_seconds=60)

    assert sys.stdout is stdout
    assert sys.stderr is stderr


def test_undecodable_filename_output_survives():
    # os.listdir/glob on Linux return lone surrogates for non-UTF-8 names.
    result = run_python_code_in_subprocess(
        "print(b'\\xff'.decode('utf-8', 'surrogateescape'))", timeout_seconds=60
    )

    assert result == {"success": True, "output": "\udcff\n", "error": ""}


def test_leftover_non_daemon_thread_does_not_turn_success_into_timeout():
    code = (
        "import threading, time\n"
        "threading.Thread(target=time.sleep, args=(600,)).start()\n"
        "print('done')\n"
    )
    started = time.monotonic()

    result = run_python_code_in_subprocess(code, timeout_seconds=60)

    assert time.monotonic() - started < 30
    assert result == {"success": True, "output": "done\n", "error": ""}


def test_unclosed_file_is_still_written_before_the_child_exits(tmp_path):
    result = run_python_code_in_subprocess(
        "report = open('report.txt', 'w')\nreport.write('kept')\n",
        working_directory=str(tmp_path),
        timeout_seconds=60,
    )

    assert result["success"] is True
    assert (tmp_path / "report.txt").read_text() == "kept"


def test_crash_reports_the_end_of_a_large_stderr():
    code = (
        # fd 2 directly: the executor captures sys.stderr into a buffer.
        "import os\nos.write(2, b'x' * 100_000 + b'LAST-LINE')\nos._exit(7)\n"
    )

    result = run_python_code_in_subprocess(code, timeout_seconds=60)

    assert result["success"] is False
    assert "exited with code 7" in result["error"]
    assert result["error"].endswith("LAST-LINE")
    assert len(result["error"]) < 3000
