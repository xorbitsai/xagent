from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xagent.core.tools.adapters.vibe.sandboxed_tool import chrome_daemon_runner


def test_validate_server_args_requires_isolation_and_blocks_managed_overrides():
    assert chrome_daemon_runner._validate_server_args(
        ["--headless", "--isolated", "--no-usage-statistics"]
    ) == ["--headless", "--isolated", "--no-usage-statistics"]
    for args in (
        ["--headless"],
        ["--isolated"],
        ["--headless", "--isolated", "--sessionId=other"],
        ["--headless", "--isolated", "--user-data-dir", "/shared"],
        ["--headless", "--isolated", "--chrome-arg=--user-data-dir=/shared"],
        [
            "--headless",
            "--isolated",
            "--chrome-arg",
            "--profile-directory=shared",
        ],
    ):
        with pytest.raises(chrome_daemon_runner.ChromeDaemonRunnerError):
            chrome_daemon_runner._validate_server_args(args)


def test_socket_request_uses_nul_framing_and_parses_one_response():
    client = MagicMock()
    client.__enter__.return_value = client
    client.recv.return_value = b'{"success":true,"result":"{}"}\0'
    with patch.object(socket, "socket", return_value=client) as constructor:
        response = chrome_daemon_runner._socket_request(
            Path("/tmp/daemon.sock"),
            {"method": "status"},
        )

    assert response == {"success": True, "result": "{}"}
    constructor.assert_called_once_with(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect.assert_called_once_with("/tmp/daemon.sock")
    client.sendall.assert_called_once_with(b'{"method":"status"}\0')


def test_socket_response_size_is_bounded_before_json_decode(monkeypatch):
    client = MagicMock()
    client.__enter__.return_value = client
    client.recv.return_value = b"123456789"
    monkeypatch.setattr(chrome_daemon_runner, "_MAX_RESPONSE_BYTES", 8)
    with patch.object(socket, "socket", return_value=client):
        with pytest.raises(
            chrome_daemon_runner.ChromeDaemonRunnerError, match="size limit"
        ):
            chrome_daemon_runner._socket_request(
                Path("/tmp/daemon.sock"), {"method": "status"}
            )


def test_start_uses_exact_pin_private_environment_and_health_check(tmp_path):
    runtime = tmp_path / "runtime"
    profile = tmp_path / "profile"
    runtime.mkdir()
    profile.mkdir()
    socket_path, pid_file = chrome_daemon_runner._runtime_paths("a" * 32, runtime)
    healthy = {
        "version": "1.6.0",
        "pid": 42,
        "args": [
            "--headless",
            "--isolated",
            "--viaCli",
            "--experimentalStructuredContent",
        ],
    }
    completed = MagicMock(returncode=0)

    with (
        patch.object(
            chrome_daemon_runner,
            "_session_environment",
            return_value=(
                {
                    "NPM_CONFIG_CACHE": "/opt/npm-cache",
                    "CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS": "1",
                },
                runtime,
                profile,
            ),
        ),
        patch.object(
            chrome_daemon_runner, "_runtime_paths", return_value=(socket_path, pid_file)
        ),
        patch.object(chrome_daemon_runner, "_status", side_effect=[None, healthy]),
        patch.object(
            chrome_daemon_runner.subprocess, "run", return_value=completed
        ) as run,
    ):
        assert (
            chrome_daemon_runner._start("a" * 32, ["--headless", "--isolated"])
            == healthy
        )

    args = run.call_args.args[0]
    assert args[:6] == [
        "npx",
        "-y",
        "--prefer-offline",
        "--package",
        "chrome-devtools-mcp@1.6.0",
        "chrome-devtools",
    ]
    assert args[-2:] == ["--headless", "--isolated"]
    assert run.call_args.kwargs["env"]["CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS"] == "1"


def test_status_contract_accepts_cli_serialization_and_official_defaults():
    status = {
        "version": "1.6.0",
        "args": [
            "--channel",
            "stable",
            "--headless",
            "--isolated",
            "--chrome-arg",
            "--disable-gpu",
            "--viaCli",
            "--experimentalStructuredContent",
        ],
    }

    assert chrome_daemon_runner._status_matches_managed_daemon(
        status,
        ["--headless", "--isolated", "--chrome-arg=--disable-gpu"],
    )


def test_status_contract_rejects_unexpected_profile_override():
    status = {
        "version": "1.6.0",
        "args": [
            "--headless",
            "--isolated",
            "--chrome-arg",
            "--user-data-dir=/shared",
            "--viaCli",
            "--experimentalStructuredContent",
        ],
    }

    assert not chrome_daemon_runner._status_matches_managed_daemon(
        status, ["--headless", "--isolated"]
    )


def test_start_fails_closed_for_healthy_daemon_with_wrong_launch_spec(tmp_path):
    runtime = tmp_path / "runtime"
    profile = tmp_path / "profile"
    daemon_home = runtime / f"chrome-devtools-mcp-{'a' * 32}"
    daemon_home.mkdir(parents=True)
    profile.mkdir()
    socket_path, pid_file = chrome_daemon_runner._runtime_paths("a" * 32, runtime)
    pid_file.write_text("42")
    wrong = {"version": "1.6.0", "args": ["--headless"]}

    with (
        patch.object(
            chrome_daemon_runner,
            "_session_environment",
            return_value=({}, runtime, profile),
        ),
        patch.object(chrome_daemon_runner, "_status", return_value=wrong),
        patch.object(chrome_daemon_runner, "_terminate_expected_daemon") as terminate,
        patch.object(
            chrome_daemon_runner.subprocess,
            "run",
            return_value=MagicMock(returncode=0),
        ) as run,
    ):
        with pytest.raises(
            chrome_daemon_runner.ChromeDaemonRunnerError,
            match="does not match",
        ):
            chrome_daemon_runner._start("a" * 32, ["--headless", "--isolated"])

    terminate.assert_not_called()
    run.assert_not_called()


def test_start_refuses_to_replace_ambiguous_live_pid(tmp_path):
    runtime = tmp_path / "runtime"
    profile = tmp_path / "profile"
    daemon_home = runtime / f"chrome-devtools-mcp-{'a' * 32}"
    daemon_home.mkdir(parents=True)
    profile.mkdir()
    socket_path, pid_file = chrome_daemon_runner._runtime_paths("a" * 32, runtime)
    pid_file.write_text("42")

    with (
        patch.object(
            chrome_daemon_runner,
            "_session_environment",
            return_value=({}, runtime, profile),
        ),
        patch.object(chrome_daemon_runner, "_status", return_value=None),
        patch.object(
            chrome_daemon_runner, "_pid_is_expected_daemon", return_value=False
        ),
        patch.object(chrome_daemon_runner, "_pid_is_alive", return_value=True),
        patch.object(chrome_daemon_runner.subprocess, "run") as run,
    ):
        with pytest.raises(
            chrome_daemon_runner.ChromeDaemonRunnerError, match="unverified live"
        ):
            chrome_daemon_runner._start("a" * 32, ["--headless", "--isolated"])

    run.assert_not_called()


def test_stop_removes_daemon_home_and_private_profile_without_live_daemon(tmp_path):
    runtime = tmp_path / "runtime"
    profile = tmp_path / "profile"
    runtime.mkdir()
    profile.mkdir()
    (profile / "history").write_text("private")
    socket_path, pid_file = chrome_daemon_runner._runtime_paths("a" * 32, runtime)
    socket_path.parent.mkdir()
    socket_path.write_text("socket")
    pid_file.write_text("pid")

    with (
        patch.object(
            chrome_daemon_runner,
            "_session_environment",
            return_value=({}, runtime, profile),
        ),
        patch.object(chrome_daemon_runner, "_status", return_value=None),
    ):
        assert chrome_daemon_runner._stop("a" * 32) == {"stopped": True}

    assert not profile.exists()
    assert not socket_path.parent.exists()


@pytest.mark.parametrize("failed_tree", ["daemon_home", "profile"])
def test_stop_cleanup_oserror_does_not_block_other_tree(tmp_path, failed_tree):
    runtime = tmp_path / "runtime"
    profile = tmp_path / "profile"
    runtime.mkdir()
    profile.mkdir()
    socket_path, _pid_file = chrome_daemon_runner._runtime_paths("a" * 32, runtime)
    daemon_home = socket_path.parent
    daemon_home.mkdir()
    original_rmtree = chrome_daemon_runner.shutil.rmtree
    failed_path = daemon_home if failed_tree == "daemon_home" else profile
    cleaned_path = profile if failed_tree == "daemon_home" else daemon_home

    def remove_tree(path):
        if path == failed_path:
            raise OSError("injected cleanup failure")
        original_rmtree(path)

    with (
        patch.object(
            chrome_daemon_runner,
            "_session_environment",
            return_value=({}, runtime, profile),
        ),
        patch.object(chrome_daemon_runner, "_status", return_value=None),
        patch.object(chrome_daemon_runner.shutil, "rmtree", side_effect=remove_tree),
    ):
        assert chrome_daemon_runner._stop("a" * 32) == {"stopped": True}

    assert failed_path.exists()
    assert not cleaned_path.exists()


def test_stop_cleanup_does_not_swallow_base_exception(tmp_path):
    runtime = tmp_path / "runtime"
    profile = tmp_path / "profile"
    runtime.mkdir()
    profile.mkdir()
    socket_path, _pid_file = chrome_daemon_runner._runtime_paths("a" * 32, runtime)

    with (
        patch.object(
            chrome_daemon_runner,
            "_session_environment",
            return_value=({}, runtime, profile),
        ),
        patch.object(chrome_daemon_runner, "_status", return_value=None),
        patch.object(
            chrome_daemon_runner.shutil,
            "rmtree",
            side_effect=KeyboardInterrupt("stop cleanup"),
        ),
        pytest.raises(KeyboardInterrupt, match="stop cleanup"),
    ):
        chrome_daemon_runner._stop("a" * 32)


def test_private_dir_rejects_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(chrome_daemon_runner.ChromeDaemonRunnerError, match="symlink"):
        chrome_daemon_runner._private_dir(link)


def test_terminate_refuses_unverified_pid(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text(str(os.getpid()))
    with pytest.raises(
        chrome_daemon_runner.ChromeDaemonRunnerError, match="unverified"
    ):
        chrome_daemon_runner._terminate_expected_daemon(pid_file)


def test_terminate_treats_process_exit_before_sigterm_as_success():
    with (
        patch.object(chrome_daemon_runner, "_read_pid", return_value=12345),
        patch.object(
            chrome_daemon_runner, "_pid_is_expected_daemon", return_value=True
        ),
        patch.object(
            chrome_daemon_runner,
            "_kill_process",
            side_effect=ProcessLookupError("already exited"),
        ) as kill,
        patch.object(chrome_daemon_runner, "_wait_for_exit") as wait_for_exit,
    ):
        chrome_daemon_runner._terminate_expected_daemon(Path("unused"))

    kill.assert_called_once_with(12345, chrome_daemon_runner.signal.SIGTERM)
    wait_for_exit.assert_not_called()


def test_terminate_treats_process_exit_before_sigkill_as_success():
    with (
        patch.object(chrome_daemon_runner, "_read_pid", return_value=12345),
        patch.object(
            chrome_daemon_runner, "_pid_is_expected_daemon", return_value=True
        ),
        patch.object(
            chrome_daemon_runner,
            "_kill_process",
            side_effect=[None, ProcessLookupError("already exited")],
        ) as kill,
        patch.object(
            chrome_daemon_runner, "_wait_for_exit", return_value=False
        ) as wait_for_exit,
    ):
        chrome_daemon_runner._terminate_expected_daemon(Path("unused"))

    assert kill.call_args_list == [
        ((12345, chrome_daemon_runner.signal.SIGTERM),),
        ((12345, chrome_daemon_runner.signal.SIGKILL),),
    ]
    wait_for_exit.assert_called_once_with(12345, 3.0)


def test_terminate_propagates_signal_errors_other_than_missing_process():
    with (
        patch.object(chrome_daemon_runner, "_read_pid", return_value=12345),
        patch.object(
            chrome_daemon_runner, "_pid_is_expected_daemon", return_value=True
        ),
        patch.object(
            chrome_daemon_runner,
            "_kill_process",
            side_effect=PermissionError("denied"),
        ),
    ):
        with pytest.raises(PermissionError, match="denied"):
            chrome_daemon_runner._terminate_expected_daemon(Path("unused"))


@pytest.mark.parametrize(
    ("value", "expected_type", "expected"),
    [
        ("W10=", list, []),
        ("e30=", dict, {}),
    ],
)
def test_decode_json_accepts_only_the_expected_shape(value, expected_type, expected):
    assert (
        chrome_daemon_runner._decode_json(value, expected_type=expected_type)
        == expected
    )


@pytest.mark.parametrize("value", ["not-base64", "bm90LWpzb24="])
def test_decode_json_rejects_invalid_payload(value):
    with pytest.raises(chrome_daemon_runner.ChromeDaemonRunnerError):
        chrome_daemon_runner._decode_json(value, expected_type=dict)


def test_write_result_is_tmp_only_and_refuses_symlink(tmp_path):
    with pytest.raises(chrome_daemon_runner.ChromeDaemonRunnerError, match="/tmp"):
        chrome_daemon_runner._write_result(tmp_path / "result.json", {"ok": True})

    target = Path("/tmp") / f"xagent-chrome-target-{os.getpid()}"
    link = Path("/tmp") / f"xagent-chrome-link-{os.getpid()}"
    try:
        target.write_text("keep")
        link.symlink_to(target)
        with pytest.raises(OSError):
            chrome_daemon_runner._write_result(link, {"ok": True})
        assert target.read_text() == "keep"
    finally:
        link.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


def test_main_rejects_nonopaque_session_id_before_operation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chrome_daemon_runner.py",
            "status",
            "--session-id",
            "../../shared",
            "--result-file",
            "/tmp/result.json",
        ],
    )
    with pytest.raises(chrome_daemon_runner.ChromeDaemonRunnerError, match="lowercase"):
        chrome_daemon_runner.main()
