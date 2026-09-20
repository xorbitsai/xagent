import os
import subprocess
import sys
from pathlib import Path

import yaml
from cryptography.fernet import Fernet

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_backend_image_defaults_to_two_agent_workers_with_persistent_key_path():
    dockerfile = (REPOSITORY_ROOT / "docker/Dockerfile.backend").read_text(
        encoding="utf-8"
    )

    assert "ENV XAGENT_WORKER_COUNT=2" in dockerfile
    assert (
        "ENV XAGENT_ENCRYPTION_KEY_FILE=/run/xagent-secrets/encryption.key"
        in dockerfile
    )


def test_compose_shares_private_key_volume_across_xagent_services():
    compose = yaml.safe_load(
        (REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )

    assert "xagent_secrets" in compose["volumes"]
    for service_name in ("backend", "worker", "scheduler"):
        assert (
            "xagent_secrets:/run/xagent-secrets"
            in compose["services"][service_name]["volumes"]
        )


def _entrypoint_environment(install_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
            "XAGENT_INSTALL_ROOT": str(install_root),
        }
    )
    for name in (
        "DISPLAY",
        "ENCRYPTION_KEY",
        "XAGENT_ENCRYPTION_KEY_FILE",
        "XAGENT_TASK_EXECUTION_ROLE",
    ):
        environment.pop(name, None)
    return environment


def _run_entrypoint(tmp_path: Path, environment: dict[str, str]):
    return subprocess.run(
        [
            "bash",
            str(REPOSITORY_ROOT / "docker/entrypoint.sh"),
            sys.executable,
            "-c",
            'import os; print(os.environ["ENCRYPTION_KEY"]); '
            'print(os.environ.get("DISPLAY", ""))',
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_entrypoint_script_has_valid_bash_syntax():
    subprocess.run(
        ["bash", "-n", str(REPOSITORY_ROOT / "docker/entrypoint.sh")], check=True
    )


def test_entrypoint_bootstraps_and_reuses_key_for_command_override(tmp_path):
    key_path = tmp_path / "secrets" / "encryption.key"
    environment = _entrypoint_environment(tmp_path)
    environment["XAGENT_ENCRYPTION_KEY_FILE"] = str(key_path)

    created = _run_entrypoint(tmp_path, environment)
    reused = _run_entrypoint(tmp_path, environment)

    created_key = created.stdout.splitlines()[0]
    assert reused.stdout.splitlines()[0] == created_key
    assert key_path.read_text(encoding="utf-8").strip() == created_key
    assert "generated a new one" in created.stderr
    assert "Reusing encryption key" in reused.stderr


def test_entrypoint_explicit_key_takes_precedence(tmp_path):
    key_path = tmp_path / "encryption.key"
    file_key = Fernet.generate_key().decode()
    key_path.write_text(file_key, encoding="utf-8")
    explicit_key = Fernet.generate_key().decode()
    environment = _entrypoint_environment(tmp_path)
    environment.update(
        {
            "ENCRYPTION_KEY": explicit_key,
            "XAGENT_ENCRYPTION_KEY_FILE": str(key_path),
        }
    )

    result = _run_entrypoint(tmp_path, environment)

    assert result.stdout.splitlines()[0] == explicit_key
    assert result.stderr == ""
    assert key_path.read_text(encoding="utf-8") == file_key


def test_entrypoint_initializes_display_for_standalone_agent_worker(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    xvfb = fake_bin / "Xvfb"
    xvfb.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    xvfb.chmod(0o755)
    environment = _entrypoint_environment(tmp_path)
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "ENCRYPTION_KEY": Fernet.generate_key().decode(),
            "XAGENT_TASK_EXECUTION_ROLE": "worker",
        }
    )

    result = _run_entrypoint(tmp_path, environment)

    assert result.stdout.splitlines()[1] == ":99"
