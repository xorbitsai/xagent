import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name == "nt", reason="Docker Bash entrypoint")
@pytest.mark.parametrize("alias", [False, True])
def test_docker_tool_key_bootstrap_reuses_persistent_key_or_explicit_alias(
    tmp_path, alias
):
    root = Path(__file__).resolve().parents[2]
    key_file = tmp_path / "secrets/tool.secret"
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
            "XAGENT_INSTALL_ROOT": str(tmp_path),
            "XAGENT_SECRET_ENCRYPTION_KEY_FILE": str(key_file),
            "XAGENT_SECRET_ENCRYPTION_KEY": " ",
            "XAGENT_TASK_EXECUTION_ROLE": "combined",
            "XAGENT_JWT_SECRET": "explicit-jwt-secret-01234567890123456789",
            "ENCRYPTION_KEY": "explicit-avoids-unrelated-bootstrap",
        }
    )
    env.pop("SECRET_KEY", None)
    if alias:
        env["SECRET_KEY"] = "existing-alias-secret-01234567890123456789"
        probe = 'test "$SECRET_KEY" = "existing-alias-secret-01234567890123456789" && test "$XAGENT_SECRET_ENCRYPTION_KEY" = " "'
    else:
        probe = 'test "${#XAGENT_SECRET_ENCRYPTION_KEY}" -ge 32 && test "$XAGENT_SECRET_ENCRYPTION_KEY" = "$(tr -d "\\n" < "$XAGENT_SECRET_ENCRYPTION_KEY_FILE")"'
    first_key = None
    for _ in range(2):
        result = subprocess.run(
            ["bash", str(root / "docker/entrypoint.sh"), "bash", "-c", probe],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        if not alias:
            assert "Tool credential secret at" in result.stderr
            assert "JWT secret" not in result.stderr
            current = key_file.read_text().strip()
            if first_key is None:
                first_key = current
            assert current == first_key
    assert key_file.exists() is not alias
