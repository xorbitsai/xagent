"""Every E2E application uses the production shared execution path."""

import os
import shutil
import socket
import subprocess
import time
from uuid import uuid4

import pytest
import redis
from cryptography.fernet import Fernet

from tests.e2e.shared_execution_harness import shared_app  # noqa: F401


@pytest.fixture(scope="session")
def shared_redis_url(tmp_path_factory):
    configured = os.getenv("XAGENT_TEST_REDIS_URL")
    process = None
    if configured:
        url = configured
    else:
        executable = shutil.which("redis-server")
        if executable is None:
            pytest.fail("E2E needs XAGENT_TEST_REDIS_URL or a local redis-server")
        directory = tmp_path_factory.mktemp("e2e-redis")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        url = f"redis://127.0.0.1:{port}/0"
        with (directory / "redis.log").open("w") as log:
            process = subprocess.Popen(
                [
                    executable,
                    "--bind",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--save",
                    "",
                    "--appendonly",
                    "no",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    client = redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                client.ping()
                break
            except redis.ConnectionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        yield url
    finally:
        client.close()
        if process is not None:
            process.terminate()
            process.wait(timeout=10)


@pytest.fixture(autouse=True)
def local_execution_unless_selected(monkeypatch, tmp_path, shared_redis_url):
    """Override the legacy-suite fixture: E2E never defaults to local execution."""
    from xagent.core.file_storage import get_unscoped_file_storage
    from xagent.core.utils.encryption import get_cipher

    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "combined")
    monkeypatch.setenv("XAGENT_CHANNEL_INGRESS_ENABLED", "false")
    monkeypatch.setenv("XAGENT_REDIS_URL", shared_redis_url)
    monkeypatch.setenv("XAGENT_TASK_EVENT_CHANNEL_PREFIX", f"e2e:{uuid4().hex}")
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    get_cipher.cache_clear()
    # Each fixture has its own root, shared with fresh worker processes.
    get_unscoped_file_storage.cache_clear()
    yield
    get_unscoped_file_storage.cache_clear()
    get_cipher.cache_clear()
