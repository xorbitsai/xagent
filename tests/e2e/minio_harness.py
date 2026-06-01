from __future__ import annotations

import json
import socket
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import fsspec
import pytest
from docker.errors import APIError

from tests.e2e.app_harness import (
    E2EAppClient,
    configure_e2e_app_environment,
    disable_external_app_services,
    init_e2e_db,
    reset_chat_agent_manager,
    run_e2e_app_client,
    seed_registered_local_file,
)
from tests.e2e.scripted_llm import build_scripted_llm_from_json
from xagent.core.file_storage.factory import get_file_storage
from xagent.web.api.auth import hash_password
from xagent.web.models.user import User

MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"


@dataclass(frozen=True)
class MinioStorage:
    bucket: str
    prefix: str
    fs: Any

    def object_bytes(self, key: str) -> bytes:
        with self.fs.open(f"{self.bucket}/{self.prefix}/{key}", "rb") as handle:
            return handle.read()

    def object_info(self, key: str) -> dict[str, Any]:
        bucket_key = f"{self.prefix}/{key}"
        return self.fs.call_s3("head_object", Bucket=self.bucket, Key=bucket_key)

    def put_object(
        self,
        key: str,
        content: bytes | str,
        content_type: str | None = None,
    ) -> None:
        del content_type
        data = content.encode("utf-8") if isinstance(content, str) else content
        parent = f"{self.bucket}/{self.prefix}/{Path(key).parent}"
        self.fs.makedirs(parent, exist_ok=True)
        with self.fs.open(f"{self.bucket}/{self.prefix}/{key}", "wb") as handle:
            handle.write(data)

    def exists(self, key: str) -> bool:
        return bool(self.fs.exists(f"{self.bucket}/{self.prefix}/{key}"))

    def list_keys(self, prefix: str = "") -> list[str]:
        full_prefix = f"{self.bucket}/{self.prefix}"
        if prefix:
            full_prefix = f"{full_prefix}/{prefix.strip('/')}"
        if not self.fs.exists(full_prefix):
            return []
        root = f"{self.bucket}/{self.prefix}/"
        return sorted(
            str(path)[len(root) :]
            for path in self.fs.find(full_prefix)
            if not str(path).rstrip("/").endswith("/")
        )


@dataclass(frozen=True)
class PersistenceApp(E2EAppClient):
    startup_repair_file_id: str
    user_id: int


def run_minio_storage(monkeypatch: pytest.MonkeyPatch) -> Iterator[MinioStorage]:
    if not _docker_available():
        pytest.skip("Requires reachable Docker daemon")

    bucket = f"xagent-e2e-{uuid4().hex}"
    client = _docker_client()
    container = None
    api_port = 0
    for _ in range(5):
        api_port = _free_port()
        console_port = _free_port()
        container_name = f"xagent-minio-e2e-{uuid4().hex[:12]}"
        try:
            container = client.containers.run(
                "minio/minio",  # Docker Hub — more reliable than quay.io
                "server /data --console-address :9001",
                detach=True,
                name=container_name,
                ports={"9000/tcp": api_port, "9001/tcp": console_port},
                tmpfs={"/data": "size=64m"},
                environment={
                    "MINIO_ROOT_USER": MINIO_ACCESS_KEY,
                    "MINIO_ROOT_PASSWORD": MINIO_SECRET_KEY,
                },
            )
            break
        except APIError as exc:
            if "address already in use" not in str(exc):
                raise
            try:
                stale = client.containers.get(container_name)
                stale.remove(force=True)
            except Exception:
                pass
    if container is None:
        pytest.skip("Could not allocate free host ports for MinIO")

    endpoint_url = f"http://127.0.0.1:{api_port}"
    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": endpoint_url, "region_name": "us-east-1"},
        "config_kwargs": {"s3": {"addressing_style": "path"}},
    }

    try:
        deadline = time.monotonic() + 30
        fs = None
        while time.monotonic() < deadline:
            try:
                fs = fsspec.filesystem("s3", **storage_options)
                fs.mkdir(bucket)
                break
            except Exception:
                time.sleep(0.5)
        if fs is None:
            raise RuntimeError("MinIO did not become ready")

        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", f"s3://{bucket}/xagent-test")
        monkeypatch.setenv("XAGENT_FILE_STORAGE_OPTIONS", json.dumps(storage_options))
        get_file_storage.cache_clear()

        yield MinioStorage(bucket=bucket, prefix="xagent-test", fs=fs)
    finally:
        get_file_storage.cache_clear()
        container.remove(force=True)


def run_file_persistence_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    llm_responses_path: Path,
) -> Iterator[PersistenceApp]:
    uploads_dir = configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)

    import xagent.web.api.chat as chat_api

    disable_external_app_services(monkeypatch)
    reset_chat_agent_manager(monkeypatch)
    scripted_llm = build_scripted_llm_from_json(llm_responses_path)
    monkeypatch.setattr(chat_api, "create_default_llm", lambda: scripted_llm)
    monkeypatch.setattr(
        chat_api,
        "resolve_llms_from_names",
        lambda llm_ids, db, user_id=None: (scripted_llm, None, None, scripted_llm),
    )

    SessionLocal = init_e2e_db()
    db = SessionLocal()
    try:
        user = User(username="minio-user", password_hash=hash_password("pw"))
        db.add(user)
        db.commit()
        db.refresh(user)
        seeded_user_id = int(user.id)
        seeded_username = str(user.username)

        repair_file_id = str(uuid4())
        seed_registered_local_file(
            db,
            uploads_dir=uploads_dir,
            file_id=repair_file_id,
            user_id=seeded_user_id,
            filename="startup-repair.txt",
            content="startup repair content\n",
            mime_type="text/plain",
            storage_status="legacy",
        )
    finally:
        db.close()

    with run_e2e_app_client(
        monkeypatch,
        username=seeded_username,
        user_id=seeded_user_id,
    ) as app_client:
        yield PersistenceApp(
            client=app_client.client,
            headers=app_client.headers,
            session_factory=app_client.session_factory,
            token=app_client.token,
            startup_repair_file_id=repair_file_id,
            user_id=seeded_user_id,
        )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _docker_available() -> bool:
    try:
        _docker_client()
        return True
    except Exception:
        return False


def _docker_client() -> Any:
    import docker

    try:
        client = docker.from_env()
        client.ping()
        return client
    except Exception as original_error:
        context_host = _docker_host_from_current_context()
        if context_host is None:
            raise original_error

        client = docker.DockerClient(base_url=context_host)
        client.ping()
        return client


def _docker_host_from_current_context() -> str | None:
    try:
        result = subprocess.run(
            [
                "docker",
                "context",
                "inspect",
                "--format",
                "{{json .Endpoints.docker.Host}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    try:
        host = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return None

    return host if isinstance(host, str) and host.strip() else None
