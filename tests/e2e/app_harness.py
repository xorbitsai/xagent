from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import jwt
import pytest
from fastapi.testclient import TestClient

from tests.shared.db_teardown import drop_all_tables
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models.database import get_engine, get_session_local, init_db
from xagent.web.models.uploaded_file import UploadedFile


@dataclass(frozen=True)
class E2EAppClient:
    client: TestClient
    headers: dict[str, str]
    session_factory: Any
    token: str


@dataclass(frozen=True)
class SeededLocalFile:
    file_id: str
    path: Path
    filename: str


@dataclass(frozen=True)
class E2EUser:
    id: int
    username: str


class _DisabledTelegramChannel:
    enabled = False

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class _DisabledFeishuChannel:
    enabled = False

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class _DisabledSlackChannel:
    enabled = False

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def build_access_token(
    *,
    username: str,
    user_id: int,
    expires_at: datetime | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": username,
            "user_id": user_id,
            "type": "access",
            "exp": expires_at or now + timedelta(hours=1),
            "iat": now,
        },
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def configure_e2e_app_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
) -> Path:
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized"))

    db_path = tmp_path / "e2e.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "lancedb"))
    monkeypatch.setenv("LANCEDB_PATH", str(tmp_path / "lancedb-path"))
    monkeypatch.setenv("LANCEDB_AUTO_MIGRATE", "false")
    return uploads_dir


def disable_external_app_services(monkeypatch: pytest.MonkeyPatch) -> None:
    import xagent.web.sandbox_manager as sandbox_manager

    monkeypatch.setattr(sandbox_manager, "get_sandbox_manager", lambda: None)
    _patch_channel_modules_disabled(monkeypatch)


def reset_chat_agent_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    import xagent.web.api.chat as chat_api

    monkeypatch.setattr(chat_api, "_global_agent_manager", None)


def init_e2e_db() -> Any:
    init_db()
    return get_session_local()


def create_e2e_user(
    db: Any,
    *,
    username: str,
    password_hash: str = "hash",
) -> E2EUser:
    from xagent.web.models.user import User

    user = User(username=username, password_hash=password_hash)
    db.add(user)
    db.commit()
    db.refresh(user)
    return E2EUser(id=int(user.id), username=str(user.username))


def seed_registered_local_file(
    db: Any,
    *,
    uploads_dir: Path,
    user_id: int,
    filename: str,
    content: bytes | str,
    file_id: str,
    relative_dir: str | None = None,
    task_id: int | None = None,
    mime_type: str | None = None,
    storage_backend: str | None = None,
    storage_key: str | None = None,
    storage_uri: str | None = None,
    checksum: str | None = None,
    etag: str | None = None,
    storage_status: str = "legacy",
) -> SeededLocalFile:
    base_dir = uploads_dir / f"user_{user_id}"
    if relative_dir:
        base_dir = base_dir / relative_dir
    path = base_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")

    file_record = UploadedFile(
        file_id=file_id,
        user_id=user_id,
        task_id=task_id,
        filename=filename,
        storage_path=str(path),
        storage_backend=storage_backend,
        storage_key=storage_key,
        storage_uri=storage_uri,
        checksum=checksum,
        etag=etag,
        mime_type=mime_type,
        file_size=path.stat().st_size,
        storage_status=storage_status,
    )
    db.add(file_record)
    db.commit()
    return SeededLocalFile(file_id=file_id, path=path, filename=filename)


@contextmanager
def run_e2e_app_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    username: str,
    user_id: int,
    configure_app_module: Callable[[ModuleType], None] | None = None,
) -> Iterator[E2EAppClient]:
    from xagent.providers.vector_store.lancedb import clear_connection_cache

    token = build_access_token(username=username, user_id=user_id)

    clear_connection_cache()
    app_module = importlib.import_module("xagent.web.app")
    app_module = importlib.reload(app_module)
    if configure_app_module is not None:
        configure_app_module(app_module)

    try:
        with TestClient(app_module.app) as client:
            yield E2EAppClient(
                client=client,
                headers={"Authorization": f"Bearer {token}"},
                session_factory=get_session_local(),
                token=token,
            )
    finally:
        reset_chat_agent_manager(monkeypatch)
        clear_connection_cache()
        try:
            drop_all_tables(get_engine())
            get_engine().dispose()
        except RuntimeError:
            pass
        get_unscoped_file_storage.cache_clear()


def _patch_channel_modules_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    telegram_module = ModuleType("xagent.web.channels.telegram.bot")
    telegram_module.get_telegram_channel = lambda: _DisabledTelegramChannel()
    monkeypatch.setitem(
        sys.modules, "xagent.web.channels.telegram.bot", telegram_module
    )

    feishu_module = ModuleType("xagent.web.channels.feishu.bot")
    feishu_module.get_feishu_channel = lambda: _DisabledFeishuChannel()
    monkeypatch.setitem(sys.modules, "xagent.web.channels.feishu.bot", feishu_module)

    slack_module = ModuleType("xagent.web.channels.slack.bot")
    slack_module.get_slack_channel = lambda: _DisabledSlackChannel()
    monkeypatch.setitem(sys.modules, "xagent.web.channels.slack.bot", slack_module)
