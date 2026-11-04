import base64
import hashlib
from unittest.mock import Mock

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import JSON, Column, Integer, MetaData, Table, create_engine, select
from sqlalchemy.orm import Session

from xagent.web.models.database import Base
from xagent.web.models.tool_config import ToolConfig, UserToolConfig
from xagent.web.tool_credentials_key_migration import migrate_tool_credentials


def cipher(raw):
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(raw.encode()).digest()))


@pytest.mark.parametrize(
    "destination",
    [
        "",
        "short-key",
        "xagent-dev-key",
        "your-secret-key-change-in-production",
        "change-me-to-a-random-session-secret",
        "a-separate-private-32-or-more-character-tool-key",
    ],
)
def test_migration_refuses_weak_destination_even_in_development(
    monkeypatch, destination
):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "development")
    monkeypatch.setenv("XAGENT_SECRET_ENCRYPTION_KEY", destination)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    db = Mock()
    with pytest.raises(RuntimeError, match="32\\+ character destination"):
        migrate_tool_credentials(db, "old-key")
    db.begin.assert_not_called()


@pytest.mark.parametrize(
    "old_key",
    [
        "xagent-dev-key",
        "short-key",
        "your-secret-key-change-in-production",
    ],
)
@pytest.mark.parametrize("corrupt", [False, True])
def test_offline_migration_is_atomic_and_idempotent(
    tmp_path, monkeypatch, old_key, corrupt
):
    new_key = "new-private-key-01234567890123456789"
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.setenv("XAGENT_SECRET_ENCRYPTION_KEY", new_key)
    engine = create_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    Base.metadata.create_all(engine)
    original = cipher(old_key).encrypt(b"api-secret").decode()
    sql = (
        "invalid"
        if corrupt
        else cipher(old_key).encrypt(b"postgresql://secret").decode()
    )
    try:
        with Session(engine) as db:
            db.add(
                ToolConfig(
                    tool_name="tavily_web_search",
                    tool_type="builtin",
                    category="search",
                    display_name="Search",
                    config={
                        "credentials": {
                            "api_key": {
                                "secret": True,
                                "ciphertext": original,
                                "masked": "***",
                            }
                        },
                        "untouched": 1,
                    },
                )
            )
            db.add(
                UserToolConfig(
                    user_id=1,
                    tool_name="sql_query",
                    config={"sql_connections": {"MAIN": {"ciphertext": sql}}},
                )
            )
            db.commit()
            if corrupt:
                with pytest.raises(RuntimeError, match="migration rolled back"):
                    migrate_tool_credentials(db, old_key)
            else:
                assert migrate_tool_credentials(db, old_key) == 2
                assert migrate_tool_credentials(db, old_key) == 0
            result = db.query(ToolConfig).one().config
            assert result["untouched"] == 1
            assert result["credentials"]["api_key"]["masked"] == "***"
            if corrupt:
                assert result["credentials"]["api_key"]["ciphertext"] == original
            else:
                assert (
                    cipher(new_key).decrypt(
                        result["credentials"]["api_key"]["ciphertext"].encode()
                    )
                    == b"api-secret"
                )
                assert (
                    cipher(new_key).decrypt(
                        db.query(UserToolConfig)
                        .one()
                        .config["sql_connections"]["MAIN"]["ciphertext"]
                        .encode()
                    )
                    == b"postgresql://secret"
                )
    finally:
        engine.dispose()


def test_offline_tool_migration_accepts_pre_upgrade_schema(tmp_path, monkeypatch):
    """Rotate before Alembic adds current ORM columns or per-user configs."""
    old_key, new_key = "xagent-dev-key", "new-private-tool-key-01234567890123456789"
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.setenv("XAGENT_SECRET_ENCRYPTION_KEY", new_key)
    engine = create_engine(f"sqlite:///{tmp_path / 'old-tool-schema.db'}")
    table = Table(
        "tool_configs",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("config", JSON),
    )
    table.create(engine)
    token = cipher(old_key).encrypt(b"tool-secret").decode()
    try:
        with engine.begin() as conn:
            conn.execute(
                table.insert().values(
                    id=1,
                    config={"credentials": {"api_key": {"ciphertext": token}}},
                )
            )
        with Session(engine) as db:
            assert migrate_tool_credentials(db, old_key) == 1
            assert migrate_tool_credentials(db, old_key) == 0
        with engine.connect() as conn:
            payload = conn.scalar(select(table.c.config))
            assert (
                cipher(new_key).decrypt(
                    payload["credentials"]["api_key"]["ciphertext"].encode()
                )
                == b"tool-secret"
            )
    finally:
        engine.dispose()
