"""Tests for scoped tool credential migration backfill."""

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, text


def get_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260602_add_scoped_tool_credentials.py"
    )

    spec = importlib.util.spec_from_file_location(
        "scoped_credentials_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _test_fernet() -> Fernet:
    raw = (
        os.getenv("XAGENT_SECRET_ENCRYPTION_KEY")
        or os.getenv("SECRET_KEY")
        or "xagent-dev-key"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def _encrypt_for_test(value: str) -> str:
    return _test_fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def _decrypt_for_test(ciphertext: str) -> str:
    return _test_fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")


@pytest.fixture
def migration_module():
    return get_migration_module()


@pytest.fixture
def legacy_tool_config_db(tmp_path, migration_module):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE tool_configs (
                id INTEGER PRIMARY KEY,
                tool_name VARCHAR(100) NOT NULL UNIQUE,
                tool_type VARCHAR(20) NOT NULL,
                category VARCHAR(50) NOT NULL,
                display_name VARCHAR(100) NOT NULL,
                description TEXT,
                enabled BOOLEAN DEFAULT TRUE,
                requires_configuration BOOLEAN DEFAULT FALSE NOT NULL,
                config TEXT,
                dependencies TEXT,
                status VARCHAR(20) DEFAULT 'available',
                status_reason VARCHAR(500),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME
            )
        """)
        )
        conn.execute(
            text("""
            INSERT INTO tool_configs (
                tool_name,
                tool_type,
                category,
                display_name,
                config,
                dependencies
            )
            VALUES
                (
                    'web_search',
                    'builtin',
                    'search',
                    'Web Search',
                    :web_search_config,
                    '[]'
                ),
                (
                    'zhipu_web_search',
                    'builtin',
                    'search',
                    'Zhipu Web Search',
                    :zhipu_config,
                    '[]'
                ),
                (
                    'sql_query',
                    'builtin',
                    'database',
                    'SQL Query',
                    :sql_config,
                    '[]'
                ),
                (
                    'browser_use',
                    'builtin',
                    'browser',
                    'Browser Use',
                    :browser_config,
                    '[]'
                )
        """),
            {
                "web_search_config": json.dumps(
                    {
                        "credentials": {
                            "api_key": {
                                "secret": True,
                                "ciphertext": _encrypt_for_test("legacy-google-key"),
                                "masked": "*************-key",
                            },
                            "cse_id": {
                                "secret": False,
                                "value": "legacy-google-cse",
                            },
                            "unknown": {"secret": False, "value": "ignore-me"},
                        }
                    }
                ),
                "zhipu_config": json.dumps(
                    {
                        "credentials": {
                            "api_key": {
                                "secret": True,
                                "ciphertext": _encrypt_for_test("legacy-zhipu-key"),
                                "masked": "************-key",
                            },
                            "base_url": {
                                "secret": False,
                                "value": "https://open.bigmodel.cn",
                            },
                        }
                    }
                ),
                "sql_config": json.dumps(
                    {
                        "credentials": {
                            "analytics": "sqlite:///legacy-analytics.db",
                            "api_key": "not-a-supported-sql-credential",
                        },
                        "sql_connections": {
                            "warehouse": "sqlite:///legacy-warehouse.db",
                        },
                    }
                ),
                "browser_config": json.dumps(
                    {
                        "credentials": {
                            "api_key": "not-a-configurable-tool",
                        }
                    }
                ),
            },
        )

    yield engine
    engine.dispose()


def test_upgrade_backfills_legacy_tool_config_credentials_only(
    legacy_tool_config_db, migration_module
):
    with legacy_tool_config_db.begin() as conn:
        context = MigrationContext.configure(conn)
        operations = Operations(context)
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(migration_module, "op", operations)

            migration_module.upgrade()

    with legacy_tool_config_db.begin() as conn:
        rows = conn.execute(
            text("""
            SELECT scope_type, scope_id, tool_name, field_name, encrypted_value, masked_value
            FROM scoped_tool_credentials
            ORDER BY tool_name, field_name
        """)
        ).mappings()

        credentials = list(rows)

    assert [
        (row["scope_type"], row["scope_id"], row["tool_name"], row["field_name"])
        for row in credentials
    ] == [
        ("instance", None, "web_search", "api_key"),
        ("instance", None, "web_search", "cse_id"),
        ("instance", None, "zhipu_web_search", "api_key"),
        ("instance", None, "zhipu_web_search", "base_url"),
    ]
    assert {
        (row["tool_name"], row["field_name"]): row["masked_value"]
        for row in credentials
    } == {
        ("web_search", "api_key"): "*************-key",
        ("web_search", "cse_id"): "*************-cse",
        ("zhipu_web_search", "api_key"): "************-key",
        ("zhipu_web_search", "base_url"): "********************l.cn",
    }

    decrypted = {
        (row["tool_name"], row["field_name"]): _decrypt_for_test(row["encrypted_value"])
        for row in credentials
    }
    assert decrypted == {
        ("web_search", "api_key"): "legacy-google-key",
        ("web_search", "cse_id"): "legacy-google-cse",
        ("zhipu_web_search", "api_key"): "legacy-zhipu-key",
        ("zhipu_web_search", "base_url"): "https://open.bigmodel.cn",
    }
