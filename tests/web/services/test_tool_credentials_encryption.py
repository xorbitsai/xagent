import base64
import hashlib
from unittest.mock import Mock

import pytest
from cryptography.fernet import Fernet

from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.sql_tool import SqlQueryTool
from xagent.web.services import tool_credentials
from xagent.web.services.tool_credentials import _build_fernet_key, _decrypt, _encrypt


def test_development_uses_the_local_fallback(monkeypatch):
    monkeypatch.delenv("XAGENT_SECRET_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("XAGENT_ENVIRONMENT", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("XAGENT_REQUIRE_SECURE_SECRETS", raising=False)

    assert _build_fernet_key() == base64.urlsafe_b64encode(
        hashlib.sha256(b"xagent-dev-key").digest()
    )


@pytest.mark.parametrize("environment", ["production", "staging", "qa"])
def test_deployed_environment_rejects_missing_encryption_key(monkeypatch, environment):
    monkeypatch.delenv("XAGENT_SECRET_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("XAGENT_ENVIRONMENT", environment)

    with pytest.raises(RuntimeError, match="XAGENT_SECRET_ENCRYPTION_KEY"):
        _build_fernet_key()


def test_secure_secret_override_rejects_the_development_fallback(monkeypatch):
    monkeypatch.delenv("XAGENT_SECRET_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "development")
    monkeypatch.setenv("XAGENT_REQUIRE_SECURE_SECRETS", "true")

    with pytest.raises(RuntimeError, match="XAGENT_SECRET_ENCRYPTION_KEY"):
        _build_fernet_key()


def test_configured_encryption_key_is_used(monkeypatch):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.setenv(
        "XAGENT_SECRET_ENCRYPTION_KEY", "deployment-secret-01234567890123456789"
    )
    monkeypatch.delenv("SECRET_KEY", raising=False)

    expected = base64.urlsafe_b64encode(
        hashlib.sha256(b"deployment-secret-01234567890123456789").digest()
    )
    assert _build_fernet_key() == expected
    ciphertext = _encrypt("private credential")
    assert Fernet(expected).decrypt(ciphertext.encode()) == b"private credential"
    assert _decrypt(ciphertext) == "private credential"


@pytest.mark.parametrize("unreadable", ["old-key", "invalid-token", "\ud800", ""])
def test_sql_connections_isolate_unreadable_entries_without_env_substitution(
    monkeypatch, caplog, unreadable
):
    monkeypatch.setenv("XAGENT_SECRET_ENCRYPTION_KEY", "s" * 40)
    if unreadable == "old-key":
        unreadable = Fernet(Fernet.generate_key()).encrypt(b"old-secret").decode()
    prefix = tool_credentials.SQL_CONNECTION_ENV_PREFIX
    monkeypatch.setenv(f"{prefix}BROKEN", "must-not-substitute")
    monkeypatch.setenv(f"{prefix}ENV_ONLY", "environment-url")
    monkeypatch.setenv(f"{prefix}HEALTHY", "overridden-environment-url")
    config = Mock(
        config={
            "sql_connections": {
                "BROKEN": {"ciphertext": unreadable},
                "HEALTHY": {"ciphertext": _encrypt("stored-url")},
            }
        }
    )
    db = Mock()
    db.query.return_value.filter.return_value.first.return_value = config
    connections = tool_credentials.get_sql_connection_map(db, 1)
    assert "BROKEN" not in connections
    assert connections["HEALTHY"] == "stored-url"
    assert connections["ENV_ONLY"] == "environment-url"
    assert tool_credentials.resolve_sql_connection(db, 1, "broken") is None
    assert tool_credentials.resolve_sql_connection(db, 1, "healthy") == "stored-url"
    assert (
        tool_credentials.resolve_sql_connection(db, 1, "env_only") == "environment-url"
    )
    assert "Skipping unreadable stored SQL connection BROKEN" in caplog.text
    assert "old-secret" not in caplog.text
    if unreadable:
        assert unreadable not in caplog.text
    assert "must-not-substitute" not in caplog.text
    tool = SqlQueryTool(connection_map=connections)
    with pytest.raises(ValueError, match="Database connection 'broken' not found"):
        tool.get_database_type("broken")
    with pytest.raises(ValueError, match="Database connection 'broken' not found"):
        tool.execute_sql_query("broken", "SELECT 1")


@pytest.mark.parametrize(
    "resolver", ["get_sql_connection_map", "resolve_sql_connection"]
)
def test_sql_connection_database_errors_still_propagate(resolver):
    db = Mock()
    db.query.side_effect = RuntimeError("database unavailable")
    args = (db, 1, "NAME") if resolver == "resolve_sql_connection" else (db, 1)
    with pytest.raises(RuntimeError, match="database unavailable"):
        getattr(tool_credentials, resolver)(*args)


@pytest.mark.parametrize("source", ["legacy", "standalone", "empty-map"])
def test_sql_environment_access_requires_no_map_or_explicit_inclusion(
    monkeypatch, source
):
    monkeypatch.setenv("XAGENT_EXTERNAL_DB_AVAILABLE", "sqlite:///sql-test.db")
    connections = (
        ToolConfig({}).get_sql_connections()
        if source == "standalone"
        else {}
        if source == "empty-map"
        else None
    )
    tool = SqlQueryTool(connection_map=connections)
    if source == "empty-map":
        with pytest.raises(ValueError, match="not found"):
            tool.get_database_type("available")
    else:
        assert tool.get_database_type("available") == "sqlite"
