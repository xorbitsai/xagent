"""Tests for the post-deploy OAuth client readiness check (#3248)."""

import io
import logging
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import encrypt_value, get_cipher
from xagent.web import oauth_readiness
from xagent.web.api.auth import _oauth_env_name
from xagent.web.builtin_mcp_registry import get_builtin_oauth_provider_rows
from xagent.web.models.database import Base
from xagent.web.models.oauth_provider import OAuthProvider

KEY_A = Fernet.generate_key().decode()
KEY_B = Fernet.generate_key().decode()
BUILTIN_PROVIDERS = [row["provider_name"] for row in get_builtin_oauth_provider_rows()]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No ambient client env var or cached cipher leaks into a test."""
    for provider in [*BUILTIN_PROVIDERS, "custom", "google-sandbox"]:
        for suffix in ("CLIENT_ID", "CLIENT_SECRET"):
            monkeypatch.delenv(_oauth_env_name(provider, suffix), raising=False)
    monkeypatch.setenv("ENCRYPTION_KEY", KEY_A)
    get_cipher.cache_clear()
    yield
    get_cipher.cache_clear()


def _row(provider, client_id="", client_secret=""):
    return SimpleNamespace(
        provider_name=provider, client_id=client_id, client_secret=client_secret
    )


def _set_env(monkeypatch, provider, *, client_id=None, client_secret=None):
    if client_id is not None:
        monkeypatch.setenv(_oauth_env_name(provider, "CLIENT_ID"), client_id)
    if client_secret is not None:
        monkeypatch.setenv(_oauth_env_name(provider, "CLIENT_SECRET"), client_secret)


def test_env_only_is_ready_from_env(monkeypatch):
    _set_env(monkeypatch, "zoom", client_id="env-id", client_secret="env-secret")

    result = oauth_readiness.check_provider("zoom", _row("zoom"))

    assert result.ready
    assert (result.client_id, result.client_secret) == ("env", "env")


def test_row_only_is_ready_from_row():
    row = _row("zoom", encrypt_value("row-id"), encrypt_value("row-secret"))

    result = oauth_readiness.check_provider("zoom", row)

    assert result.ready
    assert (result.client_id, result.client_secret) == ("row", "row")


def test_plaintext_row_counts_as_row():
    """Seed migrations copy the env value into the row unencrypted."""
    result = oauth_readiness.check_provider(
        "zoom", _row("zoom", "row-id", "row-secret")
    )

    assert result.ready
    assert (result.client_id, result.client_secret) == ("row", "row")


def test_row_wins_over_env_when_both_are_set(monkeypatch):
    _set_env(monkeypatch, "zoom", client_id="env-id", client_secret="env-secret")
    row = _row("zoom", encrypt_value("row-id"), encrypt_value("row-secret"))

    result = oauth_readiness.check_provider("zoom", row)

    assert result.ready
    assert (result.client_id, result.client_secret) == ("row", "row")


def test_neither_is_not_ready():
    result = oauth_readiness.check_provider("zoom", _row("zoom"))

    assert not result.ready
    assert (result.client_id, result.client_secret) == ("missing", "missing")


def test_missing_secret_alone_is_not_ready(monkeypatch):
    _set_env(monkeypatch, "zoom", client_id="env-id")

    result = oauth_readiness.check_provider("zoom", _row("zoom"))

    assert not result.ready
    assert (result.client_id, result.client_secret) == ("env", "missing")


def test_row_encrypted_under_another_key_is_unreadable(monkeypatch):
    """The runtime would send the ciphertext as the secret; env cannot rescue it."""
    secret_token = encrypt_value("row-secret")
    monkeypatch.setenv("ENCRYPTION_KEY", KEY_B)
    get_cipher.cache_clear()
    _set_env(monkeypatch, "zoom", client_id="env-id", client_secret="env-secret")

    result = oauth_readiness.check_provider("zoom", _row("zoom", "", secret_token))

    assert not result.ready
    assert (result.client_id, result.client_secret) == ("env", "unreadable")


def test_provider_without_row_is_not_ready(monkeypatch):
    _set_env(monkeypatch, "zoom", client_id="env-id", client_secret="env-secret")

    result = oauth_readiness.check_provider("zoom", None)

    assert not result.ready
    assert not result.has_row


def test_provider_set_is_builtin_union_rows_minus_excluded():
    rows = [_row("google"), _row("google-sandbox")]

    results = oauth_readiness.check_providers(
        rows, builtin_providers=["google", "zoom"], exclude={"zoom"}
    )

    assert [r.provider for r in results] == ["google", "google-sandbox"]


def test_unknown_exclusion_is_rejected():
    with pytest.raises(ValueError, match="zooom"):
        oauth_readiness.check_providers(
            [], builtin_providers=["google", "zoom"], exclude={"zooom"}
        )


def test_format_reports_status_and_source_only():
    ready = oauth_readiness.ProviderReadiness("google", True, "row", "env")
    missing = oauth_readiness.ProviderReadiness("zoom", True, "env", "missing")
    no_row = oauth_readiness.ProviderReadiness("jira", False, "env", "env")

    assert (
        oauth_readiness.format_result(ready)
        == "PASS oauth-client:google: client_id=row client_secret=env"
    )
    assert (
        oauth_readiness.format_result(missing)
        == "FAIL oauth-client:zoom: client_id=env client_secret=missing"
    )
    assert oauth_readiness.format_result(no_row) == (
        "FAIL oauth-client:jira: no provider row"
    )


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "xagent.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine, tables=[OAuthProvider.__table__])
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _add_provider(db, provider, client_id="", client_secret=""):
    db.add(
        OAuthProvider(
            provider_name=provider,
            name=provider,
            client_id=client_id,
            client_secret=client_secret,
            auth_url="https://provider.example/authorize",
            token_url="https://provider.example/token",
        )
    )


def _run(argv=()):
    out = io.StringIO()
    code = oauth_readiness.main(list(argv), out=out)
    return code, out.getvalue()


def test_main_fails_only_the_provider_whose_secret_is_unset(
    database, monkeypatch, caplog
):
    for index, provider in enumerate(BUILTIN_PROVIDERS):
        _add_provider(database, provider)
        _set_env(
            monkeypatch,
            provider,
            client_id=f"distinctive-client-id-{index}",
            client_secret=f"distinctive-client-secret-{index}",
        )
    database.commit()
    monkeypatch.delenv(_oauth_env_name("zoom", "CLIENT_SECRET"))

    with caplog.at_level(logging.DEBUG):
        code, output = _run()

    assert code == 1
    lines = output.splitlines()
    assert "FAIL oauth-client:zoom: client_id=env client_secret=missing" in lines
    passed = {line.split(":")[1] for line in lines if line.startswith("PASS ")}
    assert passed == set(BUILTIN_PROVIDERS) - {"zoom"}
    assert lines[-1] == f"{len(BUILTIN_PROVIDERS)} checks, 1 failed"
    for text in (output, caplog.text):
        assert "distinctive" not in text


def test_main_never_prints_row_values(database, monkeypatch, caplog):
    for provider in BUILTIN_PROVIDERS:
        _add_provider(
            database,
            provider,
            encrypt_value("distinctive-row-id"),
            encrypt_value("distinctive-row-secret"),
        )
    database.commit()
    # Rows written under KEY_A, read under KEY_B: every field is unreadable.
    monkeypatch.setenv("ENCRYPTION_KEY", KEY_B)
    get_cipher.cache_clear()

    with caplog.at_level(logging.DEBUG):
        code, output = _run()

    assert code == 1
    assert "FAIL oauth-client:zoom: client_id=unreadable client_secret=unreadable" in (
        output
    )
    for text in (output, caplog.text):
        assert "distinctive" not in text
        assert "gAAAA" not in text  # no Fernet token fragment either


def test_main_passes_when_every_provider_is_ready(database, monkeypatch):
    for provider in BUILTIN_PROVIDERS:
        _add_provider(database, provider, "row-id", "row-secret")
    database.commit()

    code, output = _run()

    assert code == 0
    assert output.splitlines()[-1] == f"{len(BUILTIN_PROVIDERS)} checks, 0 failed"


def test_main_excludes_providers(database, monkeypatch):
    for provider in BUILTIN_PROVIDERS:
        if provider not in {"zoom", "xero"}:
            _add_provider(database, provider, "row-id", "row-secret")
    database.commit()

    code, output = _run(["--exclude", "zoom", "--exclude", "xero"])

    assert code == 0
    assert "zoom" not in output and "xero" not in output


def test_main_rejects_an_unknown_exclusion(database):
    with pytest.raises(SystemExit) as exc:
        _run(["--exclude", "zooom"])
    assert exc.value.code == 2


def test_main_reports_a_database_failure_by_class_only(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'missing.db'}")

    code, output = _run()

    assert code == 1
    assert output.splitlines() == ["FAIL database: OperationalError"]


@pytest.mark.parametrize("key", [None, "not-a-fernet-key"])
def test_main_fails_once_when_this_process_has_no_usable_key(
    database, monkeypatch, key
):
    """`compose exec` skips the entrypoint that loads ENCRYPTION_KEY from its
    file. The server refuses to start without a valid key in production, so
    this process does not share the server's environment: say so once rather
    than reporting every encrypted row as unreadable."""
    for provider in BUILTIN_PROVIDERS:
        _add_provider(database, provider, encrypt_value("row-id"), "row-secret")
    database.commit()
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    if key is None:
        monkeypatch.delenv("ENCRYPTION_KEY")
    else:
        monkeypatch.setenv("ENCRYPTION_KEY", key)
    get_cipher.cache_clear()

    code, output = _run()

    assert code == 1
    assert output.splitlines() == ["FAIL encryption-key: ValueError"]
