import asyncio
import os
from unittest.mock import Mock

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import (
    JSON,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
    select,
)
from sqlalchemy.exc import OperationalError

from xagent.config import DEV_FALLBACK_ENCRYPTION_KEY
from xagent.web import (
    encryption_key_migration,
    tool_credentials_key_migration,
    worker,
    worker_pool,
)
from xagent.web.services import encryption_key_rotation as rotation
from xagent.web.services.tool_credentials import _build_fernet_key, _decrypt


@pytest.fixture
def database(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'rotate.db'}")
    monkeypatch.setattr(rotation, "get_engine", lambda: engine)
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv(
        "XAGENT_SECRET_ENCRYPTION_KEY", "new-private-tool-key-01234567890123456789"
    )
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "table_name,group",
    [("tool_configs", "credentials"), ("user_tool_configs", "sql_connections")],
)
def test_unreadable_tool_data_blocks_admission(
    database, monkeypatch, table_name, group
):
    table = Table(
        table_name,
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("config", JSON),
    )
    table.create(database)
    ciphertext = (
        Fernet(_build_fernet_key("xagent-dev-key")).encrypt(b"do-not-log").decode()
    )
    with database.begin() as conn:
        conn.execute(
            table.insert().values(
                id=7, config={group: {"api_key": {"ciphertext": ciphertext}}}
            )
        )
    with pytest.raises(RuntimeError) as error:
        rotation.validate_stored_encryption_keys()
    assert f"{table_name} row 7 field {group}" in str(error.value)
    assert "docs/tool-credential-key-upgrade.md" in str(error.value)
    assert ciphertext not in str(error.value) and "do-not-log" not in str(error.value)
    with pytest.raises(RuntimeError, match="Stored tool credential is unreadable"):
        _decrypt(ciphertext)


def test_published_master_key_rotation_preserves_raw_columns_and_json(database):
    old = Fernet(DEV_FALLBACK_ENCRYPTION_KEY.encode())
    new = Fernet(os.environ["ENCRYPTION_KEY"].encode())
    meta = MetaData()
    model = Table(
        "models",
        meta,
        Column("id", Integer, primary_key=True),
        Column("_api_key_encrypted", Text),
    )
    channel = Table(
        "user_channels",
        meta,
        Column("id", Integer, primary_key=True),
        Column("config", JSON),
    )
    oauth = Table(
        "user_oauth",
        meta,
        Column("id", Integer, primary_key=True),
        Column("access_token", Text),
        Column("refresh_token", Text),
    )
    meta.create_all(database)

    def encrypted(value):
        return old.encrypt(value.encode()).decode()

    with database.begin() as conn:
        conn.execute(
            model.insert().values(id=1, _api_key_encrypted=encrypted("model-key"))
        )
        conn.execute(
            channel.insert().values(
                id=1,
                config={
                    "bot_token": encrypted("bot-secret"),
                    "channel_id": "unchanged",
                },
            )
        )
        conn.execute(
            oauth.insert().values(
                id=1,
                access_token=encrypted("access"),
                refresh_token=encrypted("refresh"),
            )
        )
    with pytest.raises(RuntimeError, match="models row 1 field _api_key_encrypted"):
        rotation.validate_stored_encryption_keys()
    with database.begin() as conn:
        assert rotation.rotate_encryption_key(conn, old, new) == 4
    rotation.validate_stored_encryption_keys()
    with database.begin() as conn:
        assert rotation.rotate_encryption_key(conn, old, new) == 0
        assert (
            new.decrypt(conn.scalar(select(model.c._api_key_encrypted)).encode())
            == b"model-key"
        )
        config = conn.scalar(select(channel.c.config))
        assert config["channel_id"] == "unchanged"
        assert new.decrypt(config["bot_token"].encode()) == b"bot-secret"
        assert (
            new.decrypt(conn.scalar(select(oauth.c.access_token)).encode()) == b"access"
        )
        assert (
            new.decrypt(conn.scalar(select(oauth.c.refresh_token)).encode())
            == b"refresh"
        )


def test_rotation_rolls_back_every_field_if_one_is_unreadable(database):
    table = Table(
        "models",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("_api_key_encrypted", Text),
    )
    table.create(database)
    old, new = Fernet.generate_key(), Fernet.generate_key()
    good = Fernet(old).encrypt(b"private-value").decode()
    with database.begin() as conn:
        conn.execute(
            table.insert(),
            [
                {"id": 1, "_api_key_encrypted": good},
                {"id": 2, "_api_key_encrypted": "corrupt"},
            ],
        )
    with pytest.raises(RuntimeError, match="row 2 field _api_key_encrypted"):
        with database.begin() as conn:
            rotation.rotate_encryption_key(conn, Fernet(old), Fernet(new))
    with database.connect() as conn:
        assert (
            conn.scalar(select(table.c._api_key_encrypted).where(table.c.id == 1))
            == good
        )


@pytest.mark.parametrize("column", ["env", "auth"])
def test_nested_mcp_fields_rotate_without_touching_plaintext(database, column):
    old, new = Fernet(Fernet.generate_key()), Fernet(Fernet.generate_key())
    table = Table(
        "mcp_servers",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column(column, JSON),
    )
    table.create(database)
    with database.begin() as conn:
        conn.execute(
            table.insert().values(
                id=1, **{column: {"nested": [old.encrypt(b"secret").decode(), "plain"]}}
            )
        )
        assert rotation.rotate_encryption_key(conn, old, new) == 1
        values = conn.scalar(select(table.c[column]))["nested"]
        assert new.decrypt(values[0].encode()) == b"secret"
        assert values[1] == "plain"


def test_worker_entrypoints_validate_before_admitting(monkeypatch):
    fail = Mock(side_effect=RuntimeError("invalid encryption config"))
    for module in (worker, worker_pool):
        monkeypatch.setattr(module, "validate_encryption_config", fail)
    with pytest.raises(RuntimeError, match="invalid encryption config"):
        asyncio.run(worker.run_worker())
    with pytest.raises(RuntimeError, match="invalid encryption config"):
        worker_pool.run_combined_worker_pool(
            worker_count=2, host="localhost", port=8000, log_level=None
        )
    assert fail.call_count == 2


def test_standalone_worker_refuses_unreadable_storage_before_runtime(monkeypatch):
    for name in (
        "validate_encryption_config",
        "validate_task_execution_host_config",
        "configure_db",
        "validate_worker_schema",
    ):
        monkeypatch.setattr(worker, name, Mock())
    monkeypatch.setattr(worker, "get_shared_task_execution_enabled", lambda: True)
    monkeypatch.setattr(worker, "get_task_execution_role", lambda: "worker")
    runtime = Mock()
    monkeypatch.setattr(worker, "validate_interaction_rollout_at_startup", runtime)
    monkeypatch.setattr(
        rotation,
        "validate_stored_encryption_keys",
        Mock(side_effect=RuntimeError("unreadable tool credential")),
    )
    with pytest.raises(RuntimeError, match="unreadable tool credential"):
        asyncio.run(worker.run_worker())
    runtime.assert_not_called()


def test_global_authority_verifier_rotates_with_master(database):
    from xagent.web.services.global_memory_embedding_authority import (
        _credential_verifier,
    )

    old = Fernet(DEV_FALLBACK_ENCRYPTION_KEY.encode())
    new = Fernet(os.environ["ENCRYPTION_KEY"].encode())
    table = Table(
        "global_memory_embedding_authority",
        MetaData(),
        Column("authority_key", String, primary_key=True),
        Column("api_key_encrypted", Text),
        Column("credential_verifier", String),
    )
    table.create(database)
    with database.begin() as conn:
        conn.execute(
            table.insert().values(
                authority_key="global",
                api_key_encrypted=old.encrypt(b"embedding-secret").decode(),
                credential_verifier="old-verifier",
            )
        )
        assert rotation.rotate_encryption_key(conn, old, new) == 1
        assert conn.scalar(select(table.c.credential_verifier)) == _credential_verifier(
            "embedding-secret"
        )


def test_celery_disposes_pre_fork_database_pool_on_storage_failure(monkeypatch):
    from xagent.web.jobs import celery_app
    from xagent.web.models import database as db_module

    engine = Mock()
    monkeypatch.setattr(celery_app, "validate_encryption_config", Mock())
    monkeypatch.setattr(db_module, "configure_db", Mock())
    monkeypatch.setattr(db_module, "get_engine", lambda: engine)
    monkeypatch.setattr(
        rotation,
        "validate_stored_encryption_keys",
        Mock(side_effect=RuntimeError("unreadable stored key")),
    )
    with pytest.raises(SystemExit, match="unreadable stored key"):
        celery_app.validate_worker_encryption()
    engine.dispose.assert_called_once()


@pytest.mark.parametrize("unreadable", [False, True])
def test_web_initializes_engine_before_guard_and_checks_before_migrations(
    tmp_path, monkeypatch, unreadable
):
    from xagent.web import app as web_app
    from xagent.web.models import database as db_module

    url = f"sqlite:///{tmp_path / 'web-startup.db'}"
    key = Fernet.generate_key()
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.setenv("ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("XAGENT_SECRET_ENCRYPTION_KEY", "s" * 40)
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    table = Table(
        "models",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("_api_key_encrypted", Text),
    )
    seed_engine = create_engine(url)
    table.create(seed_engine)
    stored_key = Fernet.generate_key() if unreadable else key
    ciphertext = Fernet(stored_key).encrypt(b"model-secret").decode()
    with seed_engine.begin() as conn:
        conn.execute(table.insert().values(id=1, _api_key_encrypted=ciphertext))
    seed_engine.dispose()
    schema = Mock(return_value=[])
    monkeypatch.setattr(db_module, "_initialize_database_schema", schema)
    # Stop after database preparation; do not start any runtime/background work.
    admission = Mock(side_effect=RuntimeError("reached host admission"))
    monkeypatch.setattr(web_app, "run_host_startup_admissions", admission)
    expected = (
        "Unreadable encrypted credential" if unreadable else "reached host admission"
    )
    try:
        with pytest.raises(RuntimeError, match=expected):
            asyncio.run(web_app._initialize_database_and_admit_runtime(web_app.app))
        if unreadable:
            schema.assert_not_called()
            admission.assert_not_called()
        else:
            schema.assert_called_once_with(db_module.get_engine())
        with db_module.get_engine().connect() as conn:
            assert conn.scalar(select(table.c._api_key_encrypted)) == ciphertext
    finally:
        if db_module._engine is not None:
            db_module._engine.dispose()


@pytest.mark.parametrize(
    "key",
    [
        DEV_FALLBACK_ENCRYPTION_KEY,
        DEV_FALLBACK_ENCRYPTION_KEY.replace("_", "/"),
        DEV_FALLBACK_ENCRYPTION_KEY + "=",
    ],
)
def test_master_cli_rejects_equivalent_public_destination_before_database(
    monkeypatch, key
):
    monkeypatch.setenv(
        "XAGENT_ENCRYPTION_MIGRATION_KEY", Fernet.generate_key().decode()
    )
    monkeypatch.setenv("ENCRYPTION_KEY", key)
    configure = Mock()
    monkeypatch.setattr(encryption_key_migration, "configure_db", configure)
    with pytest.raises(SystemExit, match="private Fernet key"):
        encryption_key_migration.main()
    configure.assert_not_called()


def test_fresh_production_database_initializes_with_real_schema(tmp_path, monkeypatch):
    from xagent.web.models import database as db_module

    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("XAGENT_SECRET_ENCRYPTION_KEY", "s" * 40)
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    try:
        db_module.init_db(f"sqlite:///{tmp_path / 'fresh-production.db'}")
        tables = set(inspect(db_module.get_engine()).get_table_names())
        assert set(rotation.ENCRYPTED_COLUMNS) <= tables
        rotation.validate_stored_encryption_keys()
    finally:
        if db_module._engine is not None:
            db_module._engine.dispose()


def test_celery_worker_and_beat_abort_on_invalid_key(monkeypatch):
    from celery.signals import beat_init, worker_init

    from xagent.web.jobs import celery_app

    monkeypatch.setattr(
        celery_app,
        "validate_encryption_config",
        Mock(side_effect=RuntimeError("invalid key")),
    )
    for signal in (worker_init, beat_init):
        assert any(
            receiver() is celery_app.validate_worker_encryption
            for _, receiver in signal.receivers
        )
    with pytest.raises(SystemExit, match="Refusing Celery startup: invalid key"):
        celery_app.validate_worker_encryption()


@pytest.mark.parametrize(
    "module,key",
    [
        (tool_credentials_key_migration, "XAGENT_TOOL_CREDENTIALS_MIGRATION_KEY"),
        (encryption_key_migration, "XAGENT_ENCRYPTION_MIGRATION_KEY"),
    ],
)
def test_migration_cli_missing_source_fails_before_database(monkeypatch, module, key):
    monkeypatch.delenv(key, raising=False)
    configure = Mock()
    monkeypatch.setattr(module, "configure_db", configure)
    with pytest.raises(SystemExit, match=key):
        module.main()
    configure.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("operator-safe"),
        OperationalError("private-sql", {}, Exception("private-host")),
    ],
)
@pytest.mark.parametrize(
    "module", [tool_credentials_key_migration, encryption_key_migration]
)
def test_migration_cli_sanitizes_database_failures(monkeypatch, error, module):
    monkeypatch.setenv("XAGENT_TOOL_CREDENTIALS_MIGRATION_KEY", "old")
    monkeypatch.setenv("XAGENT_ENCRYPTION_MIGRATION_KEY", DEV_FALLBACK_ENCRYPTION_KEY)
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(module, "configure_db", Mock(side_effect=error))
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert "private" not in str(exc.value)


def test_master_cli_rotates_without_starting_runtime(database, monkeypatch, capsys):
    table = Table(
        "models",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("_api_key_encrypted", Text),
    )
    table.create(database)
    with database.begin() as conn:
        conn.execute(
            table.insert().values(
                id=1,
                _api_key_encrypted=Fernet(DEV_FALLBACK_ENCRYPTION_KEY.encode())
                .encrypt(b"model-secret")
                .decode(),
            )
        )
    monkeypatch.setenv("XAGENT_ENCRYPTION_MIGRATION_KEY", DEV_FALLBACK_ENCRYPTION_KEY)
    monkeypatch.setattr(encryption_key_migration, "configure_db", Mock())
    monkeypatch.setattr(encryption_key_migration, "get_engine", lambda: database)
    encryption_key_migration.main()
    output = capsys.readouterr()
    assert "Re-encrypted 1" in output.out
    assert "model-secret" not in output.out + output.err
    rotation.validate_stored_encryption_keys()
