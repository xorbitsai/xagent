import asyncio
import base64
import json
import os
import time
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
from xagent.core.tools.core.mcp.model import SENSITIVE_AUTH_FIELDS
from xagent.core.utils.encryption import get_cipher
from xagent.web import (
    encryption_key_migration,
    tool_credentials_key_migration,
    worker,
    worker_pool,
)
from xagent.web.models.trigger import AgentTrigger
from xagent.web.models.user_channel import CHANNEL_SECRET_FIELDS
from xagent.web.services import encryption_key_rotation as rotation
from xagent.web.services.tool_credentials import _build_fernet_key, _decrypt
from xagent.web.services.trigger_providers.base import CallbackRequestContext
from xagent.web.services.trigger_providers.webhook import (
    WebhookProvider,
    sign_webhook_payload,
)
from xagent.web.services.triggers import _payload_snapshot


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


@pytest.mark.parametrize(
    "table_name,column",
    [
        ("mcp_servers", "env"),
        ("user_mcpservers", "env"),
        ("actor_mcp_server_connections", "encrypted_env"),
        ("custom_apis", "env"),
    ],
)
def test_env_rotation_matches_flat_writer_contract(database, table_name, column):
    old = Fernet(Fernet.generate_key())
    new = Fernet(os.environ["ENCRYPTION_KEY"].encode())
    table = Table(
        table_name,
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column(column, JSON),
    )
    table.create(database)
    unrelated = Fernet(Fernet.generate_key()).encrypt(b"not-a-stored-secret").decode()
    payload = {
        "API_KEY": old.encrypt(b"secret").decode(),
        "PLAIN": "legacy-plaintext",
        "EMPTY": "",
        "nested": {"API_KEY": unrelated},
        "list": [unrelated],
        "number": 7,
        "none": None,
    }
    with database.begin() as conn:
        conn.execute(table.insert().values(id=1, **{column: payload}))
    with pytest.raises(RuntimeError, match=f"{table_name} row 1 field {column}"):
        rotation.validate_stored_encryption_keys()
    with database.begin() as conn:
        assert rotation.rotate_encryption_key(conn, old, new) == 1
        values = conn.scalar(select(table.c[column]))
        assert new.decrypt(values.pop("API_KEY").encode()) == b"secret"
        assert values == {
            key: value for key, value in payload.items() if key != "API_KEY"
        }
        assert rotation.rotate_encryption_key(conn, old, new) == 0
    rotation.validate_stored_encryption_keys()


@pytest.mark.parametrize("token_kind", ["shaped", "foreign_key", "source_key"])
@pytest.mark.parametrize(
    "table_name,column",
    [
        ("trigger_runs", "payload_snapshot"),
        ("user_channels", "config"),
        ("mcp_servers", "auth"),
    ],
)
def test_token_shaped_plaintext_is_never_scanned_or_rotated(
    database, table_name, column, token_kind
):
    old = Fernet(Fernet.generate_key())
    new = Fernet(os.environ["ENCRYPTION_KEY"].encode())
    token = {
        "shaped": base64.urlsafe_b64encode(b"\x80" + b"\0" * 72).decode(),
        "foreign_key": Fernet(Fernet.generate_key()).encrypt(b"plaintext-id").decode(),
        "source_key": old.encrypt(b"plaintext-id").decode(),
    }[token_kind]
    payloads = {
        "trigger_runs": {
            "metadata": {
                "source_event_id": token,
                "event_type": token,
                "resource_id": token,
            },
            "payload_sha256": token,
        },
        "user_channels": {"channel_id": token},
        "mcp_servers": {"type": token, "api_key_name": token},
    }
    # Even familiar secret field names are not encrypted when nested below an
    # unrelated key; the real writers only encrypt explicitly declared paths.
    payload = {
        **payloads[table_name],
        "nested": {
            "encrypted_payload": token,
            "bot_token": token,
            "access_token": token,
        },
        "items": [token],
    }
    table = Table(
        table_name,
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column(column, JSON),
    )
    table.create(database)
    with database.begin() as conn:
        conn.execute(table.insert().values(id=1, **{column: payload}))
    rotation.validate_stored_encryption_keys()
    with database.begin() as conn:
        assert rotation.rotate_encryption_key(conn, old, new) == 0
        assert conn.scalar(select(table.c[column])) == payload


@pytest.mark.parametrize(
    "table_name,column,field",
    [("user_channels", "config", field) for field in CHANNEL_SECRET_FIELDS]
    + [("mcp_servers", "auth", field) for field in SENSITIVE_AUTH_FIELDS]
    + [("trigger_runs", "payload_snapshot", "encrypted_payload")],
)
@pytest.mark.parametrize("readable_source", [True, False])
def test_exact_json_secret_paths_still_fail_closed_and_rotate(
    database, caplog, table_name, column, field, readable_source
):
    old = Fernet(Fernet.generate_key())
    new = Fernet(os.environ["ENCRYPTION_KEY"].encode())
    source = old if readable_source else Fernet(Fernet.generate_key())
    token = source.encrypt(b"secret-value").decode()
    payload = {field: token, "nested": {field: token}, "items": [token]}
    table = Table(
        table_name,
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column(column, JSON),
    )
    table.create(database)
    with database.begin() as conn:
        conn.execute(table.insert().values(id=1, **{column: payload}))
    if table_name == "trigger_runs":
        rotation.validate_stored_encryption_keys()
        assert f"{table_name} row 1 field {column}" in caplog.text
    else:
        with pytest.raises(RuntimeError, match=f"{table_name} row 1 field {column}"):
            rotation.validate_stored_encryption_keys()
    if not readable_source:
        with pytest.raises(RuntimeError, match="migration rolled back"):
            with database.begin() as conn:
                rotation.rotate_encryption_key(conn, old, new)
        with database.connect() as conn:
            assert conn.scalar(select(table.c[column])) == payload
        return
    with database.begin() as conn:
        assert rotation.rotate_encryption_key(conn, old, new) == 1
        migrated = conn.scalar(select(table.c[column]))
        assert new.decrypt(migrated[field].encode()) == b"secret-value"
        assert migrated["nested"] == payload["nested"]
        assert migrated["items"] == payload["items"]
        assert rotation.rotate_encryption_key(conn, old, new) == 0
    rotation.validate_stored_encryption_keys()


@pytest.mark.parametrize("store_full_payload", [False, True])
def test_signed_webhook_metadata_cannot_poison_startup_or_migration_cli(
    database, monkeypatch, capsys, store_full_payload
):
    key = Fernet(os.environ["ENCRYPTION_KEY"].encode())
    secret = "diagnostic-webhook-secret"
    trigger = AgentTrigger(
        config={"store_full_payload": store_full_payload},
        secret_encrypted=key.encrypt(secret.encode()).decode(),
    )
    token = base64.urlsafe_b64encode(b"\x80" + b"\0" * 72).decode()
    timestamp = str(int(time.time()))
    body = b'{"sample":"ordinary payload"}'
    context = CallbackRequestContext(
        provider="webhook",
        callback_id="diagnostic",
        headers={
            "x-xagent-timestamp": timestamp,
            "x-xagent-signature": sign_webhook_payload(secret, timestamp, body),
            "x-xagent-event-id": token,
            "x-xagent-event-type": token,
        },
    )
    provider = WebhookProvider()
    get_cipher.cache_clear()
    try:
        verification = asyncio.run(
            provider.verify(context, db=None, trigger=trigger, raw_body=body)
        )
        assert verification.verified
        event = asyncio.run(
            provider.parse_events(context, db=None, trigger=trigger, raw_body=body)
        )[0]
        snapshot = _payload_snapshot(
            trigger,
            event.payload,
            source_event_id=event.source_event_id,
            event_type=event.event_type,
            resource_id=event.resource_id,
        )
    finally:
        get_cipher.cache_clear()
    assert snapshot["metadata"]["source_event_id"] == token
    assert snapshot["metadata"]["event_type"] == token
    assert ("encrypted_payload" in snapshot) is store_full_payload
    if store_full_payload:
        assert (
            json.loads(key.decrypt(snapshot["encrypted_payload"].encode()))
            == event.payload
        )
    table = Table(
        "trigger_runs",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("payload_snapshot", JSON),
    )
    table.create(database)
    with database.begin() as conn:
        conn.execute(table.insert().values(id=1, payload_snapshot=snapshot))
    rotation.validate_stored_encryption_keys()
    monkeypatch.setenv(
        "XAGENT_ENCRYPTION_MIGRATION_KEY", Fernet.generate_key().decode()
    )
    monkeypatch.setattr(encryption_key_migration, "configure_db", Mock())
    monkeypatch.setattr(encryption_key_migration, "get_engine", lambda: database)
    encryption_key_migration.main()
    assert "Re-encrypted 0 persisted credential fields." in capsys.readouterr().out
    with database.connect() as conn:
        assert conn.scalar(select(table.c.payload_snapshot)) == snapshot


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


@pytest.mark.parametrize("matching_destination", [True, False])
def test_global_authority_verifier_rotates_with_master(database, matching_destination):
    from xagent.web.services.global_memory_embedding_authority import (
        _credential_verifier,
    )

    old = Fernet(DEV_FALLBACK_ENCRYPTION_KEY.encode())
    new = Fernet(
        os.environ["ENCRYPTION_KEY"].encode()
        if matching_destination
        else Fernet.generate_key()
    )
    table = Table(
        "global_memory_embedding_authority",
        MetaData(),
        Column("authority_key", String, primary_key=True),
        Column("api_key_encrypted", Text),
        Column("credential_verifier", String),
    )
    table.create(database)
    ciphertext = old.encrypt(b"embedding-secret").decode()
    model = Table(
        "models",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("_api_key_encrypted", Text),
    )
    model.create(database)
    with database.begin() as conn:
        conn.execute(model.insert().values(id=1, _api_key_encrypted=ciphertext))
        conn.execute(
            table.insert().values(
                authority_key="global",
                api_key_encrypted=ciphertext,
                credential_verifier="old-verifier",
            )
        )
    if not matching_destination:
        with pytest.raises(RuntimeError, match="must match configured ENCRYPTION_KEY"):
            with database.begin() as conn:
                rotation.rotate_encryption_key(conn, old, new)
        with database.connect() as conn:
            assert conn.scalar(select(table.c.api_key_encrypted)) == ciphertext
            assert conn.scalar(select(table.c.credential_verifier)) == "old-verifier"
            assert conn.scalar(select(model.c._api_key_encrypted)) == ciphertext
        return
    with database.begin() as conn:
        assert rotation.rotate_encryption_key(conn, old, new) == 2
        assert conn.scalar(select(table.c.credential_verifier)) == _credential_verifier(
            "embedding-secret"
        )


def test_celery_disposes_pre_fork_database_pool_on_storage_failure(monkeypatch):
    from xagent.web.jobs import celery_app
    from xagent.web.models import database as db_module

    engine = Mock()
    monkeypatch.setattr(celery_app, "validate_encryption_config", Mock())
    monkeypatch.setattr(db_module, "create_database_engine", lambda: engine)
    monkeypatch.setattr(
        rotation,
        "validate_stored_encryption_keys",
        Mock(side_effect=RuntimeError("unreadable stored key")),
    )
    with pytest.raises(SystemExit, match="unreadable stored key"):
        celery_app.validate_worker_encryption()
    engine.dispose.assert_called_once()


@pytest.mark.parametrize("failed_stage", [None, "credential", "schema"])
def test_celery_admission_preserves_lazy_child_database_initialization(
    tmp_path, monkeypatch, failed_stage
):
    from xagent.web.jobs import celery_app, tasks
    from xagent.web.models import database as db_module

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'celery-child.db'}")
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("XAGENT_SECRET_ENCRYPTION_KEY", "s" * 40)
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    celery_app.validate_worker_encryption()
    assert db_module._engine is None
    assert db_module._SessionLocal is None
    try:
        if failed_stage:
            module, name = (
                (rotation, "validate_stored_encryption_keys")
                if failed_stage == "credential"
                else (db_module, "_initialize_database_schema")
            )
            with monkeypatch.context() as failure:
                failure.setattr(
                    module, name, Mock(side_effect=RuntimeError("rejected"))
                )
                for _ in range(2):
                    with pytest.raises(RuntimeError, match="rejected"):
                        tasks._open_worker_session()
                    assert db_module._engine is None
                    assert db_module._SessionLocal is None
        with tasks._open_worker_session() as session:
            assert "background_jobs" in inspect(session.get_bind()).get_table_names()
    finally:
        if db_module._engine is not None:
            db_module._engine.dispose()


@pytest.mark.parametrize(
    "table_name,column",
    [
        ("user_oauth", "access_token"),
        ("user_oauth", "refresh_token"),
        ("mcp_oauth_flow_states", "code_verifier"),
        ("task_runtime_secrets", "ciphertext"),
        ("trigger_runs", "payload_snapshot"),
    ],
)
def test_reconnectable_or_transient_data_warns_without_changing_ciphertext(
    database, caplog, table_name, column
):
    table = Table(
        table_name,
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column(column, JSON if column == "payload_snapshot" else Text),
    )
    table.create(database)
    old = Fernet(Fernet.generate_key())
    token = old.encrypt(b"private-stale-token").decode()
    value = {"encrypted_payload": token} if column == "payload_snapshot" else token
    with database.begin() as conn:
        conn.execute(table.insert().values(id=1, **{column: value}))
    rotation.validate_stored_encryption_keys()
    assert f"{table_name} row 1 field {column}" in caplog.text
    assert "ciphertext preserved" in caplog.text
    assert token not in caplog.text and "private-stale-token" not in caplog.text
    with database.connect() as conn:
        assert conn.scalar(select(table.c[column])) == value
    # Offline migration remains strict, even though service recovery is allowed.
    with pytest.raises(RuntimeError, match="migration rolled back"):
        with database.begin() as conn:
            rotation.rotate_encryption_key(
                conn,
                Fernet(Fernet.generate_key()),
                Fernet(os.environ["ENCRYPTION_KEY"].encode()),
            )


def test_admin_reset_still_works_with_an_unreadable_oauth_token(
    tmp_path, monkeypatch, caplog
):
    from types import SimpleNamespace

    from sqlalchemy import text

    from xagent.web import reset_admin_password
    from xagent.web.api.auth import verify_password
    from xagent.web.models import database as db_module
    from xagent.web.models.user import User
    from xagent.web.models.user_oauth import UserOAuth, is_undecryptable_oauth_token

    url = f"sqlite:///{tmp_path / 'admin-recovery.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("XAGENT_SECRET_ENCRYPTION_KEY", "s" * 40)
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    monkeypatch.setattr(reset_admin_password, "load_dotenv", Mock())
    monkeypatch.setattr(
        reset_admin_password,
        "parse_args",
        lambda: SimpleNamespace(
            username="recovery-admin",
            password="replacement-admin-password",
            yes=True,
        ),
    )
    token = Fernet(Fernet.generate_key()).encrypt(b"old-oauth-secret").decode()
    get_cipher.cache_clear()
    try:
        db_module.init_db()
        with db_module.get_session_local()() as db:
            db.add(
                User(username="recovery-admin", password_hash="old-hash", is_admin=True)
            )
            db.commit()
            user_id = db.query(User.id).filter_by(username="recovery-admin").scalar()
        with db_module.get_engine().begin() as conn:
            # Raw SQL must bypass the ORM encryption decorator for an old key.
            conn.execute(
                text(
                    "INSERT INTO user_oauth (user_id, provider, access_token) "
                    "VALUES (:user_id, 'google-drive', :token)"
                ),
                {"user_id": user_id, "token": token},
            )
        db_module.get_engine().dispose()
        reset_admin_password.main()
        with db_module.get_session_local()() as db:
            user = db.query(User).filter_by(username="recovery-admin").one()
            assert verify_password("replacement-admin-password", user.password_hash)
            assert is_undecryptable_oauth_token(db.query(UserOAuth).one().access_token)
            assert (
                db.execute(text("SELECT access_token FROM user_oauth")).scalar_one()
                == token
            )
        assert "Unreadable recoverable credential: user_oauth" in caplog.text
        assert token not in caplog.text and "old-oauth-secret" not in caplog.text
    finally:
        get_cipher.cache_clear()
        if db_module._engine is not None:
            db_module._engine.dispose()


def test_recoverable_warnings_are_bounded_for_large_stale_history(database, caplog):
    table = Table(
        "user_oauth",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("access_token", Text),
    )
    table.create(database)
    old = Fernet(Fernet.generate_key())
    with database.begin() as conn:
        conn.execute(
            table.insert(),
            [
                {"id": row_id, "access_token": old.encrypt(b"stale-secret").decode()}
                for row_id in range(1, 13)
            ],
        )
    rotation.validate_stored_encryption_keys()
    assert caplog.text.count("Unreadable recoverable credential:") == 5
    assert "7 additional unreadable recoverable credential fields" in caplog.text
    assert "stale-secret" not in caplog.text


def test_rotation_pages_are_fully_read_before_updates(database, monkeypatch):
    from sqlalchemy import event

    monkeypatch.setattr(rotation, "_ROW_BATCH_SIZE", 2)
    table = Table(
        "models",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("_api_key_encrypted", Text),
    )
    table.create(database)
    old, new = (
        Fernet(Fernet.generate_key()),
        Fernet(os.environ["ENCRYPTION_KEY"].encode()),
    )
    with database.begin() as conn:
        conn.execute(
            table.insert(),
            [
                {
                    "id": row_id,
                    "_api_key_encrypted": old.encrypt(str(row_id).encode()).decode(),
                }
                for row_id in range(1, 8)
            ]
            + [{"id": 8, "_api_key_encrypted": None}],
        )
    statements = []

    def record(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.startswith("SELECT models."):
            statements.append(statement)

    event.listen(database, "before_cursor_execute", record)
    try:
        with database.begin() as conn:
            assert rotation.rotate_encryption_key(conn, old, new) == 7
        assert len(statements) == 5
        assert all(
            "LIMIT" in statement and "IS NOT NULL" in statement
            for statement in statements
        )
        with database.connect() as conn:
            for row_id, token in conn.execute(select(table)).all():
                if row_id == 8:
                    assert token is None
                else:
                    assert new.decrypt(token.encode()).decode() == str(row_id)
        rotation.validate_stored_encryption_keys()
    finally:
        event.remove(database, "before_cursor_execute", record)


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
            assert db_module._engine is None
            assert db_module._SessionLocal is None
        else:
            schema.assert_called_once_with(db_module.get_engine())
        with seed_engine.connect() as conn:
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


def test_catalog_matches_live_columns_and_encryption_naming_conventions():
    from xagent.web.models.database import Base
    from xagent.web.models.user_oauth import EncryptedOAuthToken

    catalog = {
        (table_name, column)
        for table_name, columns in rotation.ENCRYPTED_COLUMNS.items()
        for column in columns
    }
    for table_name, column in catalog:
        assert column in Base.metadata.tables[table_name].c
    # A new conventionally named encrypted column or OAuth decorator must be
    # explicitly admitted to the rotation catalog, not silently omitted.
    candidates = {
        (table.name, column.name)
        for table in Base.metadata.tables.values()
        for column in table.c
        if "encrypted" in column.name
        or "ciphertext" in column.name
        or isinstance(column.type, EncryptedOAuthToken)
        or column.name
        in {"access_token", "refresh_token", "client_secret", "code_verifier", "env"}
    }
    # This is a signed application JWT refresh token, not Fernet-encrypted data.
    candidates.discard(("users", "refresh_token"))
    assert candidates <= catalog
    json_columns = {
        (table_name, column)
        for table_name, column in catalog
        if isinstance(Base.metadata.tables[table_name].c[column].type, JSON)
    }
    assert json_columns == rotation._ENV_COLUMNS | rotation._JSON_SECRET_FIELDS.keys()


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
