"""Tests for migration 20261106_hash_user_refresh_tokens."""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

PARENT_REVISION = "20261105_update_atlassian_name_and_description"
TARGET_REVISION = "20261106_hash_user_refresh_tokens"

JWT_A = "eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxfQ.c2lnbmF0dXJlLWE"
JWT_B = "eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoyfQ.c2lnbmF0dXJlLWI"


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@pytest.fixture
def config(tmp_path: Path) -> Config:
    url = f"sqlite:///{tmp_path / 'test.db'}"
    cfg = Config()
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.set_main_option("script_location", "src/xagent/migrations")
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        conn.execute(
            sa.text("INSERT INTO alembic_version VALUES (:rev)"),
            {"rev": PARENT_REVISION},
        )
        conn.execute(
            sa.text(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, "
                "username VARCHAR(50) NOT NULL, refresh_token VARCHAR(255), "
                "refresh_token_expires_at DATETIME)"
            )
        )
    engine.dispose()
    return cfg


def _insert(config: Config, rows: list[tuple[int, str | None]]) -> None:
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    with engine.begin() as conn:
        for user_id, token in rows:
            conn.execute(
                sa.text(
                    "INSERT INTO users (id, username, refresh_token, "
                    "refresh_token_expires_at) VALUES (:id, :name, :token, "
                    "'2026-12-01 00:00:00')"
                ),
                {"id": user_id, "name": f"u{user_id}", "token": token},
            )
    engine.dispose()


def _tokens(config: Config) -> dict[int, str | None]:
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    try:
        with engine.connect() as conn:
            return dict(
                conn.execute(sa.text("SELECT id, refresh_token FROM users")).all()
            )
    finally:
        engine.dispose()


def test_upgrade_replaces_plaintext_tokens_with_their_digest(config: Config) -> None:
    _insert(config, [(1, JWT_A), (2, JWT_B), (3, None)])

    command.upgrade(config, TARGET_REVISION)

    assert _tokens(config) == {1: _digest(JWT_A), 2: _digest(JWT_B), 3: None}


def test_upgrade_does_not_hash_an_already_converted_row(config: Config) -> None:
    _insert(config, [(1, _digest(JWT_A)), (2, JWT_B)])

    command.upgrade(config, TARGET_REVISION)

    assert _tokens(config) == {1: _digest(JWT_A), 2: _digest(JWT_B)}


def _run_upgrade_directly(
    config: Config, *, batch_size: int | None = None, before=None
) -> None:
    """Run upgrade() itself, so a patched module constant reaches the code that runs.

    Alembic loads each revision file under a module id of its own, so patching
    the imported module has no effect on ``command.upgrade``.
    """
    module = import_module(
        "xagent.migrations.versions.20261106_hash_user_refresh_tokens"
    )
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    if before is not None:
        before(engine)
    original = module.BATCH_SIZE
    try:
        if batch_size is not None:
            module.BATCH_SIZE = batch_size
        with (
            engine.begin() as connection,
            Operations.context(MigrationContext.configure(connection)),
        ):
            module.upgrade()
    finally:
        module.BATCH_SIZE = original
        engine.dispose()


def test_upgrade_converts_more_rows_than_one_batch(config: Config) -> None:
    tokens = {i: f"h.p.sig{i}" for i in range(1, 8)}
    _insert(config, list(tokens.items()))

    _run_upgrade_directly(config, batch_size=2)

    assert _tokens(config) == {i: _digest(t) for i, t in tokens.items()}


def test_upgrade_does_not_overwrite_a_token_rotated_after_it_was_read(
    config: Config,
) -> None:
    _insert(config, [(1, JWT_A), (2, JWT_B)])
    rotated = _digest("rotated.after.read")
    fired = []

    def rotate_after_the_select(engine: sa.Engine) -> None:
        @sa.event.listens_for(engine, "after_cursor_execute")
        def _(conn, cursor, statement, parameters, context, executemany):
            if "ORDER BY" in statement and "refresh_token" in statement and not fired:
                fired.append(True)
                conn.exec_driver_sql(
                    "UPDATE users SET refresh_token = ? WHERE id = 1", (rotated,)
                )

    _run_upgrade_directly(config, before=rotate_after_the_select)

    assert fired
    # The concurrent writer's digest survives; the untouched row is converted.
    assert _tokens(config) == {1: rotated, 2: _digest(JWT_B)}


def test_upgrade_without_the_column_is_a_noop(config: Config) -> None:
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    with engine.begin() as conn:
        conn.execute(sa.text("DROP TABLE users"))
        conn.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
    engine.dispose()

    command.upgrade(config, TARGET_REVISION)


def _expiries(config: Config) -> dict[int, object]:
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    try:
        with engine.connect() as conn:
            return dict(
                conn.execute(
                    sa.text("SELECT id, refresh_token_expires_at FROM users")
                ).all()
            )
    finally:
        engine.dispose()


def test_downgrade_revokes_converted_sessions_and_keeps_plaintext_ones(
    config: Config,
) -> None:
    _insert(config, [(1, JWT_A)])
    command.upgrade(config, TARGET_REVISION)
    # A row an older release wrote after the upgrade is still a usable token.
    _insert(config, [(2, JWT_B)])

    command.downgrade(config, PARENT_REVISION)

    assert _tokens(config) == {1: None, 2: JWT_B}
    expiries = _expiries(config)
    assert expiries[1] is None
    assert expiries[2] is not None
