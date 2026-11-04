"""Migration tests for the trigger provider foundation schema."""

from __future__ import annotations

import pytest
from alembic import command
from sqlalchemy import create_engine, inspect, text

from xagent.db.config import create_alembic_config

PREVIOUS_REVISION = "20260629_add_gmail_watch_states"
FOUNDATION_REVISION = "20260702_add_trigger_provider_foundation"

NEW_TRIGGER_COLUMNS = {
    "provider",
    "callback_id",
    "resource_id",
    "secret_encrypted",
    "provisioning_status",
    "provisioning_error",
}
NEW_GMAIL_WATCH_COLUMNS = {
    "callback_id",
    "push_audience",
    "subscription_name",
    "status",
}


@pytest.fixture()
def engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'foundation.db'}")


def _columns(engine, table: str) -> dict[str, dict]:
    return {col["name"]: col for col in inspect(engine).get_columns(table)}


def _table_names(engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def _upgrade(engine, revision: str) -> None:
    command.upgrade(create_alembic_config(engine), revision)


def _downgrade(engine, revision: str) -> None:
    command.downgrade(create_alembic_config(engine), revision)


class TestUpgrade:
    def test_upgrade_adds_nullable_identity_columns_and_audit_table(self, engine):
        _upgrade(engine, FOUNDATION_REVISION)

        trigger_columns = _columns(engine, "agent_triggers")
        assert NEW_TRIGGER_COLUMNS <= set(trigger_columns)
        for name in NEW_TRIGGER_COLUMNS:
            assert trigger_columns[name]["nullable"], name

        gmail_columns = _columns(engine, "gmail_watch_states")
        assert NEW_GMAIL_WATCH_COLUMNS <= set(gmail_columns)
        for name in NEW_GMAIL_WATCH_COLUMNS:
            assert gmail_columns[name]["nullable"], name

        audit_columns = _columns(engine, "trigger_audits")
        assert {
            "id",
            "trigger_id",
            "provider",
            "callback_id",
            "outcome",
            "detail",
            "remote_ip",
            "created_at",
        } <= set(audit_columns)
        assert audit_columns["trigger_id"]["nullable"]
        assert not audit_columns["outcome"]["nullable"]

    def test_audit_trigger_fk_preserves_rows_with_set_null(self, engine):
        _upgrade(engine, FOUNDATION_REVISION)
        foreign_keys = inspect(engine).get_foreign_keys("trigger_audits")
        trigger_fk = next(
            fk for fk in foreign_keys if fk["constrained_columns"] == ["trigger_id"]
        )
        assert trigger_fk["referred_table"] == "agent_triggers"
        assert trigger_fk["options"].get("ondelete") == "SET NULL"

    def test_upgrade_backfills_provider_from_type(self, engine):
        _upgrade(engine, PREVIOUS_REVISION)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO agent_triggers "
                    "(user_id, agent_id, type, name, enabled, config) "
                    "VALUES (1, 1, 'webhook', 'Legacy', 1, '{}')"
                )
            )

        _upgrade(engine, FOUNDATION_REVISION)

        with engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT provider, callback_id, resource_id, secret_encrypted, "
                    "provisioning_status, provisioning_error "
                    "FROM agent_triggers WHERE name = 'Legacy'"
                )
            ).one()
        assert tuple(row) == ("webhook", None, None, None, None, None)

    def test_upgrade_backfills_gmail_resource_id_from_oauth_account(self, engine):
        _upgrade(engine, PREVIOUS_REVISION)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO user_oauth "
                    "(id, user_id, provider, access_token, email) "
                    "VALUES (7, 1, 'gmail', 'token', 'Legacy.User@Gmail.com')"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO agent_triggers "
                    "(user_id, agent_id, type, name, enabled, config) "
                    "VALUES (1, 1, 'gmail', 'Legacy Gmail', 1, "
                    "'{\"oauth_account_id\": 7}')"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO agent_triggers "
                    "(user_id, agent_id, type, name, enabled, config) "
                    "VALUES (1, 1, 'gmail', 'Orphan Gmail', 1, "
                    "'{\"oauth_account_id\": 999}')"
                )
            )

        _upgrade(engine, FOUNDATION_REVISION)

        with engine.begin() as conn:
            backfilled = conn.execute(
                text(
                    "SELECT provider, resource_id FROM agent_triggers "
                    "WHERE name = 'Legacy Gmail'"
                )
            ).one()
            orphan = conn.execute(
                text(
                    "SELECT provider, resource_id FROM agent_triggers "
                    "WHERE name = 'Orphan Gmail'"
                )
            ).one()
        assert tuple(backfilled) == ("gmail", "legacy.user@gmail.com")
        assert tuple(orphan) == ("gmail", None)

    def test_upgrade_is_idempotent_when_rerun_against_same_schema(self, engine):
        _upgrade(engine, FOUNDATION_REVISION)
        _downgrade(engine, PREVIOUS_REVISION)
        _upgrade(engine, FOUNDATION_REVISION)
        assert "trigger_audits" in _table_names(engine)


class TestDowngrade:
    def test_downgrade_removes_columns_and_audit_table(self, engine):
        _upgrade(engine, FOUNDATION_REVISION)
        _downgrade(engine, PREVIOUS_REVISION)

        assert "trigger_audits" not in _table_names(engine)
        assert not (NEW_TRIGGER_COLUMNS & set(_columns(engine, "agent_triggers")))
        assert not (
            NEW_GMAIL_WATCH_COLUMNS & set(_columns(engine, "gmail_watch_states"))
        )
