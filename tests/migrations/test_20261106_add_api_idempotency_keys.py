"""The api_idempotency_keys revision matches the model and is idempotent."""

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models.api_idempotency_key import ApiIdempotencyKey
from xagent.web.models.database import Base

engine = engine_fixture

MIGRATION = "xagent.migrations.versions.20261106_add_api_idempotency_keys"


def test_upgrade_matches_the_model_and_is_idempotent(engine):
    migration = importlib.import_module(MIGRATION)
    Base.metadata.create_all(engine)
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        migration.downgrade()
        assert not sa.inspect(connection).has_table(migration.TABLE)

        migration.upgrade()
        migration.upgrade()

        inspector = sa.inspect(connection)
        assert {c["name"] for c in inspector.get_columns(migration.TABLE)} == set(
            ApiIdempotencyKey.__table__.columns.keys()
        )
        uniques = {
            tuple(u["column_names"])
            for u in inspector.get_unique_constraints(migration.TABLE)
        }
        assert ("scope_kind", "scope_id", "operation", "idempotency_key") in uniques
        foreign_keys = {
            fk["constrained_columns"][0]: (
                fk["referred_table"],
                (fk.get("options") or {}).get("ondelete", "").upper(),
            )
            for fk in inspector.get_foreign_keys(migration.TABLE)
        }
        assert foreign_keys == {"task_id": ("tasks", "SET NULL")}
        assert migration.EXPIRES_INDEX in {
            i["name"] for i in inspector.get_indexes(migration.TABLE)
        }

        migration.downgrade()
        migration.downgrade()
        assert not sa.inspect(connection).has_table(migration.TABLE)
