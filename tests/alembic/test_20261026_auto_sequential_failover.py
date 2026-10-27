"""Migration coverage for ordered Auto sequential failover."""

from importlib import import_module

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION = import_module(
    "xagent.migrations.versions.20261026_auto_sequential_failover"
)


def _old_schema(metadata: sa.MetaData) -> tuple[sa.Table, sa.Table]:
    configs = sa.Table(
        "auto_model_configs",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
    )
    candidates = sa.Table(
        "auto_model_candidates",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("config_id", sa.Integer, nullable=False),
        sa.Column("routing_model_id", sa.String(200), nullable=False),
        sa.Column("target_model_id", sa.Integer, nullable=False),
        sa.UniqueConstraint(
            "config_id",
            "routing_model_id",
            name="uq_auto_candidate_routing_model",
        ),
        sa.UniqueConstraint(
            "config_id",
            "target_model_id",
            name="uq_auto_candidate_target_model",
        ),
    )
    return configs, candidates


def test_upgrade_allows_duplicate_profiles_and_refuses_lossy_downgrade():
    engine = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    configs, candidates = _old_schema(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(configs.insert().values(id=1))
        connection.execute(
            candidates.insert(),
            [
                {
                    "id": 8,
                    "config_id": 1,
                    "routing_model_id": "deepseek/deepseek-chat",
                    "target_model_id": 80,
                },
                {
                    "id": 9,
                    "config_id": 1,
                    "routing_model_id": "z-ai/glm-5.3",
                    "target_model_id": 90,
                },
            ],
        )
        with Operations.context(MigrationContext.configure(connection)):
            MIGRATION.upgrade()

        upgraded = sa.Table(
            "auto_model_candidates", sa.MetaData(), autoload_with=connection
        )
        connection.execute(
            upgraded.insert().values(
                id=10,
                config_id=1,
                routing_model_id="deepseek/deepseek-chat",
                target_model_id=100,
                position=2,
            )
        )
        rows = connection.execute(
            sa.select(
                upgraded.c.routing_model_id,
                upgraded.c.target_model_id,
                upgraded.c.position,
            ).order_by(upgraded.c.position)
        ).all()

        assert rows == [
            ("deepseek/deepseek-chat", 80, 0),
            ("z-ai/glm-5.3", 90, 1),
            ("deepseek/deepseek-chat", 100, 2),
        ]
        assert "mode" in {
            column["name"]
            for column in sa.inspect(connection).get_columns("auto_model_configs")
        }

        with pytest.raises(RuntimeError, match="without losing bindings"):
            with Operations.context(MigrationContext.configure(connection)):
                MIGRATION.downgrade()

        protected = sa.Table(
            "auto_model_candidates", sa.MetaData(), autoload_with=connection
        )
        assert "position" in protected.c
        assert (
            connection.execute(
                sa.select(sa.func.count()).select_from(protected)
            ).scalar_one()
            == 3
        )
        assert "uq_auto_candidate_routing_model" not in {
            constraint["name"]
            for constraint in sa.inspect(connection).get_unique_constraints(
                "auto_model_candidates"
            )
        }


def test_downgrade_restores_old_schema_when_profiles_are_unique():
    engine = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    configs, candidates = _old_schema(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(configs.insert().values(id=1))
        connection.execute(
            candidates.insert().values(
                id=8,
                config_id=1,
                routing_model_id="deepseek/deepseek-chat",
                target_model_id=80,
            )
        )
        with Operations.context(MigrationContext.configure(connection)):
            MIGRATION.upgrade()
            MIGRATION.downgrade()

        restored = sa.Table(
            "auto_model_candidates", sa.MetaData(), autoload_with=connection
        )
        assert "position" not in restored.c
        assert "mode" not in {
            column["name"]
            for column in sa.inspect(connection).get_columns("auto_model_configs")
        }
        assert "uq_auto_candidate_routing_model" in {
            constraint["name"]
            for constraint in sa.inspect(connection).get_unique_constraints(
                "auto_model_candidates"
            )
        }


def test_upgrade_repairs_a_partially_applied_position_constraint():
    engine = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    sa.Table(
        "auto_model_configs",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("mode", sa.String(32), nullable=False, server_default="routing"),
    )
    candidates = sa.Table(
        "auto_model_candidates",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("config_id", sa.Integer, nullable=False),
        sa.Column("routing_model_id", sa.String(200), nullable=False),
        sa.Column("target_model_id", sa.Integer, nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.UniqueConstraint(
            "config_id",
            "routing_model_id",
            name="uq_auto_candidate_routing_model",
        ),
        sa.UniqueConstraint(
            "config_id",
            "target_model_id",
            name="uq_auto_candidate_target_model",
        ),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            candidates.insert().values(
                id=1,
                config_id=1,
                routing_model_id="deepseek/deepseek-chat",
                target_model_id=80,
                position=0,
            )
        )
        with Operations.context(MigrationContext.configure(connection)):
            MIGRATION.upgrade()

        constraints = {
            constraint["name"]
            for constraint in sa.inspect(connection).get_unique_constraints(
                "auto_model_candidates"
            )
        }
        assert "uq_auto_candidate_position" in constraints
        assert "uq_auto_candidate_routing_model" not in constraints
