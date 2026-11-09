"""``mcp_oauth_flow_states.access_path``: the CHECK and the migration."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from tests.alembic.lock_retry_support import counting
from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.db.config import create_alembic_config
from xagent.web.models import MCPOAuthClient, MCPOAuthFlowState
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User

REVISION = "20261105_mcp_oauth_flow_access_path"
DOWN_REVISION = "20261105_task_structured_output_record"
ROOT = Path(__file__).parents[2]
MIGRATIONS_DIR = ROOT / "src" / "xagent" / "migrations"
MIGRATION_PATH = (
    MIGRATIONS_DIR / "versions" / "20261105_add_mcp_oauth_flow_access_path.py"
)
TABLE = "mcp_oauth_flow_states"
CONSTRAINT = "ck_mcp_oauth_flow_states_access_path"

# (access_path, generation present, owner key is the user's default, accepted)
COMBINATIONS = [
    ("association", True, True, True),
    ("association", False, True, True),
    ("association", True, False, True),
    ("team_member", False, True, True),
    ("team_member", True, True, False),
    ("team_member", False, False, False),
]


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgresql)]
)
def engine(request):
    if request.param == "postgresql":
        with disposable_database_factory("xagent_flow_access_path") as make:
            yield make("flow")
    else:
        yield sa.create_engine("sqlite://", poolclass=sa.pool.StaticPool)


def _create_tables(engine: sa.Engine) -> None:
    Base.metadata.create_all(
        engine,
        tables=[
            User.__table__,
            MCPServer.__table__,
            UserMCPServer.__table__,
            PublicMCPApp.__table__,
            MCPOAuthClient.__table__,
            MCPOAuthFlowState.__table__,
        ],
    )


def test_access_path_constraint(engine) -> None:
    _create_tables(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        user = User(username="flow-user", password_hash="x")
        server = MCPServer.from_config(
            {
                "name": "records",
                "managed": "external",
                "transport": "streamable_http",
                "url": "https://mcp.example.com/mcp",
                "auth": {"type": "mcp_oauth"},
            }
        )
        db.add_all([user, server])
        db.flush()
        client = MCPOAuthClient(
            mcp_server_id=server.id,
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            client_id="client",
            token_endpoint_auth_method="none",
            redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
        )
        db.add(client)
        db.commit()
        user_id, server_id, client_id = int(user.id), int(server.id), int(client.id)

    def flow(state: str, **values) -> MCPOAuthFlowState:
        return MCPOAuthFlowState(
            state=state,
            mcp_server_id=server_id,
            user_id=user_id,
            mcp_oauth_client_id=client_id,
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            code_verifier="verifier",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
            **values,
        )

    for index, (access_path, generation, default_key, accepted) in enumerate(
        COMBINATIONS
    ):
        with factory() as db:
            db.add(
                flow(
                    f"state-{index}",
                    access_path=access_path,
                    association_lifecycle_generation=(
                        uuid.uuid4() if generation else None
                    ),
                    resource_owner_key=(
                        f"xagent:user:{user_id}" if default_key else "actor:other"
                    ),
                )
            )
            if accepted:
                db.commit()
            else:
                with pytest.raises(IntegrityError, match=CONSTRAINT):
                    db.commit()

    with factory() as db:
        default_row = flow("state-default", resource_owner_key="k")
        db.add(default_row)
        db.commit()
        assert default_row.access_path == "association"


def test_model_carries_the_migration_check_on_the_column() -> None:
    migration = load_migration_module(MIGRATION_PATH, "access_path_model")
    column = MCPOAuthFlowState.__table__.c["access_path"]
    checks = {item.name: str(item.sqltext) for item in column.constraints}
    assert checks == {CONSTRAINT: migration.CHECK_CONDITION}


# --- migration ----------------------------------------------------------------


def _stamp_version(connection: sa.Connection) -> None:
    connection.execute(
        sa.text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
    )
    connection.execute(
        sa.text("INSERT INTO alembic_version VALUES (:revision)"),
        {"revision": DOWN_REVISION},
    )


def _legacy_flow_table(connection: sa.Connection, *, with_column: bool) -> None:
    generation_type = "UUID" if connection.dialect.name == "postgresql" else "CHAR(32)"
    extra = (
        ", access_path VARCHAR(16) NOT NULL DEFAULT 'association'"
        if with_column
        else ""
    )
    connection.execute(
        sa.text(
            f"CREATE TABLE {TABLE} ("
            "id INTEGER PRIMARY KEY, "
            "state VARCHAR(255) NOT NULL UNIQUE, "
            "user_id INTEGER NOT NULL, "
            "resource_owner_key VARCHAR(512) NOT NULL, "
            f"association_lifecycle_generation {generation_type}"
            f"{extra})"
        )
    )
    connection.execute(
        sa.text(
            f"INSERT INTO {TABLE} (id, state, user_id, resource_owner_key) "
            "VALUES (1, 'legacy-with-null', 7, 'xagent:user:7'), "
            "(2, 'legacy-other', 7, 'actor:other')"
        )
    )
    _stamp_version(connection)


def _config(connection: sa.Connection):
    config = create_alembic_config(connection.engine)
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = connection
    return config


def _check_names(connection: sa.Connection) -> set[str]:
    return {
        item["name"] for item in sa.inspect(connection).get_check_constraints(TABLE)
    }


def _assert_constraint_enforced(connection: sa.Connection) -> None:
    generation = (
        str(uuid.uuid4())
        if connection.dialect.name == "postgresql"
        else uuid.uuid4().hex
    )
    insert = sa.text(
        f"INSERT INTO {TABLE} (id, state, user_id, resource_owner_key, "
        "access_path, association_lifecycle_generation) "
        "VALUES (:id, :state, 7, :key, 'team_member', :generation)"
    )
    for row in (
        {"id": 10, "state": "s10", "key": "xagent:user:7", "generation": generation},
        {"id": 11, "state": "s11", "key": "actor:other", "generation": None},
    ):
        with pytest.raises(IntegrityError, match=CONSTRAINT):
            with connection.begin_nested():
                connection.execute(insert, row)
    with connection.begin_nested() as accepted:
        connection.execute(
            insert,
            {"id": 12, "state": "s12", "key": "xagent:user:7", "generation": None},
        )
        accepted.rollback()


def test_upgrade_gives_old_rows_association_and_round_trips(engine) -> None:
    with engine.connect() as connection:
        _legacy_flow_table(connection, with_column=False)
        config = _config(connection)
        command.upgrade(config, REVISION)
        connection.commit()
        rows = connection.execute(
            sa.text(f"SELECT access_path FROM {TABLE} ORDER BY id")
        ).scalars()
        assert list(rows) == ["association", "association"]
        assert CONSTRAINT in _check_names(connection)
        if connection.dialect.name == "postgresql":
            # NOT VALID: adding it scanned nothing; new writes are still checked.
            validated = connection.execute(
                sa.text("SELECT convalidated FROM pg_constraint WHERE conname = :c"),
                {"c": CONSTRAINT},
            ).scalar_one()
            assert validated is False
        _assert_constraint_enforced(connection)

        command.downgrade(config, DOWN_REVISION)
        connection.commit()
        columns = {item["name"] for item in sa.inspect(connection).get_columns(TABLE)}
        assert "access_path" not in columns
        assert CONSTRAINT not in _check_names(connection)

        command.upgrade(config, REVISION)
        connection.commit()
        assert CONSTRAINT in _check_names(connection)


def test_upgrade_adds_only_the_check_and_a_second_run_changes_nothing(engine) -> None:
    with engine.connect() as connection:
        _legacy_flow_table(connection, with_column=True)
        assert CONSTRAINT not in _check_names(connection)
        config = _config(connection)
        command.upgrade(config, REVISION)
        connection.commit()
        assert CONSTRAINT in _check_names(connection)
        _assert_constraint_enforced(connection)

        # Back to the previous revision with the column and the CHECK in place,
        # as after create_all: the upgrade body runs again and alters nothing.
        connection.execute(
            sa.text("UPDATE alembic_version SET version_num = :revision"),
            {"revision": DOWN_REVISION},
        )
        connection.commit()
        with counting(engine, "ALTER TABLE") as alters:
            command.upgrade(config, REVISION)
            connection.commit()
        assert alters == []
        assert CONSTRAINT in _check_names(connection)


def test_upgrade_without_the_table_does_nothing(engine) -> None:
    with engine.connect() as connection:
        _stamp_version(connection)
        command.upgrade(_config(connection), REVISION)
        assert TABLE not in sa.inspect(connection).get_table_names()


# --- offline (--sql) ----------------------------------------------------------


def _offline_sql(dialect: str, operation: str) -> str:
    migration = load_migration_module(MIGRATION_PATH, f"access_path_{dialect}")
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        getattr(migration, operation)()
    return output.getvalue()


@pytest.mark.parametrize("operation", ["upgrade", "downgrade"])
def test_offline_postgresql_runs_each_statement_under_a_lock_timeout(
    operation,
) -> None:
    migration = load_migration_module(MIGRATION_PATH, "access_path_statements")
    statements = {
        "upgrade": [
            f"ALTER TABLE {TABLE} ADD COLUMN access_path "
            "VARCHAR(16) NOT NULL DEFAULT 'association'",
            f"ALTER TABLE {TABLE} ADD CONSTRAINT {CONSTRAINT} "
            f"CHECK ({migration.CHECK_CONDITION}) NOT VALID",
        ],
        "downgrade": [
            f"ALTER TABLE {TABLE} DROP CONSTRAINT IF EXISTS {CONSTRAINT}",
            f"ALTER TABLE {TABLE} DROP COLUMN access_path",
        ],
    }[operation]
    sql = _offline_sql("postgresql", operation)

    positions = []
    for statement in statements:
        assert sql.count(statement) == 1, statement
        positions.append(sql.index(statement))
        assert sql.rfind("SET lock_timeout = '5s'", 0, positions[-1]) != -1
        assert sql.index("RESET lock_timeout", positions[-1]) != -1
    assert positions == sorted(positions)
    assert sql.count("SET lock_timeout = '5s'") == len(statements)


def test_offline_sqlite_adds_the_check_with_the_column_and_can_drop_it() -> None:
    migration = load_migration_module(MIGRATION_PATH, "access_path_sqlite")
    upgrade = " ".join(_offline_sql("sqlite", "upgrade").split())
    assert (
        f"ALTER TABLE {TABLE} ADD COLUMN access_path VARCHAR(16) "
        f"DEFAULT 'association' NOT NULL CONSTRAINT {CONSTRAINT} "
        f"CHECK ({migration.CHECK_CONDITION})"
    ) in upgrade

    downgrade = [
        statement
        for statement in _offline_sql("sqlite", "downgrade").split(";")
        if statement.strip()
    ]
    migrated = sa.create_engine("sqlite://", poolclass=sa.pool.StaticPool)
    with migrated.connect() as connection:
        _legacy_flow_table(connection, with_column=False)
        command.upgrade(_config(connection), REVISION)
        connection.commit()
    built = sa.create_engine("sqlite://", poolclass=sa.pool.StaticPool)
    _create_tables(built)
    # The --sql downgrade runs on a table the online upgrade or create_all made.
    for target in (migrated, built):
        with target.begin() as connection:
            for statement in downgrade:
                connection.execute(sa.text(statement))
            columns = {c["name"] for c in sa.inspect(connection).get_columns(TABLE)}
            assert "access_path" not in columns
