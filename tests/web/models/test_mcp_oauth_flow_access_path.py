"""``mcp_oauth_flow_states.access_path``: the CHECK and the migration."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.db.config import create_alembic_config
from xagent.web.models import MCPOAuthClient, MCPOAuthFlowState
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User

REVISION = "20261105_mcp_oauth_flow_access_path"
DOWN_REVISION = "20261105_task_structured_output_record"
MIGRATIONS_DIR = Path(__file__).parents[3] / "src" / "xagent" / "migrations"
CONSTRAINT = "ck_mcp_oauth_flow_states_access_path"

# (access_path, generation present, accepted)
COMBINATIONS = [
    pytest.param("association", True, True, id="association_with_generation"),
    pytest.param("association", False, True, id="association_without_generation"),
    pytest.param("team_member", False, True, id="team_member_without_generation"),
    pytest.param("team_member", True, False, id="team_member_with_generation"),
]


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


def _assert_combinations(engine: sa.Engine) -> None:
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
        ids = (int(user.id), int(server.id), int(client.id))

    for index, param in enumerate(COMBINATIONS):
        access_path, with_generation, accepted = param.values
        with factory() as db:
            db.add(
                MCPOAuthFlowState(
                    state=f"state-{index}",
                    mcp_server_id=ids[1],
                    user_id=ids[0],
                    access_path=access_path,
                    association_lifecycle_generation=(
                        uuid.uuid4() if with_generation else None
                    ),
                    mcp_oauth_client_id=ids[2],
                    resource_owner_key=f"xagent:user:{ids[0]}",
                    issuer="https://auth.example.com",
                    resource="https://mcp.example.com/mcp",
                    scope="",
                    code_verifier="verifier",
                    expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
                )
            )
            if accepted:
                db.commit()
            else:
                with pytest.raises(IntegrityError):
                    db.commit()
                db.rollback()

    with factory() as db:
        default_row = MCPOAuthFlowState(
            state="state-default",
            mcp_server_id=ids[1],
            user_id=ids[0],
            mcp_oauth_client_id=ids[2],
            resource_owner_key="k",
            issuer="i",
            resource="r",
            code_verifier="v",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        )
        db.add(default_row)
        db.commit()
        assert default_row.access_path == "association"


def test_sqlite_access_path_constraint() -> None:
    engine = sa.create_engine("sqlite://", poolclass=sa.pool.StaticPool)
    _create_tables(engine)
    _assert_combinations(engine)


@pytest.fixture()
def postgresql_engine_factory():
    with disposable_database_factory("xagent_flow_access_path") as make:
        yield make


@pytest.mark.postgresql
def test_postgresql_access_path_constraint(postgresql_engine_factory) -> None:
    engine = postgresql_engine_factory("constraint")
    _create_tables(engine)
    _assert_combinations(engine)


# --- migration ----------------------------------------------------------------


def _legacy_flow_table(connection: sa.Connection, *, with_column: bool) -> None:
    generation_type = "UUID" if connection.dialect.name == "postgresql" else "CHAR(32)"
    extra = (
        ", access_path VARCHAR(16) NOT NULL DEFAULT 'association'"
        if with_column
        else ""
    )
    connection.execute(
        sa.text(
            "CREATE TABLE mcp_oauth_flow_states ("
            "id INTEGER PRIMARY KEY, "
            "state VARCHAR(255) NOT NULL UNIQUE, "
            f"association_lifecycle_generation {generation_type}"
            f"{extra})"
        )
    )
    connection.execute(
        sa.text(
            "INSERT INTO mcp_oauth_flow_states (id, state) "
            "VALUES (1, 'legacy-with-null'), (2, 'legacy-other')"
        )
    )
    connection.execute(
        sa.text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
    )
    connection.execute(
        sa.text("INSERT INTO alembic_version VALUES (:revision)"),
        {"revision": DOWN_REVISION},
    )


def _config(engine: sa.Engine):
    config = create_alembic_config(engine)
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    return config


def _check_names(connection: sa.Connection) -> set[str]:
    return {
        item["name"]
        for item in sa.inspect(connection).get_check_constraints(
            "mcp_oauth_flow_states"
        )
    }


def _assert_constraint_enforced(connection: sa.Connection) -> None:
    generation = (
        str(uuid.uuid4())
        if connection.dialect.name == "postgresql"
        else uuid.uuid4().hex
    )
    with pytest.raises(IntegrityError):
        connection.execute(
            sa.text(
                "INSERT INTO mcp_oauth_flow_states "
                "(id, state, access_path, association_lifecycle_generation) "
                "VALUES (10, 'bad', 'team_member', :generation)"
            ),
            {"generation": generation},
        )
    connection.rollback()
    connection.execute(
        sa.text(
            "INSERT INTO mcp_oauth_flow_states (id, state, access_path) "
            "VALUES (11, 'good', 'team_member')"
        )
    )
    connection.rollback()


def _migration_cycle(engine: sa.Engine) -> None:
    config = _config(engine)
    with engine.connect() as connection:
        _legacy_flow_table(connection, with_column=False)
        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)
        connection.commit()
        rows = connection.execute(
            sa.text("SELECT access_path FROM mcp_oauth_flow_states ORDER BY id")
        ).scalars()
        assert list(rows) == ["association", "association"]
        assert CONSTRAINT in _check_names(connection)
        _assert_constraint_enforced(connection)

        command.downgrade(config, DOWN_REVISION)
        connection.commit()
        columns = {
            item["name"]
            for item in sa.inspect(connection).get_columns("mcp_oauth_flow_states")
        }
        assert "access_path" not in columns
        assert CONSTRAINT not in _check_names(connection)

        command.upgrade(config, REVISION)
        connection.commit()
        assert CONSTRAINT in _check_names(connection)


def _existing_column_cycle(engine: sa.Engine) -> None:
    config = _config(engine)
    with engine.connect() as connection:
        _legacy_flow_table(connection, with_column=True)
        assert CONSTRAINT not in _check_names(connection)
        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)
        connection.commit()
        assert CONSTRAINT in _check_names(connection)
        _assert_constraint_enforced(connection)
        # Running it again changes nothing.
        command.downgrade(config, DOWN_REVISION)
        connection.commit()
        command.upgrade(config, REVISION)
        connection.commit()
        assert CONSTRAINT in _check_names(connection)


def _missing_table(engine: sa.Engine) -> None:
    config = _config(engine)
    with engine.connect() as connection:
        connection.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        connection.execute(
            sa.text("INSERT INTO alembic_version VALUES (:revision)"),
            {"revision": DOWN_REVISION},
        )
        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)
        assert "mcp_oauth_flow_states" not in sa.inspect(connection).get_table_names()


def test_sqlite_upgrade_gives_old_rows_association_and_round_trips() -> None:
    _migration_cycle(sa.create_engine("sqlite:///:memory:"))


def test_sqlite_upgrade_when_the_column_already_exists() -> None:
    _existing_column_cycle(sa.create_engine("sqlite:///:memory:"))


def test_sqlite_upgrade_without_the_table_does_nothing() -> None:
    _missing_table(sa.create_engine("sqlite:///:memory:"))


@pytest.mark.postgresql
def test_postgresql_upgrade_gives_old_rows_association_and_round_trips(
    postgresql_engine_factory,
) -> None:
    _migration_cycle(postgresql_engine_factory("cycle"))


@pytest.mark.postgresql
def test_postgresql_upgrade_when_the_column_already_exists(
    postgresql_engine_factory,
) -> None:
    _existing_column_cycle(postgresql_engine_factory("existing"))


@pytest.mark.postgresql
def test_postgresql_upgrade_without_the_table_does_nothing(
    postgresql_engine_factory,
) -> None:
    _missing_table(postgresql_engine_factory("missing"))
