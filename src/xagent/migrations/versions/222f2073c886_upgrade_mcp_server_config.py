"""upgrade mcp server config

Revision ID: 222f2073c886
Revises: 441d4f5d399c
Create Date: 2025-11-17 15:49:46.141714

"""

import json
import logging
from typing import Any, Callable, Dict, Optional, Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

# revision identifiers, used by Alembic.
revision: str = "222f2073c886"
down_revision: Union[str, None] = "a7186c6c5d89"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger(__name__)


def get_json_type(bind) -> Any:
    """
    Get the appropriate JSON type based on the database dialect.
    Returns postgresql.JSON for PostgreSQL, sa.JSON for other databases (SQLite).
    """
    dialect_name = bind.dialect.name

    if dialect_name == "postgresql":
        from sqlalchemy.dialects import postgresql

        return postgresql.JSON(astext_type=sa.Text())
    else:
        # For SQLite and other databases
        return sa.JSON()


# Create a base for our temporary models
Base = declarative_base()


class LegacyMCPServer(Base):  # type: ignore[valid-type, misc]
    """Temporary model for legacy mcp_servers table"""

    __tablename__ = "mcp_servers_legacy"

    id = sa.Column(sa.Integer, primary_key=True)
    user_id = sa.Column(sa.Integer, nullable=False)
    name = sa.Column(sa.String(100), nullable=False)
    description = sa.Column(sa.Text, nullable=True)
    transport = sa.Column(sa.String(50), nullable=False)
    config = sa.Column(sa.JSON, nullable=False)
    is_active = sa.Column(sa.Boolean, default=True)
    is_default = sa.Column(sa.Boolean, default=False)
    created_at = sa.Column(sa.DateTime(timezone=True))
    updated_at = sa.Column(sa.DateTime(timezone=True))


class NewMCPServer(Base):  # type: ignore[valid-type, misc]
    """Temporary model for new mcp_servers table"""

    __tablename__ = "mcp_servers"

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String(100), nullable=False, unique=True)
    description = sa.Column(sa.Text, nullable=True)
    managed = sa.Column(sa.String(20), nullable=False)
    transport = sa.Column(sa.String(50), nullable=False)
    command = sa.Column(sa.String(500), nullable=True)
    args = sa.Column(sa.JSON(), nullable=True)
    url = sa.Column(sa.String(500), nullable=True)
    env = sa.Column(sa.JSON(), nullable=True)
    cwd = sa.Column(sa.String(500), nullable=True)
    headers = sa.Column(sa.JSON(), nullable=True)
    docker_url = sa.Column(sa.String(500), nullable=True)
    docker_image = sa.Column(sa.String(200), nullable=True)
    docker_environment = sa.Column(sa.JSON(), nullable=True)
    docker_working_dir = sa.Column(sa.String(500), nullable=True)
    volumes = sa.Column(sa.JSON(), nullable=True)
    bind_ports = sa.Column(sa.JSON(), nullable=True)
    restart_policy = sa.Column(sa.String(50), nullable=False, server_default="no")
    auto_start = sa.Column(sa.Boolean, nullable=True)
    container_id = sa.Column(sa.String(100), nullable=True)
    container_name = sa.Column(sa.String(200), nullable=True)
    container_logs = sa.Column(sa.JSON(), nullable=True)
    created_at = sa.Column(sa.DateTime(timezone=True), server_default=sa.text("now()"))
    updated_at = sa.Column(sa.DateTime(timezone=True), server_default=sa.text("now()"))


class UserMCPServer(Base):  # type: ignore[valid-type, misc]
    """Temporary model for user_mcpservers table"""

    __tablename__ = "user_mcpservers"

    id = sa.Column(sa.Integer, primary_key=True)
    user_id = sa.Column(sa.Integer, nullable=False)
    mcpserver_id = sa.Column(sa.Integer, nullable=False)
    is_owner = sa.Column(sa.Boolean, nullable=False, server_default="false")
    can_edit = sa.Column(sa.Boolean, nullable=False, server_default="false")
    can_delete = sa.Column(sa.Boolean, nullable=False, server_default="false")
    is_shared = sa.Column(sa.Boolean, nullable=False, server_default="false")
    is_active = sa.Column(sa.Boolean, nullable=False, server_default="true")
    is_default = sa.Column(sa.Boolean, nullable=False, server_default="false")
    created_at = sa.Column(sa.DateTime(timezone=True), server_default=sa.text("now()"))
    updated_at = sa.Column(sa.DateTime(timezone=True))


# Keep all the parsing logic the same
class ConfigFieldParser:
    """Modular parser for configuration fields with type-specific parsing strategies."""

    @staticmethod
    def parse_string_list(value: str) -> list:
        """Parse a string into a list of strings."""
        try:
            import shlex

            return shlex.split(value)
        except ValueError:
            return [
                arg.strip() for arg in value.replace("\n", " ").split() if arg.strip()
            ]

    @staticmethod
    def parse_key_value_dict(value: str) -> Dict[str, str]:
        """Parse a string into a dictionary of key-value pairs."""
        try:
            result = json.loads(value)
            if isinstance(result, dict):
                return result
            raise ValueError("Not a dictionary")
        except (json.JSONDecodeError, ValueError):
            result = {}
            lines = value.replace("\n", " ").split()
            for line in lines:
                if "=" in line:
                    key, val = line.split("=", 1)
                    result[key.strip()] = val.strip()
            return result

    @staticmethod
    def parse_port_mappings(value: str) -> Dict[str, Union[int, str]]:
        """Parse port mappings as container_port:host_port."""
        try:
            result = json.loads(value)
            if isinstance(result, dict):
                return result
            raise ValueError("Not a dictionary")
        except (json.JSONDecodeError, ValueError):
            result = {}
            lines = value.replace("\n", " ").split()
            for line in lines:
                if ":" in line:
                    container_port, host_port = line.split(":", 1)
                    result[container_port.strip()] = host_port.strip()
            return result

    @staticmethod
    def parse_boolean(value: str) -> bool:
        """Parse a string into a boolean."""
        return value.lower() in ("true", "1", "yes", "on")


class MCPConfigFieldRegistry:
    """Registry of field parsers for different configuration fields."""

    STRING_LIST_FIELDS = {"args", "volumes"}
    KEY_VALUE_DICT_FIELDS = {"env", "headers", "docker_environment"}
    PORT_MAPPING_FIELDS = {"bind_ports"}
    BOOLEAN_FIELDS = {"auto_start"}

    @classmethod
    def get_parser_for_field(cls, field_name: str) -> Optional[Callable]:
        """Get the appropriate parser function for a field."""
        if field_name in cls.STRING_LIST_FIELDS:
            return ConfigFieldParser.parse_string_list
        elif field_name in cls.KEY_VALUE_DICT_FIELDS:
            return ConfigFieldParser.parse_key_value_dict
        elif field_name in cls.PORT_MAPPING_FIELDS:
            return ConfigFieldParser.parse_port_mappings
        elif field_name in cls.BOOLEAN_FIELDS:
            return ConfigFieldParser.parse_boolean
        return None


def parse_config_field(
    field_name: str, value: Any, transport: str | None = None
) -> Any:
    """Parse configuration field based on its expected type."""
    if value is None or value == "":
        return None

    if not isinstance(value, str):
        return value

    value = value.strip()
    if not value:
        return None

    parser = MCPConfigFieldRegistry.get_parser_for_field(field_name)

    if parser:
        try:
            result = parser(value)
            if isinstance(result, (dict, list)) and not result:
                return None
            return result
        except Exception as e:
            logger.warning(
                f"Failed to parse field '{field_name}' with value '{value}': {str(e)}"
            )
            return value

    return value


def normalize_config_fields(config: Dict[str, Any], transport: str) -> Dict[str, Any]:
    """Extract and normalize config fields to new table structure."""
    normalized = {
        "managed": "external",
        "command": None,
        "args": None,
        "url": None,
        "env": None,
        "cwd": None,
        "headers": None,
        "docker_url": None,
        "docker_image": None,
        "docker_environment": None,
        "docker_working_dir": None,
        "volumes": None,
        "bind_ports": None,
        "restart_policy": "no",
        "auto_start": None,
        "container_id": None,
        "container_name": None,
        "container_logs": None,
    }

    # Process each config field
    for field_name, value in config.items():
        if field_name in normalized:
            try:
                parsed_value = parse_config_field(field_name, value, transport)
                if parsed_value is not None:
                    normalized[field_name] = parsed_value
            except Exception as e:
                logger.warning(
                    f"Failed to parse field '{field_name}' with value '{value}': {e}"
                )
                normalized[field_name] = value

    # Detect internal management
    if any(
        key in config for key in ["docker_image", "docker_url", "volumes", "bind_ports"]
    ):
        normalized["managed"] = "internal"

    # Handle cwd
    if normalized["cwd"]:
        normalized["cwd"] = str(normalized["cwd"])

    # Ensure restart_policy has a default
    if not normalized["restart_policy"]:
        normalized["restart_policy"] = "no"

    return normalized


def migrate_single_server_orm(session: Session, legacy_server: LegacyMCPServer) -> None:
    """Migrate a single server using ORM - automatically handles database differences."""
    # Parse the legacy config
    config = legacy_server.config
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config for server {legacy_server.name}: {e}")
            config = {}
    elif config is None:
        config = {}

    # Normalize the config
    normalized_fields = normalize_config_fields(config, str(legacy_server.transport))  # type: ignore[arg-type]

    # Create new server record
    new_server = NewMCPServer(
        name=legacy_server.name,
        description=legacy_server.description,
        transport=legacy_server.transport,
        created_at=legacy_server.created_at,
        updated_at=legacy_server.updated_at,
        **normalized_fields,  # Unpack all normalized fields
    )

    try:
        # Add and flush to get the ID (works for both PostgreSQL and SQLite)
        session.add(new_server)
        session.flush()  # This assigns the ID without committing

        # Create user relationship
        user_relationship = UserMCPServer(
            user_id=legacy_server.user_id,
            mcpserver_id=new_server.id,  # ORM automatically provides the ID
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_shared=False,
            is_active=legacy_server.is_active,
            is_default=legacy_server.is_default,
            created_at=legacy_server.created_at,
            updated_at=legacy_server.updated_at,
        )

        session.add(user_relationship)

        logger.info(f"Successfully migrated server '{legacy_server.name}'")

    except Exception as e:
        logger.error(f"Failed to migrate server '{legacy_server.name}': {e}")
        session.rollback()  # Rollback this specific transaction
        raise


def migrate_data_orm() -> None:
    """Migrate data using SQLAlchemy ORM."""
    # Get database connection and create session
    connection = op.get_bind()
    session = Session(bind=connection)

    try:
        # Get all legacy servers
        legacy_servers = (
            session.query(LegacyMCPServer).order_by(LegacyMCPServer.id).all()
        )

        logger.info(
            f"Migrating {len(legacy_servers)} MCP servers from legacy format..."
        )

        migrated_count = 0
        failed_count = 0

        # Migrate each server individually with separate transactions
        for legacy_server in legacy_servers:
            # Create a new session for each server to isolate transactions
            server_session = Session(bind=connection)
            try:
                migrate_single_server_orm(server_session, legacy_server)
                server_session.commit()
                migrated_count += 1
                logger.info(f"Successfully migrated server '{legacy_server.name}'")
            except Exception as e:
                server_session.rollback()
                failed_count += 1
                logger.error(f"Failed to migrate server '{legacy_server.name}': {e}")
                # Continue with next server
                continue
            finally:
                server_session.close()

        logger.info(
            f"Migration completed: {migrated_count} successful, {failed_count} failed"
        )

        if failed_count > 0:
            logger.warning(
                f"{failed_count} servers failed to migrate. Check logs and legacy table for details."
            )

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        session.close()


def upgrade() -> None:
    """Upgrade to new MCP server structure using ORM."""
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    logger.info("Starting upgrade to new MCP server structure")

    try:
        # Check if old table exists and rename it to preserve it
        existing_tables = inspector.get_table_names()
        if (
            "mcp_servers" in existing_tables
            and "mcp_servers_legacy" not in existing_tables
        ):
            logger.info("Renaming old table mcp_servers to mcp_servers_legacy")
            op.rename_table("mcp_servers", "mcp_servers_legacy")

        # Create new tables if they don't exist
        if "mcp_servers" not in existing_tables:
            logger.info("Creating new tables")
            create_new_tables()

        # Migrate data using ORM if legacy table exists
        if "mcp_servers_legacy" in existing_tables:
            logger.info("Migrating data using ORM")
            migrate_data_orm()

        logger.info("Successfully completed upgrade to new MCP server structure")
    except Exception as e:
        logger.error(f"Failed to upgrade to new MCP server structure: {e}")
        raise


def create_new_tables() -> None:
    """Create the new normalized table structure."""
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    dialect_name = bind.dialect.name

    # Get the appropriate JSON type for the current database
    json_type = get_json_type(bind)

    # Get the appropriate timestamp default for the current database
    # PostgreSQL uses now(), SQLite uses CURRENT_TIMESTAMP
    timestamp_default = (
        sa.text("now()")
        if dialect_name == "postgresql"
        else sa.text("CURRENT_TIMESTAMP")
    )

    # Check and create new mcp_servers table
    existing_tables = inspector.get_table_names()
    if "mcp_servers" not in existing_tables:
        op.create_table(
            "mcp_servers",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(length=100), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("managed", sa.String(length=20), nullable=False),
            sa.Column("transport", sa.String(length=50), nullable=False),
            sa.Column("command", sa.String(length=500), nullable=True),
            sa.Column("args", json_type, nullable=True),
            sa.Column("url", sa.String(length=500), nullable=True),
            sa.Column("env", json_type, nullable=True),
            sa.Column("cwd", sa.String(length=500), nullable=True),
            sa.Column("headers", json_type, nullable=True),
            sa.Column("docker_url", sa.String(length=500), nullable=True),
            sa.Column("docker_image", sa.String(length=200), nullable=True),
            sa.Column("docker_environment", json_type, nullable=True),
            sa.Column("docker_working_dir", sa.String(length=500), nullable=True),
            sa.Column("volumes", json_type, nullable=True),
            sa.Column("bind_ports", json_type, nullable=True),
            sa.Column(
                "restart_policy",
                sa.String(length=50),
                nullable=False,
                server_default="no",
            ),
            sa.Column("auto_start", sa.Boolean(), nullable=True),
            sa.Column("container_id", sa.String(length=100), nullable=True),
            sa.Column("container_name", sa.String(length=200), nullable=True),
            sa.Column("container_logs", json_type, nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=timestamp_default,
                nullable=True,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=timestamp_default,
                nullable=True,
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("name"),
        )

    # Create user_mcpservers relationship table
    if "user_mcpservers" not in existing_tables:
        # Build table constraints dynamically based on what tables exist
        foreign_keys = [
            sa.ForeignKeyConstraint(
                ["mcpserver_id"], ["mcp_servers.id"], ondelete="CASCADE"
            )
        ]

        # Only add users foreign key if users table exists
        # This handles the case where migrations are run from empty database
        # and users table hasn't been created yet (will be created by SQLAlchemy)
        if "users" in existing_tables:
            foreign_keys.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )

        op.create_table(
            "user_mcpservers",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("mcpserver_id", sa.Integer(), nullable=False),
            sa.Column("is_owner", sa.Boolean(), nullable=False, server_default="false"),
            sa.Column("can_edit", sa.Boolean(), nullable=False, server_default="false"),
            sa.Column(
                "can_delete", sa.Boolean(), nullable=False, server_default="false"
            ),
            sa.Column(
                "is_shared", sa.Boolean(), nullable=False, server_default="false"
            ),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column(
                "is_default", sa.Boolean(), nullable=False, server_default="false"
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=timestamp_default,
                nullable=True,
            ),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            *foreign_keys,
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("user_id", "mcpserver_id", name="uq_user_mcpservers"),
        )

    # Check and create index
    existing_indexes = (
        [idx["name"] for idx in inspector.get_indexes("user_mcpservers")]
        if "user_mcpservers" in existing_tables
        else []
    )
    if "ix_user_mcpservers_id" not in existing_indexes:
        op.create_index(
            op.f("ix_user_mcpservers_id"), "user_mcpservers", ["id"], unique=False
        )


def downgrade() -> None:
    """Downgrade to old MCP server structure."""
    # Check if legacy table exists before attempting downgrade
    inspector = sa.inspect(op.get_bind())
    existing_tables = inspector.get_table_names()

    if "mcp_servers_legacy" not in existing_tables:
        logger.warning(
            "Legacy table 'mcp_servers_legacy' not found. "
            "Cannot perform downgrade - legacy data may have been removed."
        )
        return

    op.drop_table("user_mcpservers")
    op.drop_table("mcp_servers")
    op.rename_table("mcp_servers_legacy", "mcp_servers")
