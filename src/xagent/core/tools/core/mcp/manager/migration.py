import logging
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import inspect
from sqlalchemy.orm import Session

from .db import DatabaseMCPServerManager, MCPServer
from .file import YamlMCPServerManager

logger = logging.getLogger(__name__)


class MCPServerMigrator:
    """Migrates MCP server configurations from YAML to database format."""

    def __init__(self, storage_root: Path, db_session: Session):
        """
        Initialize migrator.

        Args:
            storage_root: Root directory where YAML files are stored
            db_session: SQLAlchemy database session
        """
        self.storage_root = storage_root
        self.yaml_config_file = self.storage_root / "mcp_servers.yaml"
        self.db_session = db_session

    def should_migrate(self) -> bool:
        """
        Check if migration should be performed.

        Returns:
            True if YAML file exists and database table is empty
        """
        # Check if YAML file exists
        if not self.yaml_config_file.exists():
            logger.debug("No YAML config file found, migration not needed")
            return False

        # Check if database table exists and has data
        inspector = inspect(self.db_session.bind)
        if inspector is None:
            logger.info("Unable to inspect, skipping migration")
            return False
        if "mcp_servers" not in inspector.get_table_names():
            logger.info("Database table doesn't exist, migration needed")
            return True

        # Check if table is empty
        count = self.db_session.query(MCPServer).count()
        if count == 0:
            logger.info("Database table is empty, migration needed")
            return True

        logger.debug(f"Database table has {count} records, migration not needed")
        return False

    def migrate(self) -> Dict[str, Any]:
        """
        Perform migration from YAML to database.

        Returns:
            Migration result with statistics
        """
        if not self.should_migrate():
            return {
                "migrated": False,
                "reason": "Migration not needed",
                "servers_migrated": 0,
                "errors": [],
            }

        logger.info("Starting MCP server migration from YAML to database")

        yaml_manager = YamlMCPServerManager(str(self.storage_root))
        yaml_servers = yaml_manager.list_servers()
        db_manager = DatabaseMCPServerManager(self.db_session)

        migrated_count = 0
        errors = []

        for server_data in yaml_servers:
            try:
                # Check if server already exists in DB
                existing = (
                    self.db_session.query(MCPServer)
                    .filter(MCPServer.name == server_data.config.name)
                    .first()
                )

                if existing:
                    logger.warning(
                        f"Server '{server_data.config.name}' already exists in database, skipping"
                    )
                    continue

                # Add to database
                db_manager.add_server(server_data.config)
                migrated_count += 1
                logger.info(f"Migrated server '{server_data.config.name}'")

            except Exception as e:
                error_msg = f"Failed to migrate server '{server_data.config.name}': {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Backup YAML file
        if migrated_count > 0:
            backup_file = self.yaml_config_file.with_suffix(".yaml.backup")
            try:
                self.yaml_config_file.rename(backup_file)
                logger.info(f"Backed up YAML config to {backup_file}")
            except Exception as e:
                logger.warning(f"Failed to backup YAML file: {e}")

        result = {
            "migrated": True,
            "servers_migrated": migrated_count,
            "total_servers": len(yaml_servers),
            "errors": errors,
        }

        logger.info(
            f"Migration completed: {migrated_count}/{len(yaml_servers)} servers migrated"
        )
        return result


def try_migrate(storage_root: Path, db_session: Session) -> Optional[Dict[str, Any]]:
    """
    Automatically perform migration if needed.

    Args:
        storage_root: Root directory where YAML files are stored
        db_session: SQLAlchemy database session

    Returns:
        Migration result if migration was performed, None otherwise
    """
    migrator = MCPServerMigrator(storage_root, db_session)

    if migrator.should_migrate():
        return migrator.migrate()

    return None
