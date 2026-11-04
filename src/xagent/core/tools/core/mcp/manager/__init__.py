import logging
from typing import Optional

from .....storage.manager import get_db_session, get_storage_root
from .base import MCPServerManager
from .db import DatabaseMCPServerManager
from .migration import try_migrate

_mcp_server_manager: Optional[MCPServerManager] = None


logger = logging.getLogger(__name__)


def initialize_mcp_manager() -> MCPServerManager:
    """
    Initialize the global MCP server manager singleton.

    Args:
        storage_root: Root directory where MCP configuration files are stored.

    Returns:
        MCPServerManager: The initialized global instance.

    Raises:
        RuntimeError: If the manager is already initialized.
    """
    global _mcp_server_manager

    if _mcp_server_manager is not None:
        logger.warning(
            "MCP Server Manager already initialized, returning existing instance"
        )
        return _mcp_server_manager

    logger.info("Initializing MCP Server Manager")

    db = next(get_db_session())
    storage_root = get_storage_root()
    migration_result = try_migrate(storage_root, db)
    if migration_result:
        logger.info(f"Successfully migrated from yaml to database: {migration_result}")

    _mcp_server_manager = DatabaseMCPServerManager(db)

    # Auto-start all internal managed servers with auto_start=True
    try:
        servers = _mcp_server_manager.list_servers()
        started_count = 0

        for server_data in servers:
            if (
                server_data.config.managed == "internal"
                and server_data.config.auto_start
            ):
                try:
                    _mcp_server_manager.start_server(server_data.config.name)
                    logger.info(f"Auto-started MCP server: {server_data.config.name}")
                    started_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to auto-start MCP server '{server_data.config.name}': {e}"
                    )

        if started_count > 0:
            logger.info(f"Auto-started {started_count} MCP servers")
        else:
            logger.debug("No MCP servers configured for auto-start")

    except Exception as e:
        logger.error(f"Error during MCP server auto-start: {e}")

    return _mcp_server_manager


def get_mcp_manager() -> MCPServerManager:
    """
    Get the global MCP server manager instance.

    Returns:
        MCPServerManager: The global instance.

    Raises:
        RuntimeError: If the manager is not initialized.
    """
    if _mcp_server_manager is None:
        raise RuntimeError(
            "MCP Server Manager not initialized. Call initialize_mcp_manager() first."
        )
    return _mcp_server_manager


def is_initialized() -> bool:
    """
    Check if the MCP server manager is initialized.

    Returns:
        bool: True if initialized, False otherwise.
    """
    return _mcp_server_manager is not None


def shutdown_mcp_manager() -> None:
    """
    Shutdown the global MCP server manager and cleanup resources.

    This function stops all running internal servers and cleans up the singleton.
    """
    global _mcp_server_manager

    if _mcp_server_manager is None:
        logger.debug("MCP Server Manager not initialized, nothing to shutdown")
        return

    logger.info("Shutting down MCP Server Manager...")

    try:
        # Stop all internal servers
        servers = _mcp_server_manager.list_servers()
        stopped_count = 0

        for server_data in servers:
            if server_data.config.managed == "internal":
                try:
                    _mcp_server_manager.stop_server(server_data.config.name)
                    logger.info(f"Stopped MCP server: {server_data.config.name}")
                    stopped_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to stop MCP server '{server_data.config.name}': {e}"
                    )

        if stopped_count > 0:
            logger.info(f"Stopped {stopped_count} MCP servers")

    except Exception as e:
        logger.error(f"Error during MCP server shutdown: {e}")
    finally:
        # Cleanup singleton
        _mcp_server_manager = None
        logger.info("MCP Server Manager shutdown complete")
