"""
MCP Server Management API Endpoints

Provides REST API endpoints for managing MCP server configurations
in the web application.
"""

import json
import logging
import shlex
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...core.tools.core.mcp.data_config import MCPServerConfig
from ...core.tools.core.mcp.manager.db import DatabaseMCPServerManager
from ..auth_dependencies import get_current_user
from ..mcp_apps import MCP_APPS_LIBRARY, get_app_by_name
from ..models.database import get_db
from ..models.mcp import MCPServer, UserMCPServer
from ..models.user import User

logger = logging.getLogger(__name__)


# Pydantic models for API
class MCPServerCreate(BaseModel):
    """Request model for creating MCP server."""

    name: str = Field(..., min_length=1, max_length=100, description="Server name")
    transport: str = Field(
        ..., description="Transport type (stdio, sse, websocket, streamable_http)"
    )
    description: Optional[str] = Field(None, description="Server description")
    config: dict = Field(..., description="Transport-specific configuration")
    is_active: bool = Field(True, description="Whether the server is active")


class MCPServerUpdate(BaseModel):
    """Request model for updating MCP server."""

    name: Optional[str] = Field(
        None, min_length=1, max_length=100, description="Server name"
    )
    transport: Optional[str] = Field(None, description="Transport type")
    description: Optional[str] = Field(None, description="Server description")
    config: Optional[dict] = Field(None, description="Transport-specific configuration")
    is_active: Optional[bool] = Field(None, description="Whether the server is active")


class MCPServerResponse(BaseModel):
    """Response model for MCP server."""

    id: int
    user_id: int
    name: str
    transport: str
    description: Optional[str]
    config: dict
    is_active: bool
    is_default: bool
    transport_display: str
    created_at: str
    updated_at: str
    connected_account: Optional[str] = None
    app_id: Optional[str] = None
    provider: Optional[str] = None

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class MCPConnectionTest(BaseModel):
    """Request model for testing MCP connection."""

    name: str = Field(..., description="Connection name")
    transport: str = Field(..., description="Transport type")
    config: dict[str, Any] = Field(..., description="Connection configuration")


class MCPConnectionTestResponse(BaseModel):
    """Response model for MCP connection test."""

    success: bool
    message: str
    details: Optional[dict] = None


# Create router
mcp_router = APIRouter(prefix="/api/mcp", tags=["MCP Management"])


class ConfigFieldParser:
    """Modular parser for configuration fields with type-specific parsing strategies."""

    @staticmethod
    def parse_string_list(value: str) -> List[str]:
        """Parse a string into a list of strings."""
        try:
            # Try JSON first
            result = json.loads(value)
            if isinstance(result, list):
                return result
            raise ValueError("Not a list")
        except (json.JSONDecodeError, ValueError):
            try:
                # Try to parse as shell command line
                return shlex.split(value)
            except ValueError:
                # Fall back to splitting by whitespace and newlines
                return [
                    arg.strip()
                    for arg in value.replace("\n", " ").split()
                    if arg.strip()
                ]

    @staticmethod
    def parse_key_value_dict(value: str) -> Dict[str, str]:
        """Parse a string into a dictionary of key-value pairs."""
        try:
            # Try JSON first
            result = json.loads(value)
            if isinstance(result, dict):
                return result
            raise ValueError("Not a dictionary")
        except (json.JSONDecodeError, ValueError):
            # Parse as key=value pairs (one per line or space-separated)
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
            # Try JSON first
            result = json.loads(value)
            if isinstance(result, dict):
                return result
            raise ValueError("Not a dictionary")
        except (json.JSONDecodeError, ValueError):
            # Parse as port:port pairs
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

    @staticmethod
    def parse_json_or_fallback(
        value: str, fallback_parser: Callable[[Any], Any] | None = None
    ) -> Any:
        """Try to parse as JSON, fall back to another parser if provided."""
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            if fallback_parser:
                return fallback_parser(value)
            return value


class MCPConfigFieldRegistry:
    """Registry of field parsers for different configuration fields."""

    # Field type mappings
    STRING_LIST_FIELDS = {"args", "volumes"}
    KEY_VALUE_DICT_FIELDS = {"env", "headers", "docker_environment"}
    PORT_MAPPING_FIELDS = {"bind_ports"}
    BOOLEAN_FIELDS = {"auto_start"}
    JSON_FIELDS = {"headers"}  # Fields that should prefer JSON parsing

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


class TransportFieldValidator:
    """Validate fields based on transport type."""

    TRANSPORT_REQUIRED_FIELDS = {
        "stdio": {"command"},
        "sse": {"url"},
        "websocket": {"url"},
        "streamable_http": {"url"},
    }

    TRANSPORT_OPTIONAL_FIELDS = {
        "stdio": {"args", "env", "cwd"},
        "sse": {"headers"},
        "websocket": {"headers"},
        "streamable_http": {"headers"},
    }

    @classmethod
    def validate_transport_fields(
        cls, transport: str, config_dict: Dict[str, Any]
    ) -> None:
        """Validate that required fields are present for the transport type."""
        required_fields = cls.TRANSPORT_REQUIRED_FIELDS.get(transport, set())

        for field in required_fields:
            if field not in config_dict or config_dict[field] is None:
                raise ValueError(f"Transport '{transport}' requires field '{field}'")


def _build_server_config(
    server_data: MCPServerCreate, existing_server: Optional[MCPServer] = None
) -> MCPServerConfig:
    """Build MCPServerConfig from API request data using modular parsing."""
    # Start with base config
    config_dict = {
        "name": server_data.name,
        "transport": server_data.transport,
        "description": server_data.description,
        "managed": "external",  # Default for user-created servers
    }

    # Parse and add config fields
    if server_data.config:
        for field_name, value in server_data.config.items():
            if field_name not in [
                "name",
                "transport",
                "description",
            ]:  # Skip already handled fields
                try:
                    parsed_value = _parse_config_field(
                        field_name, value, server_data.transport
                    )
                    if parsed_value is not None:
                        config_dict[field_name] = parsed_value
                except ValueError as e:
                    raise ValueError(
                        f"Configuration error in field '{field_name}': {str(e)}"
                    )

    # For updates, preserve existing values if not provided
    if existing_server:
        existing_config = existing_server.to_config_dict()
        for key, value in existing_config.items():
            if key not in config_dict and value is not None:
                config_dict[key] = value

    # Validate transport-specific requirements
    TransportFieldValidator.validate_transport_fields(
        server_data.transport, config_dict
    )

    return MCPServerConfig(**config_dict)


def _update_server_from_config(server: MCPServer, config: MCPServerConfig) -> None:
    """Update database server object from MCPServerConfig."""
    # Map config fields to database fields
    field_mapping = {
        "name": "name",
        "description": "description",
        "transport": "transport",
        "managed": "managed",
        "command": "command",
        "args": "args",
        "url": "url",
        "env": "env",
        "cwd": "cwd",
        "headers": "headers",
        "docker_url": "docker_url",
        "docker_image": "docker_image",
        "docker_environment": "docker_environment",
        "docker_working_dir": "docker_working_dir",
        "volumes": "volumes",
        "bind_ports": "bind_ports",
        "restart_policy": "restart_policy",
        "auto_start": "auto_start",
    }

    for config_field, db_field in field_mapping.items():
        if hasattr(config, config_field) and hasattr(server, db_field):
            value = getattr(config, config_field)
            setattr(server, db_field, value)


def _parse_config_field(
    field_name: str, value: Any, transport: str | None = None
) -> Any:
    """
    Parse configuration field based on its expected type.

    Args:
        field_name: Name of the configuration field
        value: Raw value to parse
        transport: Transport type (for transport-specific parsing if needed)

    Returns:
        Parsed value in the appropriate type
    """
    # Handle None or empty values
    if value is None or value == "":
        return None

    # If not a string, return as-is (already parsed)
    if not isinstance(value, str):
        return value

    # Clean up string value
    value = value.strip()
    if not value:
        return None

    # Get parser for this field
    parser = MCPConfigFieldRegistry.get_parser_for_field(field_name)

    if parser:
        try:
            result = parser(value)
            # Return None for empty results
            if isinstance(result, (dict, list)) and not result:
                return None
            return result
        except Exception as e:
            raise ValueError(f"Failed to parse field '{field_name}': {str(e)}")

    # Default: return string value as-is
    return value


def _db_server_to_response(
    server: MCPServer,
    user_mcp: UserMCPServer,
    manager: DatabaseMCPServerManager,
    connected_account: Optional[str] = None,
    app_id: Optional[str] = None,
    provider: Optional[str] = None,
) -> MCPServerResponse:
    """Convert database MCPServer to response model."""
    # Get status from manager if available
    config = server.to_config_dict()

    # Ensure JSON fields are properly serialized for frontend
    serialized_config: dict[str, Any] = {}
    for key, value in config.items():
        if value is None:
            serialized_config[key] = None
        elif isinstance(value, (dict, list)):
            # Convert to JSON string for frontend display
            serialized_config[key] = json.dumps(value, ensure_ascii=False, indent=2)
        else:
            serialized_config[key] = value

    return MCPServerResponse(
        id=server.id,
        user_id=user_mcp.user_id,
        name=server.name,
        transport=server.transport,
        description=server.description,
        config=serialized_config,
        is_active=user_mcp.is_active,
        is_default=user_mcp.is_default,
        transport_display=server.transport_display,
        created_at=str(server.created_at.isoformat()),
        updated_at=str(server.updated_at.isoformat()),
        connected_account=connected_account,
        app_id=app_id,
        provider=provider,
    )


def _enrich_oauth_server_info(
    server: MCPServer, oauth_emails: dict
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (app_id, provider, connected_account) for an OAuth-based MCPServer.
    This encapsulates the logic of looking up app information in O(1) time.
    """
    if server.transport != "oauth":
        return None, None, None

    app_info = get_app_by_name(str(server.name))
    if not app_info:
        return None, None, None

    provider = app_info.get("provider")
    app_id = app_info.get("id")
    connected_account = oauth_emails.get(app_id) or oauth_emails.get(provider)

    return app_id, provider, connected_account


@mcp_router.get("/apps", response_model=List[dict])
async def list_mcp_apps(
    search: Optional[str] = None,
    category: Optional[str] = "All",
    location: Optional[str] = "remote",
    status: Optional[str] = "all",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[dict]:
    """Get the list of available MCP applications in the library."""

    # Query connected servers for the current user
    user_mcps = (
        db.query(MCPServer, UserMCPServer)
        .join(UserMCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
        .filter(UserMCPServer.user_id == current_user.id)
        .all()
    )

    # Also fetch user oauth accounts to get the connected email
    from ..models.user_oauth import UserOAuth

    oauth_accounts = (
        db.query(UserOAuth).filter(UserOAuth.user_id == current_user.id).all()
    )
    oauth_emails = {
        str(oauth.provider): str(oauth.email) for oauth in oauth_accounts if oauth.email
    }

    # Try to map connected names to emails
    connected_apps = {}
    for server, _ in user_mcps:
        connected_apps[server.name.lower()] = server.id

    results = []

    if location in ["remote", "all"]:
        for app in MCP_APPS_LIBRARY:
            if search:
                search_lower = search.lower()
                if (
                    search_lower not in app["name"].lower()
                    and search_lower not in app.get("description", "").lower()
                ):
                    continue

            if category and category != "All":
                if app.get("category") != category:
                    continue

            app_copy = app.copy()
            is_connected = (
                app["name"].lower() in connected_apps
                or app["id"].lower() in connected_apps
            )
            app_copy["is_connected"] = is_connected

            if is_connected:
                # Find the server id
                server_id = connected_apps.get(
                    app["name"].lower()
                ) or connected_apps.get(app["id"].lower())
                app_copy["server_id"] = server_id

                # Find connected email
                provider = app.get("provider")
                app_id = app.get("id")
                # Check for email in both provider and app_id keys
                email = oauth_emails.get(str(app_id)) if app_id else None
                if not email and provider:
                    email = oauth_emails.get(str(provider))
                if email:
                    app_copy["connected_account"] = email

            if status == "verified" and not app_copy["is_connected"]:
                continue

            results.append(app_copy)

    if location in ["local", "all"]:
        library_names = {app["name"].lower() for app in MCP_APPS_LIBRARY}
        for server, user_mcp in user_mcps:
            if server.name.lower() in library_names:
                continue

            if search:
                search_lower = search.lower()
                if search_lower not in server.name.lower() and (
                    server.description
                    and search_lower not in server.description.lower()
                ):
                    continue

            if category and category != "All":
                continue

            results.append(
                {
                    "id": server.name,
                    "name": server.name,
                    "description": server.description or "Custom MCP Server",
                    "icon": "",
                    "users": "1",
                    "transport": server.transport,
                    "is_connected": True,
                    "provider": "custom",
                    "category": "Local",
                    "is_local": True,
                    "server_id": server.id,
                }
            )

    return results


@mcp_router.get("/servers", response_model=List[MCPServerResponse])
async def list_mcp_servers(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[MCPServerResponse]:
    """List MCP servers for the current user."""
    try:
        manager = DatabaseMCPServerManager(db)
        user_id = current_user.id

        # Get user's MCP servers
        user_mcps = (
            db.query(UserMCPServer, MCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(UserMCPServer.user_id == user_id)
            .order_by(MCPServer.created_at.desc())
            .all()
        )

        logger.info(f"user_mcps: {user_mcps}")

        # Fetch oauth emails
        from ..models.user_oauth import UserOAuth

        oauth_accounts = db.query(UserOAuth).filter(UserOAuth.user_id == user_id).all()
        oauth_emails = {
            str(oauth.provider): str(oauth.email)
            for oauth in oauth_accounts
            if oauth.email
        }

        responses = []
        for user_mcp, server in user_mcps:
            app_id, provider, connected_account = _enrich_oauth_server_info(
                server, oauth_emails
            )
            responses.append(
                _db_server_to_response(
                    server, user_mcp, manager, connected_account, app_id, provider
                )
            )

        return responses

    except Exception as e:
        logger.error(f"Failed to list MCP servers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list MCP servers",
        )


@mcp_router.get("/servers/{server_id}", response_model=MCPServerResponse)
async def get_mcp_server(
    server_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MCPServerResponse:
    """Get a specific MCP server."""
    try:
        manager = DatabaseMCPServerManager(db)
        user_id = current_user.id

        # Check user has access to this server
        result = (
            db.query(UserMCPServer, MCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
            .first()
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found"
            )

        user_mcp, server = result

        # Fetch oauth emails for this user to enrich the server info
        from ..models.user_oauth import UserOAuth

        oauth_accounts = db.query(UserOAuth).filter(UserOAuth.user_id == user_id).all()
        oauth_emails = {
            oauth.provider: oauth.email for oauth in oauth_accounts if oauth.email
        }

        app_id, provider, connected_account = _enrich_oauth_server_info(
            server, oauth_emails
        )

        return _db_server_to_response(
            server, user_mcp, manager, connected_account, app_id, provider
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MCP server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCP server",
        )


@mcp_router.post(
    "/servers", response_model=MCPServerResponse, status_code=status.HTTP_201_CREATED
)
async def create_mcp_server(
    server_data: MCPServerCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MCPServerResponse:
    """Create a new MCP server."""
    try:
        manager = DatabaseMCPServerManager(db)
        user_id = current_user.id

        # Check if server name already exists
        existing = (
            db.query(MCPServer).filter(MCPServer.name == server_data.name).first()
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"MCP server '{server_data.name}' already exists",
            )

        # Build and validate config
        try:
            config = _build_server_config(server_data)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid configuration: {str(e)}",
            )

        # Add server using manager
        manager.add_server(config)

        # Get the created server
        server = db.query(MCPServer).filter(MCPServer.name == server_data.name).first()
        if not server:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create server",
            )

        # Create user-server association
        user_mcp = UserMCPServer(
            user_id=user_id, mcpserver_id=server.id, is_active=server_data.is_active
        )
        db.add(user_mcp)
        db.commit()
        db.refresh(user_mcp)

        logger.info(f"Created MCP server '{server_data.name}' for user {user_id}")
        return _db_server_to_response(server, user_mcp, manager)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create MCP server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create MCP server: {str(e)}",
        )


@mcp_router.put("/servers/{server_id}", response_model=MCPServerResponse)
async def update_mcp_server(
    server_id: int,
    server_data: MCPServerUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MCPServerResponse:
    """Update an existing MCP server."""
    try:
        manager = DatabaseMCPServerManager(db)
        user_id = current_user.id

        # Check user has access to this server
        result = (
            db.query(UserMCPServer, MCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
            .first()
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found"
            )

        user_mcp, server = result

        # Check for name conflicts if updating name
        if server_data.name and server_data.name != server.name:
            existing = (
                db.query(MCPServer)
                .filter(MCPServer.name == server_data.name, MCPServer.id != server_id)
                .first()
            )
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"MCP server '{server_data.name}' already exists",
                )

        # Build update config - only include provided fields
        update_data = MCPServerCreate(
            name=server_data.name or server.name,
            transport=server_data.transport or server.transport,
            description=server_data.description
            if server_data.description is not None
            else server.description,
            config=server_data.config or {},
            is_active=server_data.is_active
            if server_data.is_active is not None
            else user_mcp.is_active,
        )

        # Build and validate config
        try:
            config = _build_server_config(update_data, server)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid configuration: {str(e)}",
            )

        # Update server fields
        _update_server_from_config(server, config)

        # Update user association if needed
        if server_data.is_active is not None:
            user_mcp.is_active = server_data.is_active

        db.commit()
        db.refresh(server)
        db.refresh(user_mcp)

        logger.info(f"Updated MCP server '{server.name}' for user {user_id}")
        return _db_server_to_response(server, user_mcp, manager)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update MCP server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update MCP server: {str(e)}",
        )


@mcp_router.delete("/servers/{server_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mcp_server(
    server_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Delete an MCP server."""
    try:
        manager = DatabaseMCPServerManager(db)
        user_id = current_user.id

        # Check user has access to this server
        result = (
            db.query(UserMCPServer, MCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
            .first()
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found"
            )

        user_mcp, server = result
        server_name = server.name

        # If it's an OAuth server, also delete the corresponding OAuth tokens
        if server.transport == "oauth":
            from ..models.user_oauth import UserOAuth

            # Find the corresponding app_id and provider
            app_info = next(
                (app for app in MCP_APPS_LIBRARY if app.get("name") == server.name),
                None,
            )
            if app_info:
                provider = app_info.get("provider")
                app_id = app_info.get("id")

                # Delete tokens for this specific app
                providers_to_delete = [p for p in [provider, app_id] if p is not None]
                if providers_to_delete:
                    db.query(UserOAuth).filter(
                        UserOAuth.user_id == user_id,
                        UserOAuth.provider.in_(providers_to_delete),
                    ).delete(synchronize_session=False)

        # Remove user-server association
        db.delete(user_mcp)
        db.commit()

        # Check if any other users are using this server
        other_users = (
            db.query(UserMCPServer)
            .filter(UserMCPServer.mcpserver_id == server_id)
            .first()
        )

        # Only remove from manager and delete if no other users
        if not other_users:
            manager.remove_server(server_name)
            logger.info(f"Deleted MCP server '{server_name}'")
        else:
            logger.info(f"Removed user {user_id} access to MCP server '{server_name}'")

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete MCP server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete MCP server",
        )


@mcp_router.post("/servers/{server_id}/toggle", response_model=MCPServerResponse)
async def toggle_mcp_server(
    server_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MCPServerResponse:
    """Toggle MCP server active status."""
    try:
        manager = DatabaseMCPServerManager(db)
        user_id = current_user.id

        # Check user has access to this server
        result = (
            db.query(UserMCPServer, MCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
            .first()
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found"
            )

        user_mcp, server = result

        # Toggle active status
        user_mcp.is_active = not user_mcp.is_active
        db.commit()
        db.refresh(user_mcp)

        status_text = "activated" if user_mcp.is_active else "deactivated"
        logger.info(
            f"{status_text.capitalize()} MCP server '{server.name}' for user {user_id}"
        )

        return _db_server_to_response(server, user_mcp, manager)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to toggle MCP server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to toggle MCP server",
        )


@mcp_router.get("/servers/{server_id}/logs")
async def get_mcp_server_logs(
    server_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    lines: int = 100,
) -> Dict[str, Any]:
    """Get logs for an internal MCP server."""
    try:
        manager = DatabaseMCPServerManager(db)
        user_id = current_user.id

        if not (1 <= lines <= 1000):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="lines must be between 1 and 1000",
            )

        # Check user has access to this server
        result = (
            db.query(UserMCPServer, MCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
            .first()
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found"
            )

        _, server = result

        if server.managed != "internal":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Logs only available for internal servers",
            )

        log_lines = manager.get_logs(server.name, lines)
        return {"server_name": server.name, "logs": log_lines or []}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MCP server logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCP server logs",
        )


@mcp_router.post("/test-connection", response_model=MCPConnectionTestResponse)
async def test_mcp_connection(
    test_data: MCPConnectionTest, db: Session = Depends(get_db)
) -> MCPConnectionTestResponse:
    """Test MCP server connection without saving."""
    try:
        from ...core.tools.adapters.vibe.mcp_adapter import (
            load_mcp_tools_as_agent_tools,
        )

        connection: dict[str, Any] = {
            "name": test_data.name,
            "transport": test_data.transport,
        }

        connection.update(**test_data.config)

        try:
            connections_dict: Dict[str, Any] = {"test": connection}
            tools = await load_mcp_tools_as_agent_tools(
                connections_dict, name_prefix="test_"
            )

            if tools:
                return MCPConnectionTestResponse(
                    success=True,
                    message=f"Successfully connected to {test_data.name}. Loaded {len(tools)} tools.",
                    details={"tool_count": len(tools)},
                )
            else:
                return MCPConnectionTestResponse(
                    success=True,
                    message=f"Connected to {test_data.name}, but no tools were loaded.",
                    details={"tool_count": 0},
                )

        except Exception as conn_error:
            return MCPConnectionTestResponse(
                success=False,
                message=f"Failed to connect to {test_data.name}: {str(conn_error)}",
                details={"error": str(conn_error)},
            )

    except Exception as e:
        logger.error(f"Failed to test MCP connection: {e}")
        return MCPConnectionTestResponse(
            success=False,
            message=f"Connection test failed: {str(e)}",
            details={"error": str(e)},
        )


@mcp_router.get("/transports")
def get_supported_transports() -> dict:
    """Get list of supported transport types with descriptions."""
    return {
        "transports": [
            {
                "id": "stdio",
                "name": "STDIO",
                "description": "Standard input/output transport for local processes",
                "config_fields": [
                    {
                        "name": "command",
                        "type": "string",
                        "required": True,
                        "description": "Command to execute",
                    },
                    {
                        "name": "args",
                        "type": "array",
                        "required": False,
                        "description": "Command arguments",
                    },
                    {
                        "name": "env",
                        "type": "object",
                        "required": False,
                        "description": "Environment variables",
                    },
                    {
                        "name": "cwd",
                        "type": "string",
                        "required": False,
                        "description": "Working directory",
                    },
                ],
            },
            {
                "id": "sse",
                "name": "Server-Sent Events",
                "description": "HTTP-based transport using Server-Sent Events",
                "config_fields": [
                    {
                        "name": "url",
                        "type": "string",
                        "required": True,
                        "description": "Server URL",
                    },
                    {
                        "name": "headers",
                        "type": "object",
                        "required": False,
                        "description": "HTTP headers",
                    },
                ],
            },
            {
                "id": "websocket",
                "name": "WebSocket",
                "description": "WebSocket-based transport for real-time communication",
                "config_fields": [
                    {
                        "name": "url",
                        "type": "string",
                        "required": True,
                        "description": "WebSocket URL",
                    },
                    {
                        "name": "headers",
                        "type": "object",
                        "required": False,
                        "description": "WebSocket headers",
                    },
                ],
            },
            {
                "id": "streamable_http",
                "name": "Streamable HTTP",
                "description": "HTTP transport with streaming capabilities",
                "config_fields": [
                    {
                        "name": "url",
                        "type": "string",
                        "required": True,
                        "description": "Server URL",
                    },
                    {
                        "name": "headers",
                        "type": "object",
                        "required": False,
                        "description": "HTTP headers",
                    },
                ],
            },
        ]
    }


@mcp_router.get("/servers/{server_id}/tools")
async def get_mcp_server_tools(
    server_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Get tools available from a specific MCP server."""
    try:
        user_id = current_user.id

        # Check user has access to this server
        result = (
            db.query(UserMCPServer, MCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
            .first()
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="MCP server not found",
            )

        _, server = result

        # Get connection from server
        connection = server.to_connection_dict()

        # Try to load tools
        from ...core.tools.adapters.vibe.mcp_adapter import (
            load_mcp_tools_as_agent_tools,
        )

        server_name = server.name
        if isinstance(server_name, str):
            connections_dict: Dict[str, Any] = {server_name: connection}
            tools = await load_mcp_tools_as_agent_tools(
                connections_dict, name_prefix=f"server_{server_id}_"
            )

        tools_list: List[Any] = tools if isinstance(tools, list) else []

        return {
            "server_name": server.name,
            "tool_count": len(tools_list),
            "tools": [
                {
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(
                        tool, "description", "No description available"
                    ),
                }
                for tool in tools_list
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MCP server tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCP server tools",
        )
