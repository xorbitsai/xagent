"""Centralized registry for MCP Applications and OAuth Providers.

This module provides a scalable structure for defining supported MCP applications,
their OAuth configurations, and server launch configurations.
"""

from typing import Any, Dict, List

from sqlalchemy.orm import Session

from .models.public_mcp import PublicMCPApp


def _app_to_dict(app: PublicMCPApp) -> Dict[str, Any]:
    return {
        "id": app.app_id,
        "name": app.name,
        "description": app.description,
        "icon": app.icon,
        "transport": app.transport,
        "provider": app.provider_name,
        "category": app.category,
        "oauth_scopes": app.oauth_scopes or [],
        "launch_config": app.launch_config or {},
    }


def get_all_mcp_apps(db: Session) -> List[Dict[str, Any]]:
    """Retrieve all MCP apps from the database dynamically."""
    apps = db.query(PublicMCPApp).all()
    return [_app_to_dict(app) for app in apps]


def get_app_by_id(db: Session, app_id: str) -> Dict[str, Any] | None:
    """Retrieve an MCP app configuration by its ID."""
    app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == app_id).first()
    return _app_to_dict(app) if app else None


def get_app_by_name(db: Session, name: str) -> Dict[str, Any] | None:
    """Retrieve an MCP app configuration by its exact name."""
    app = db.query(PublicMCPApp).filter(PublicMCPApp.name == name).first()
    return _app_to_dict(app) if app else None
