"""Tool Management API Route Handlers"""

import asyncio
import logging
from datetime import datetime
from typing import Any, DefaultDict, Dict, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..init_tool_configs import get_default_tool_configs
from ..models.database import get_db
from ..models.tool_config import ToolConfig, ToolUsage
from ..models.user import User
from ..services.tool_credentials import (
    TOOL_CREDENTIAL_SPECS,
    clear_tool_credential,
    delete_sql_connection,
    get_tool_credential_view,
    list_configurable_tool_names,
    list_sql_connections,
    set_sql_connection,
    set_tool_credentials,
)
from ..tools.config import WebToolConfig

logger = logging.getLogger(__name__)


def _require_user_id(current_user: User) -> int:
    user_id: object = getattr(current_user, "id", None)
    if isinstance(user_id, int):
        return user_id
    raise HTTPException(status_code=500, detail="Authenticated user is missing an id")


# Category display names (for frontend display)
CATEGORY_DISPLAY_NAMES = {
    "vision": "Vision",
    "image": "Image",
    "audio": "Audio",
    "knowledge": "Knowledge",
    "file": "File",
    "basic": "Basic",
    "browser": "Browser",
    "ppt": "PPT",
    "agent": "Agent",
    "mcp": "MCP",
    "skill": "Skill",
    "other": "Other",
}

# 创建路由器
tools_router = APIRouter(prefix="/api/tools", tags=["tools"])


class CredentialFieldUpdate(BaseModel):
    value: str


class ToolCredentialUpdateRequest(BaseModel):
    credentials: Dict[str, CredentialFieldUpdate]


class ToolEnableUpdateRequest(BaseModel):
    enabled: bool


class SqlConnectionUpsertRequest(BaseModel):
    connection_url: str


def _create_tool_info(
    tool: Any,
    category: str,
    vision_model: Any = None,
    image_models: Any = None,
    asr_models: Any = None,
    tts_models: Any = None,
) -> Dict[str, Any]:
    """Create tool information based on category instead of hardcoded names"""
    tool_name = getattr(tool, "name", tool.__class__.__name__)

    # 基于类别设置状态和类型信息
    status = "available"
    status_reason = None
    enabled = True
    tool_type = "basic"

    if category == "vision":
        tool_type = "vision"
        # vision tool depends on vision model
        if not vision_model:
            status = "missing_model"
            status_reason = (
                "Vision model not configured, "
                "please add a vision model in model management page"
            )
            enabled = False

    elif category == "image":
        tool_type = "image"
        # image tool depends on image models
        if not image_models:
            status = "missing_model"
            status_reason = (
                "Image model not configured, please add an "
                "image generation model in model management page"
            )
            enabled = False
        elif tool_name == "edit_image":
            # Special check for image editing capability
            has_edit_capability = any(
                "edit" in model.abilities for model in image_models.values()
            )
            if not has_edit_capability:
                status = "missing_capability"
                status_reason = (
                    "Current image model does not support editing, "
                    "please add an image model with editing support"
                )
                enabled = False

    elif category == "audio":
        tool_type = "audio"
        # audio tool depends on ASR/TTS models
        if not asr_models and not tts_models:
            status = "missing_model"
            status_reason = (
                "Audio model not configured, please add an "
                "ASR or TTS model in model management page"
            )
            enabled = False
        elif tool_name == "transcribe_audio" and not asr_models:
            status = "missing_model"
            status_reason = (
                "ASR model not configured, please add a "
                "speech recognition model in model management page"
            )
            enabled = False
        elif tool_name == "synthesize_speech" and not tts_models:
            status = "missing_model"
            status_reason = (
                "TTS model not configured, please add a "
                "text-to-speech model in model management page"
            )
            enabled = False

    elif category == "file":
        tool_type = "file"
    elif category == "knowledge":
        tool_type = "knowledge"
    elif category == "special_image":
        tool_type = "image"
    elif category == "mcp":
        tool_type = "mcp"
        # Extract server name from tool name (format: server_name_tool_name)
        # MCP tools are prefixed with server name
        parts = tool_name.split("_", 1)
        if len(parts) > 1:
            server_name = parts[0]
            # Add server info to description if available
            description = getattr(tool, "description", "")
            if server_name and f"[MCP Server: {server_name}]" not in description:
                # Server name is already in description from mcp_adapter
                pass
    elif category == "ppt":
        tool_type = "office"
    elif category == "browser":
        tool_type = "browser"
    elif category == "agent":
        tool_type = "agent"
    elif category == "skill":
        tool_type = "skill"

    return {
        "name": tool_name,
        "description": getattr(tool, "description", ""),
        "type": tool_type,
        "category": category,
        "display_category": CATEGORY_DISPLAY_NAMES.get(category, category.capitalize()),
        "enabled": enabled,
        "requires_configuration": False,
        "status": status,
        "status_reason": status_reason,
        "config": {},
        "dependencies": [],
    }


@tools_router.get("/available")
async def get_available_tools(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get list of all available tools, including MCP tools.

    Tools are self-describing - each tool declares its own category via
    metadata.category field. No manual category mapping needed.
    """

    # Create a temporary request object (simulating WebToolConfig requirements)
    class MockRequest:
        def __init__(self) -> None:
            self.credentials: Optional[Any] = None

    # Create WebToolConfig, now includes MCP tools
    # Note: llm=None for tool listing (display only, no execution)
    current_user_id = _require_user_id(current_user)
    tool_config = WebToolConfig(
        db=db,
        request=MockRequest(),
        user_id=current_user_id,
        is_admin=bool(current_user.is_admin),
        llm=None,  # Not needed for tool listing
        workspace_config={
            "base_dir": "./uploads",
            "task_id": "tools_list",  # Use a generic task ID for workspace creation
        },
        include_mcp_tools=True,  # Enable MCP tools
        task_id="tools_list",  # Generic task ID for tool listing
        browser_tools_enabled=True,  # Enable browser automation tools
    )

    # Use ToolFactory.create_all_tools() to get all tools
    # This ensures consistency between backend execution and frontend display
    from ...core.tools.adapters.vibe.factory import ToolFactory

    all_tools = await ToolFactory.create_all_tools(tool_config)

    # Helper function to get category from tool's metadata
    def get_tool_category(tool: Any) -> str:
        """Get category from tool's self-describing metadata.

        Tools declare their category via the category class attribute.
        """
        try:
            metadata = getattr(tool, "metadata", None)
            category = getattr(metadata, "category", None)
            value = getattr(category, "value", None)
            if isinstance(value, str) and value:
                return value
        except Exception:
            logger.warning(
                "Failed to read tool category from metadata for %s",
                getattr(tool, "name", tool.__class__.__name__),
                exc_info=True,
            )
        return "other"

    # Get models for tool status checking
    vision_model = tool_config.get_vision_model()
    image_models = tool_config.get_image_models()
    asr_models = tool_config.get_asr_models()
    tts_models = tool_config.get_tts_models()

    # Convert tools to API format with category information
    tools: List[Dict[str, Any]] = []
    for tool in all_tools:
        category = get_tool_category(tool)
        tools.append(
            _create_tool_info(
                tool,
                category,
                vision_model,
                image_models,
                asr_models,
                tts_models,
            )
        )

    # Calculate tool usage count from ToolUsage table (execution stats)
    from collections import defaultdict

    usage_map: DefaultDict[str, int] = defaultdict(int)
    try:
        usage_stats: List[Any] = db.query(ToolUsage).all()
        for stat in usage_stats:
            usage_map[stat.tool_name] = stat.usage_count
    except Exception as e:
        logger.error(f"Failed to fetch tool usage stats: {e}")

    # Add usage_count to tools
    for tool_item in tools:
        tool_name = tool_item.get("name", "")
        tool_item["usage_count"] = usage_map[tool_name]

    default_configs = {item["tool_name"]: item for item in get_default_tool_configs()}
    config_rows = db.query(ToolConfig).all()
    enabled_map: dict[str, bool] = {}
    requires_configuration_map: dict[str, bool] = {
        tool_name: bool(config.get("requires_configuration", False))
        for tool_name, config in default_configs.items()
    }
    for row in config_rows:
        row_tool_name = cast(Any, row.tool_name)
        if isinstance(row_tool_name, str):
            enabled_map[row_tool_name] = bool(cast(Any, row.enabled))
            requires_configuration_map[row_tool_name] = bool(
                cast(Any, getattr(row, "requires_configuration", False))
            )
    for tool_item in tools:
        tool_name = str(tool_item.get("name") or "")
        if tool_name in enabled_map:
            tool_item["enabled"] = enabled_map[tool_name]
            if not enabled_map[tool_name]:
                tool_item["status"] = "disabled"
                tool_item["status_reason"] = "Disabled by tool policy"
        requires_configuration = requires_configuration_map.get(tool_name, False)
        if not requires_configuration and tool_item.get("category") == "database":
            requires_configuration = requires_configuration_map.get("sql_query", False)
        tool_item["requires_configuration"] = requires_configuration

    return {
        "tools": tools,
        "count": len(tools),
    }


@tools_router.get("/configurable")
async def get_configurable_tools(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if not bool(current_user.is_admin):
        raise HTTPException(status_code=403, detail="Admin privileges required")

    items: List[Dict[str, Any]] = []
    for tool_name in list_configurable_tool_names():
        view = get_tool_credential_view(db, tool_name)
        items.append(
            {
                "tool_name": tool_name,
                "display_name": view.get("display_name", tool_name),
                "configured": view["configured"],
                "fields": view["fields"],
            }
        )

    return {
        "tools": items,
        "count": len(items),
    }


@tools_router.get("/sql-connections")
async def get_sql_connections(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    items = list_sql_connections(db, _require_user_id(current_user))
    return {
        "connections": items,
        "count": len(items),
    }


@tools_router.put("/sql-connections/{name}")
async def upsert_sql_connection(
    name: str,
    payload: SqlConnectionUpsertRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    current_user_id = _require_user_id(current_user)
    try:
        set_sql_connection(db, current_user_id, name, payload.connection_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "connections": list_sql_connections(db, current_user_id),
    }


@tools_router.delete("/sql-connections/{name}")
async def remove_sql_connection(
    name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    current_user_id = _require_user_id(current_user)
    delete_sql_connection(db, current_user_id, name)
    return {
        "connections": list_sql_connections(db, current_user_id),
    }


@tools_router.put("/{tool_name}/enabled")
async def update_tool_enabled(
    tool_name: str,
    payload: ToolEnableUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if not bool(current_user.is_admin):
        raise HTTPException(status_code=403, detail="Admin privileges required")

    config_row = db.query(ToolConfig).filter(ToolConfig.tool_name == tool_name).first()
    if not config_row:
        defaults = {item["tool_name"]: item for item in get_default_tool_configs()}
        default_data = defaults.get(tool_name)
        if default_data:
            config_row = ToolConfig(**default_data)
        else:
            config_row = ToolConfig(
                tool_name=tool_name,
                tool_type="builtin",
                category="other",
                display_name=tool_name,
                description="",
                enabled=payload.enabled,
                config={},
                dependencies=[],
                status="available",
            )
        db.add(config_row)

    cast(Any, config_row).enabled = payload.enabled
    db.add(config_row)
    db.commit()
    return {
        "tool_name": tool_name,
        "enabled": bool(cast(Any, config_row).enabled),
    }


@tools_router.get("/{tool_name}/credentials")
async def get_tool_credentials(
    tool_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if not bool(current_user.is_admin):
        raise HTTPException(status_code=403, detail="Admin privileges required")

    if tool_name not in TOOL_CREDENTIAL_SPECS:
        raise HTTPException(
            status_code=404, detail=f"Tool '{tool_name}' is not configurable"
        )

    return get_tool_credential_view(db, tool_name)


@tools_router.put("/{tool_name}/credentials")
async def update_tool_credentials(
    tool_name: str,
    payload: ToolCredentialUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if not bool(current_user.is_admin):
        raise HTTPException(status_code=403, detail="Admin privileges required")

    if tool_name not in TOOL_CREDENTIAL_SPECS:
        raise HTTPException(
            status_code=404, detail=f"Tool '{tool_name}' is not configurable"
        )

    updates = {
        field_name: field_update.value
        for field_name, field_update in payload.credentials.items()
    }

    try:
        set_tool_credentials(db, tool_name, updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return get_tool_credential_view(db, tool_name)


@tools_router.delete("/{tool_name}/credentials/{field_name}")
async def delete_tool_credential(
    tool_name: str,
    field_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if not bool(current_user.is_admin):
        raise HTTPException(status_code=403, detail="Admin privileges required")

    if tool_name not in TOOL_CREDENTIAL_SPECS:
        raise HTTPException(
            status_code=404, detail=f"Tool '{tool_name}' is not configurable"
        )

    if field_name not in TOOL_CREDENTIAL_SPECS[tool_name]:
        raise HTTPException(
            status_code=404, detail=f"Field '{field_name}' is not configurable"
        )

    clear_tool_credential(db, tool_name, field_name)
    return get_tool_credential_view(db, tool_name)


@tools_router.get("/usage")
async def get_tool_usage(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Get tool usage statistics"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _get_tool_usage_sync() -> List[Dict[str, Any]]:
            usage_stats = db.query(ToolUsage).all()

            result = []
            for stat in usage_stats:
                usage_count = cast(int, getattr(stat, "usage_count", 0) or 0)
                success_count = cast(int, getattr(stat, "success_count", 0) or 0)
                error_count = cast(int, getattr(stat, "error_count", 0) or 0)
                last_used_at = getattr(stat, "last_used_at", None)
                result.append(
                    {
                        "tool_name": stat.tool_name,
                        "usage_count": usage_count,
                        "success_count": success_count,
                        "error_count": error_count,
                        "success_rate": (success_count / usage_count * 100)
                        if usage_count > 0
                        else 0,
                        "last_used_at": last_used_at.isoformat()
                        if isinstance(last_used_at, datetime)
                        else None,
                    }
                )

            return result

        # Execute in thread pool to avoid blocking
        return await asyncio.to_thread(_get_tool_usage_sync)

    except Exception as e:
        logger.error(f"Get tool usage failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
