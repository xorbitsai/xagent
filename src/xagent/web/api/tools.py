"""Tool Management API Route Handlers"""

import asyncio
import logging
from typing import Any, DefaultDict, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...core.tools.adapters.vibe.image_tool import create_image_tool
from ...core.tools.adapters.vibe.vision_tool import get_vision_tool
from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.tool_config import ToolUsage
from ..models.user import User
from ..tools.config import WebToolConfig

logger = logging.getLogger(__name__)

# 创建路由器
tools_router = APIRouter(prefix="/api/tools", tags=["tools"])


def _create_tool_info(
    tool: Any, category: str, vision_model: Any = None, image_models: Any = None
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
            status_reason = "Vision model not configured, please add a vision model in model management page"
            enabled = False

    elif category == "image":
        tool_type = "image"
        # image tool depends on image models
        if not image_models:
            status = "missing_model"
            status_reason = "Image model not configured, please add an image generation model in model management page"
            enabled = False
        elif tool_name == "edit_image":
            # Special check for image editing capability
            has_edit_capability = any(
                "edit" in model.abilities for model in image_models.values()
            )
            if not has_edit_capability:
                status = "missing_capability"
                status_reason = "Current image model does not support editing, please add an image model with editing support"
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

    return {
        "name": tool_name,
        "description": getattr(tool, "description", ""),
        "type": tool_type,
        "category": category,
        "enabled": enabled,
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
    """Get list of all available tools, including MCP tools"""

    # Create a temporary request object (simulating WebToolConfig requirements)
    class MockRequest:
        def __init__(self) -> None:
            self.credentials: Optional[Any] = None

    # Create WebToolConfig, now includes MCP tools
    tool_config = WebToolConfig(
        db=db,
        request=MockRequest(),
        user_id=int(current_user.id),
        is_admin=bool(current_user.is_admin),
        workspace_config={
            "base_dir": "./uploads",
            "task_id": None,  # No task ID for listing available tools
        },
        include_mcp_tools=True,  # Enable MCP tools
        task_id=None,  # No task ID needed for tool listing
        browser_tools_enabled=True,  # Enable browser automation tools
    )

    # Use ToolFactory to create all tools and track their source category
    tools: List[Dict[str, Any]] = []

    # Create workspace for tool creation
    from ...core.tools.adapters.vibe.factory import ToolFactory
    from ...core.workspace import WorkspaceManager

    workspace_manager = WorkspaceManager()
    workspace = workspace_manager.get_or_create_workspace(
        "./uploads",
        "tools_list",  # Generic workspace for listing tools
    )

    # Basic tools (always enabled if config allows)
    if tool_config.get_basic_tools_enabled():
        basic_tools = ToolFactory._create_basic_tools(workspace)
        for tool in basic_tools:
            tools.append(_create_tool_info(tool, "basic"))

    # Knowledge base tools
    knowledge_tools = ToolFactory._create_knowledge_tools()
    for tool in knowledge_tools:
        tools.append(_create_tool_info(tool, "knowledge"))

    # File tools (workspace-bound)
    if tool_config.get_file_tools_enabled() and workspace:
        file_tools = ToolFactory._create_file_tools(workspace)
        for tool in file_tools:
            tools.append(_create_tool_info(tool, "file"))

    # Vision tools
    vision_model = tool_config.get_vision_model()
    if vision_model:
        vision_tools = get_vision_tool(vision_model=vision_model, workspace=workspace)
        for tool in vision_tools:
            tools.append(_create_tool_info(tool, "vision", vision_model=vision_model))

    # Image tools
    image_models = tool_config.get_image_models()
    if image_models:
        default_generate_model = tool_config.get_image_generate_model()
        default_edit_model = tool_config.get_image_edit_model()
        image_tools = create_image_tool(
            image_models,
            workspace=workspace,
            default_generate_model=default_generate_model,
            default_edit_model=default_edit_model,
        )
        for tool in image_tools:
            tools.append(_create_tool_info(tool, "image", image_models=image_models))

    # Special image tools (workspace-bound)
    if workspace:
        special_image_tools = ToolFactory._create_special_image_tools(workspace)
        for tool in special_image_tools:
            tools.append(_create_tool_info(tool, "special_image"))

    # MCP tools - Create using ToolFactory
    try:
        mcp_configs = tool_config.get_mcp_server_configs()
        logger.info(f"mcp config: {mcp_configs}")
        if mcp_configs:
            logger.info(f"Loading MCP tools from {len(mcp_configs)} servers")
            mcp_tools = await ToolFactory._create_mcp_tools_from_configs(mcp_configs)
            logger.info(f"Loaded {len(mcp_tools)} MCP tools")
            for tool in mcp_tools:
                tools.append(_create_tool_info(tool, "mcp"))
    except Exception as e:
        logger.error(f"Failed to load MCP tools: {e}", exc_info=True)
        # Continue without MCP tools rather than failing the entire request

    # Browser automation tools
    try:
        if tool_config.get_browser_tools_enabled():
            task_id = tool_config.get_task_id()
            browser_tools = ToolFactory._create_browser_tools(task_id, workspace)
            logger.info(f"Loaded {len(browser_tools)} browser automation tools")
            for tool in browser_tools:
                tools.append(_create_tool_info(tool, "browser"))
    except Exception as e:
        logger.error(f"Failed to load browser tools: {e}", exc_info=True)
        # Continue without browser tools rather than failing the entire request

    # Published agent tools
    try:
        if tool_config.get_enable_agent_tools():
            task_id = tool_config.get_task_id()
            agent_tools = ToolFactory._create_agent_tools(
                db=db, user_id=int(current_user.id), task_id=task_id
            )
            logger.info(f"Loaded {len(agent_tools)} published agent tools")
            for tool in agent_tools:
                tools.append(_create_tool_info(tool, "agent"))
    except Exception as e:
        logger.error(f"Failed to load agent tools: {e}", exc_info=True)
        # Continue without agent tools rather than failing the entire request

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

    return {
        "tools": tools,
        "count": len(tools),
    }


@tools_router.get("/usage")
async def get_tool_usage(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Get tool usage statistics"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _get_tool_usage_sync() -> List[Dict[str, Any]]:
            usage_stats = db.query(ToolUsage).all()

            result = []
            for stat in usage_stats:
                result.append(
                    {
                        "tool_name": stat.tool_name,
                        "usage_count": stat.usage_count,
                        "success_count": stat.success_count,
                        "error_count": stat.error_count,
                        "success_rate": (stat.success_count / stat.usage_count * 100)
                        if stat.usage_count > 0
                        else 0,
                        "last_used_at": stat.last_used_at.isoformat()
                        if stat.last_used_at
                        else None,
                    }
                )

            return result

        # Execute in thread pool to avoid blocking
        return await asyncio.to_thread(_get_tool_usage_sync)

    except Exception as e:
        logger.error(f"Get tool usage failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
