"""Tool Management API Route Handlers"""

import asyncio
import logging
from typing import Any, DefaultDict, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.tool_config import ToolUsage
from ..models.user import User
from ..tools.config import WebToolConfig

logger = logging.getLogger(__name__)

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
    tool_config = WebToolConfig(
        db=db,
        request=MockRequest(),
        user_id=int(current_user.id),
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
        return str(tool.metadata.category.value)

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
