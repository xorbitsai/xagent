"""Initialize tool configurations in database."""

import logging
from typing import Any, Dict

from sqlalchemy.orm import Session

from .models.database import get_db
from .models.tool_config import ToolConfig

logger = logging.getLogger(__name__)


def get_default_tool_configs() -> list[Dict[str, Any]]:
    """Get default tool configurations."""
    return [
        {
            "tool_name": "python_executor",
            "tool_type": "builtin",
            "category": "development",
            "display_name": "Python 代码执行器",
            "description": "执行 Python 代码，支持数据分析和计算任务",
            "enabled": True,
            "config": {"working_dir": "/workspace/output"},
            "dependencies": [],
        },
        {
            "tool_name": "bash",
            "tool_type": "builtin",
            "category": "development",
            "display_name": "Bash 命令执行器",
            "description": "执行 Bash 命令，进行系统操作和文件管理",
            "enabled": True,
            "config": {},
            "dependencies": [],
        },
        {
            "tool_name": "web_search",
            "tool_type": "builtin",
            "category": "search",
            "display_name": "网络搜索",
            "description": "使用 Google 搜索引擎进行网络搜索",
            "enabled": True,
            "config": {
                "provider": "google",
                "api_key_env": "GOOGLE_API_KEY",
                "cse_id_env": "GOOGLE_CSE_ID",
            },
            "dependencies": ["google_api_key", "google_cse_id"],
        },
        {
            "tool_name": "tavily_web_search",
            "tool_type": "builtin",
            "category": "search",
            "display_name": "Tavily 网络搜索",
            "description": "使用 Tavily Search API 进行网络搜索，无需 Google 账号",
            "enabled": True,
            "config": {
                "provider": "tavily",
                "api_key_env": "TAVILY_API_KEY",
            },
            "dependencies": ["tavily_api_key"],
        },
        {
            "tool_name": "zhipu_web_search",
            "tool_type": "builtin",
            "category": "search",
            "display_name": "智谱网络搜索",
            "description": "使用智谱 Web Search API 进行网络搜索",
            "enabled": True,
            "config": {
                "provider": "zhipu",
                "api_key_env": "ZHIPU_API_KEY",
                "base_url_env": "ZHIPU_BASE_URL",
            },
            "dependencies": ["zhipu_api_key"],
        },
        {
            "tool_name": "fetch_webpage",
            "tool_type": "builtin",
            "category": "search",
            "display_name": "网页内容获取",
            "description": "获取网页内容，支持文本提取和解析",
            "enabled": True,
            "config": {},
            "dependencies": [],
        },
        {
            "tool_name": "calculator",
            "tool_type": "builtin",
            "category": "development",
            "display_name": "计算器",
            "description": "进行数学计算和表达式求值",
            "enabled": True,
            "config": {},
            "dependencies": [],
        },
        {
            "tool_name": "read_file",
            "tool_type": "file",
            "category": "file_operations",
            "display_name": "文件读取",
            "description": "读取文件内容，支持多种文件格式",
            "enabled": True,
            "config": {},
            "dependencies": [],
        },
        {
            "tool_name": "write_file",
            "tool_type": "file",
            "category": "file_operations",
            "display_name": "文件写入",
            "description": "写入文件内容，支持多种文件格式",
            "enabled": True,
            "config": {},
            "dependencies": [],
        },
        {
            "tool_name": "list_directory",
            "tool_type": "file",
            "category": "file_operations",
            "display_name": "目录列表",
            "description": "列出目录内容，支持文件过滤和排序",
            "enabled": True,
            "config": {},
            "dependencies": [],
        },
        {
            "tool_name": "vision_tool",
            "tool_type": "vision",
            "category": "ai_tools",
            "display_name": "视觉分析工具",
            "description": "分析图像内容，支持图像识别和理解",
            "enabled": True,
            "config": {},
            "dependencies": ["vision_model"],
        },
        {
            "tool_name": "image_generation",
            "tool_type": "image",
            "category": "ai_tools",
            "display_name": "图像生成",
            "description": "根据文本描述生成图像",
            "enabled": True,
            "config": {},
            "dependencies": ["image_model"],
        },
    ]


def init_tool_configs(db: Session) -> None:
    """Initialize tool configurations in database."""
    logger.info("Initializing tool configurations...")

    default_configs = get_default_tool_configs()

    for config_data in default_configs:
        # Check if tool config already exists
        existing = (
            db.query(ToolConfig)
            .filter(ToolConfig.tool_name == config_data["tool_name"])
            .first()
        )

        if existing:
            logger.info(
                f"Tool config '{config_data['tool_name']}' already exists, skipping..."
            )
            continue

        # Create new tool config
        tool_config = ToolConfig(**config_data)
        db.add(tool_config)
        logger.info(f"Added tool config: {config_data['tool_name']}")

    db.commit()
    logger.info("Tool configurations initialized successfully.")


def main() -> None:
    """Main function to initialize tool configurations."""
    logging.basicConfig(level=logging.INFO)

    db = next(get_db())
    try:
        init_tool_configs(db)
        print("Tool configurations initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize tool configurations: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
