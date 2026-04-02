"""
Template Manager - 管理 templates 的扫描和检索
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class TemplateManager:
    """Template 系统核心管理器"""

    def __init__(self, templates_root: Path):
        """
        Args:
            templates_root: templates 目录路径
        """
        self.templates_root = Path(templates_root)

        # 确保目录存在
        self.templates_root.mkdir(parents=True, exist_ok=True)

        self._templates_cache: Dict[str, Dict] = {}
        self._initialized = False
        self._init_task: Optional[Any] = None

    async def ensure_initialized(self) -> None:
        """确保已初始化（懒加载模式）"""
        if self._initialized:
            return

        # 如果已有初始化任务在运行，等待它完成
        if self._init_task is not None:
            await self._init_task
            return

        # 创建并执行初始化任务
        self._init_task = asyncio.create_task(self._do_initialize())
        await self._init_task

    async def _do_initialize(self) -> None:
        """实际的初始化逻辑"""
        await self.initialize()
        self._init_task = None

    async def initialize(self) -> None:
        """初始化：扫描所有 templates"""
        logger.info("📂 Scanning templates...")
        logger.info(f"  from {self.templates_root}...")
        await self.reload()
        self._initialized = True
        logger.info(f"✓ Loaded {len(self._templates_cache)} templates")

    async def reload(self) -> None:
        """重新加载所有 templates"""
        self._templates_cache.clear()

        if not self.templates_root.exists():
            logger.warning(f"Templates directory does not exist: {self.templates_root}")
            return

        logger.debug(f"Scanning directory: {self.templates_root}")
        found_count = 0

        for yaml_file in self.templates_root.glob("*.yaml"):
            try:
                template_info = self._parse_yaml_file(yaml_file)
                template_id = template_info.get("id")
                if not template_id:
                    logger.warning(f"Skipping {yaml_file.name}: missing 'id' field")
                    continue

                self._templates_cache[template_id] = template_info
                logger.info(f"  ✓ Loaded: {template_info['name']}")
                found_count += 1
            except Exception as e:
                logger.error(f"  ✗ Error loading {yaml_file.name}: {e}", exc_info=True)

        logger.info(f"Total templates loaded: {len(self._templates_cache)}")

    def _parse_yaml_file(self, yaml_file: Path) -> Dict[str, Any]:
        """解析单个 YAML 文件"""
        with open(yaml_file, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        # 验证必需字段
        required_fields = ["id", "name", "category", "descriptions"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # 验证 descriptions 包含英文
        descriptions = data.get("descriptions", {})
        if not isinstance(descriptions, dict):
            raise ValueError("'descriptions' must be a dictionary")
        if "en" not in descriptions:
            raise ValueError("'descriptions' must contain at least 'en' key")

        # 确保 agent_config 存在
        if "agent_config" not in data:
            data["agent_config"] = {}

        # 设置默认值
        data.setdefault("tags", [])
        data.setdefault("features", [])
        data.setdefault("setup_time", "5 min setup")
        data.setdefault("author", "xAgent")
        data.setdefault("version", "1.0")
        data.setdefault("featured", False)

        # agent_config 默认值
        agent_config = data["agent_config"]
        agent_config.setdefault("instructions", "")
        agent_config.setdefault("skills", [])
        agent_config.setdefault("tool_categories", [])

        return data

    async def list_templates(self) -> List[Dict]:
        """列出所有 templates（简要信息）"""
        await self.ensure_initialized()
        return [
            {
                "id": template["id"],
                "name": template["name"],
                "category": template.get("category", ""),
                "featured": template.get("featured", False),
                "descriptions": template.get("descriptions", {}),
                "features": template.get("features", []),
                "setup_time": template.get("setup_time", "5 min setup"),
                "tags": template.get("tags", []),
                "author": template.get("author", ""),
                "version": template.get("version", ""),
            }
            for template in self._templates_cache.values()
        ]

    async def get_template(self, template_id: str) -> Optional[Dict]:
        """获取单个 template（完整信息）"""
        await self.ensure_initialized()
        return self._templates_cache.get(template_id)

    def has_templates(self) -> bool:
        """是否有可用的 templates"""
        return len(self._templates_cache) > 0
