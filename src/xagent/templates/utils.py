"""
Template utilities - 创建 template_manager 的工具函数
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .manager import TemplateManager


def create_template_manager(templates_root: Optional[Path] = None) -> "TemplateManager":
    """
    创建 template_manager（不初始化）

    Args:
        templates_root: 可选的 templates 目录路径，默认使用内置路径

    Returns:
        TemplateManager 实例（未初始化）
    """
    from .manager import TemplateManager

    # 默认的 templates 目录
    if templates_root is None:
        templates_root = Path(__file__).parent / "built_in"

    # 创建 template_manager（不初始化）
    template_manager = TemplateManager(templates_root=templates_root)

    return template_manager
