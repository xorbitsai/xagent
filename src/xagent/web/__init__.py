"""xagent Web 模块

这个模块提供了xagent的Web界面，包括：
- REST API接口
- WebSocket实时通信
- 前端用户界面
- 监控和管理功能

使用方式:
    # 命令行启动
    python -m xagent.web

    # 程序中启动
    from xagent.web import run_server
    run_server(host="0.0.0.0", port=8000)
"""

from importlib.metadata import version
from typing import Any

try:
    __version__ = version("xagent")
except Exception:
    __version__ = "0.0.0+unknown"


def run_server(
    host: str = "127.0.0.1", port: int = 8000, reload: bool = False, **kwargs: Any
) -> None:
    """快速启动Web服务器

    Args:
        host: 服务器主机地址
        port: 服务器端口
        reload: 是否启用自动重载
        **kwargs: 其他uvicorn参数
    """
    import uvicorn

    uvicorn.run("xagent.web.app:app", host=host, port=port, reload=reload, **kwargs)


__all__ = ["run_server", "__version__"]
