#!/usr/bin/env bash
# 使用 Xvfb 虚拟显示启动后端，使 Playwright 浏览器/网页截图在无显示器环境可用
set -euo pipefail
exec xvfb-run -a /opt/xagent/deploy/entrypoint.sh
