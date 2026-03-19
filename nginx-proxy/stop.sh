#!/usr/bin/env bash
# 停止 nginx 反向代理进程
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/nginx-proxy.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "未找到 pid 文件，代理可能未在运行。"
  exit 0
fi
PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  MAIN_CONF="${SCRIPT_DIR}/nginx.conf.generated"
  [[ -f "$MAIN_CONF" ]] && nginx -c "$MAIN_CONF" -s stop || kill "$PID"
  echo "已停止 nginx 代理。"
else
  echo "进程 $PID 已不存在。"
  rm -f "$PID_FILE"
fi
