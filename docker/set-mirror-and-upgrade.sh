#!/usr/bin/env bash
# 1) 配置 Docker 镜像源为 https://docker.1ms.run
# 2) 重启 Docker 使配置生效
# 3) 拉取最新 xagent 镜像并升级
# 用法：在 xagent 项目根目录执行 ./docker/set-mirror-and-upgrade.sh（会提示输入 sudo 密码）

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MIRROR="https://docker.1ms.run"
DAEMON_JSON="/etc/docker/daemon.json"

echo "=== 1) 配置 Docker 镜像源为 $MIRROR"
if [[ -f "$DAEMON_JSON" ]]; then
  sudo cp -a "$DAEMON_JSON" "${DAEMON_JSON}.bak.$(date +%Y%m%d%H%M%S)"
  echo "    已备份原配置到 ${DAEMON_JSON}.bak.*"
fi
echo "{\"registry-mirrors\": [\"$MIRROR\"]}" | sudo tee "$DAEMON_JSON" > /dev/null
echo "    已写入 $DAEMON_JSON"

echo ""
echo "=== 2) 重启 Docker 使镜像源生效"
sudo systemctl restart docker
echo "    已重启 Docker"
sleep 2
if docker info 2>/dev/null | grep -q "Registry Mirrors"; then
  echo "    当前镜像源："
  docker info 2>/dev/null | grep -A 3 "Registry Mirrors" || true
else
  echo "    (无法在此环境显示 Registry Mirrors，请在本机执行: docker info | grep -A 5 Registry)"
fi

echo ""
echo "=== 3) 拉取最新 xagent 镜像并升级（使用上述镜像源）"
docker compose pull
echo ""
echo "=== 4) 重新创建并启动容器（升级到最新版）"
docker compose up -d
echo ""
echo "=== 完成。xagent 已使用镜像源 $MIRROR 升级到最新版。"
echo "    访问: http://localhost:${PORT:-80}"
