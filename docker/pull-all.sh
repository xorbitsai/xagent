#!/usr/bin/env bash
# 分步拉取 docker-compose 所需镜像，适用于网络时间受限（如每次仅十几秒）的环境。
# 用法：在项目根目录执行 ./docker/pull-all.sh，全部拉取完成后再执行 docker compose up -d

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== 在项目根目录: $REPO_ROOT"
echo "=== 将按服务逐个拉取镜像，每次拉取一个，拉完再执行下一项。"
echo ""

# 顺序：先拉体积较小/依赖少的，再拉可能较大的
SERVICES=(postgres nginx backend frontend)

for svc in "${SERVICES[@]}"; do
  echo ">>> 正在拉取服务: $svc"
  if docker compose pull "$svc"; then
    echo ">>> $svc 拉取完成"
  else
    echo ">>> $svc 拉取失败，可稍后重新运行本脚本（已拉取的不会重复拉）"
    exit 1
  fi
  echo ""
done

echo "=== 全部镜像已拉取完成。可执行: docker compose up -d"
