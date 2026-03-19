#!/usr/bin/env bash
# 在「有外网」的机器上执行：拉取 compose 所需镜像并导出为一个 tar 包。
# 将生成的 xagent-images.tar 拷到本机的 docker/offline-images/ 目录后，在本机执行 ./docker/load-images.sh
#
# 目标机器是 x86_64/AMD64 时，请指定平台再拉取，避免拷过去的是 ARM 镜像无法运行：
#   PLATFORM=linux/amd64 ./docker/save-images.sh
# 或  ./docker/save-images.sh xagent-images.tar linux/amd64

set -e
OUTPUT="${1:-xagent-images.tar}"
PLATFORM="${PLATFORM:-$2}"

PULL_EXTRA=()
[[ -n "$PLATFORM" ]] && PULL_EXTRA=(--platform "$PLATFORM") && echo "=== 目标平台: $PLATFORM"

echo "=== 拉取镜像..."
docker pull "${PULL_EXTRA[@]}" nginx:latest
docker pull "${PULL_EXTRA[@]}" xprobe/xagent-frontend:latest
docker pull "${PULL_EXTRA[@]}" xprobe/xagent-backend:latest
docker pull "${PULL_EXTRA[@]}" postgres:16-bookworm

echo "=== 导出为 $OUTPUT ..."
docker save -o "$OUTPUT" \
  nginx:latest \
  xprobe/xagent-frontend:latest \
  xprobe/xagent-backend:latest \
  postgres:16-bookworm

echo "=== 完成。请将 $OUTPUT 拷贝到本机项目目录下的 docker/offline-images/ 中，然后执行 ./docker/load-images.sh"
