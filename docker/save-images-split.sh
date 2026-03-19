#!/usr/bin/env bash
# 在「有外网」的机器上执行：拉取镜像并**按镜像分别**导出为多个 .tar，便于单独校验与补拷。
# 用法: ./save-images-split.sh [输出目录]
#  默认输出到当前目录下的 xagent-images/；可传 PLATFORM=linux/amd64 指定平台。
# 生成后拷贝整个目录到本机 docker/offline-images/，在本机执行 ./load-images.sh [该目录名]

set -e
OUT_DIR="${1:-xagent-images}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

PLATFORM="${PLATFORM:-}"
PULL_EXTRA=()
[[ -n "$PLATFORM" ]] && PULL_EXTRA=(--platform "$PLATFORM") && echo "=== 目标平台: $PLATFORM"

echo "=== 拉取并分别导出..."
docker pull "${PULL_EXTRA[@]}" nginx:latest
docker save -o nginx-latest.tar nginx:latest

docker pull "${PULL_EXTRA[@]}" postgres:16-bookworm
docker save -o postgres-16-bookworm.tar postgres:16-bookworm

docker pull "${PULL_EXTRA[@]}" xprobe/xagent-frontend:latest
docker save -o xprobe-xagent-frontend-latest.tar xprobe/xagent-frontend:latest

docker pull "${PULL_EXTRA[@]}" xprobe/xagent-backend:latest
docker save -o xprobe-xagent-backend-latest.tar xprobe/xagent-backend:latest

echo "=== 生成校验和..."
sha256sum *.tar > checksums.sha256
echo "=== 完成。请将目录 $OUT_DIR 拷贝到本机 docker/ 下（如 docker/offline-images），再执行 ./load-images.sh"
echo "  若仅 backend 损坏，可只重新执行上面 backend 的 pull+save，覆盖 xprobe-xagent-backend-latest.tar 后重新生成 checksums.sha256"
