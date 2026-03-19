#!/usr/bin/env bash
# 从离线镜像包加载镜像。
# 用法:
#   ./load-images.sh                    # 默认从 docker/offline-images/ 加载
#   ./load-images.sh <目录>             # 从指定目录加载（优先找该目录下的 xagent-images.tar）
#   ./load-images.sh <路径/to/xxx.tar>  # 直接加载单个 tar 文件
#
# 若目录下有 xagent-images.tar（save-images.sh 生成的四合一包），则只加载该文件；
# 否则加载该目录下全部 .tar。若目录下有 checksums.sha256 则先校验再加载。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARG="${1:-$SCRIPT_DIR/offline-images}"

if [[ -f "$ARG" ]]; then
  if [[ "$ARG" == *.tar ]]; then
    echo "=== 从单文件导入: $ARG"
    docker load -i "$ARG"
    echo "=== 完成。"
    exit 0
  else
    echo "错误: 不是 .tar 文件: $ARG" >&2
    exit 1
  fi
fi

IMAGES_DIR="$ARG"
if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "错误: 目录不存在: $IMAGES_DIR" >&2
  echo "请先将 save-images.sh 生成的 xagent-images.tar 放到 docker/offline-images/ 下，或传入目录/ tar 路径。" >&2
  exit 1
fi

echo "=== 校验传输完整性..."
if [[ -f "$IMAGES_DIR/checksums.sha256" ]]; then
  cd "$IMAGES_DIR"
  if sha256sum -c checksums.sha256 2>/dev/null; then
    echo "校验通过"
  else
    echo "错误: 校验失败，tar 可能传输不完整，请重新拷贝" >&2
    exit 1
  fi
  cd - > /dev/null
else
  echo "未找到 checksums.sha256，跳过校验"
fi

echo "=== 加载镜像..."
SINGLE_TAR="$IMAGES_DIR/xagent-images.tar"
if [[ -f "$SINGLE_TAR" ]]; then
  echo "  loading xagent-images.tar（四合一）..."
  docker load -i "$SINGLE_TAR"
else
  for tar in "$IMAGES_DIR"/*.tar; do
    [[ -f "$tar" ]] || continue
    echo "  loading $(basename "$tar")..."
    docker load -i "$tar"
  done
fi

echo "=== 完成。"
