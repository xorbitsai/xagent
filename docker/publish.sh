#!/usr/bin/env bash
# Build and push Docker images to Docker Hub
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REGISTRY="xprobe"
BACKEND_IMAGE="${REGISTRY}/xagent-backend"
FRONTEND_IMAGE="${REGISTRY}/xagent-frontend"
TAG="${1:-latest}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
PUSH="${PUSH:-${CI:-false}}"

if [[ "${PUSH}" == "true" || "${PUSH}" == "1" ]]; then
  BUILD_OUTPUT_FLAG="--push"
  ACTION_LABEL="Building and pushing"
else
  if [[ "${PLATFORMS}" == *,* ]]; then
    echo "Error: local multi-platform builds require PUSH=true."
    echo "Hint: set PUSH=true to publish, or use a single platform with --load (e.g. PLATFORMS=linux/arm64)."
    exit 1
  fi
  BUILD_OUTPUT_FLAG="--load"
  ACTION_LABEL="Building"
fi

echo "${ACTION_LABEL} images with tag: ${TAG}"
echo "Target platforms: ${PLATFORMS}"
echo "Push enabled: ${PUSH}"

docker buildx inspect >/dev/null 2>&1 || docker buildx create --use --name xagent-builder

# Build backend
echo "Building backend image..."
docker buildx build \
  --platform "${PLATFORMS}" \
  -f "${REPO_ROOT}/docker/Dockerfile.backend" \
  -t "${BACKEND_IMAGE}:${TAG}" \
  "${BUILD_OUTPUT_FLAG}" \
  "${REPO_ROOT}"

# Build frontend
echo "Building frontend image..."
docker buildx build \
  --platform "${PLATFORMS}" \
  -f "${REPO_ROOT}/docker/Dockerfile.frontend" \
  -t "${FRONTEND_IMAGE}:${TAG}" \
  "${BUILD_OUTPUT_FLAG}" \
  "${REPO_ROOT}/frontend"

if [[ "${BUILD_OUTPUT_FLAG}" == "--push" ]]; then
  echo "Images published successfully:"
else
  echo "Images built successfully:"
fi
echo "  - ${BACKEND_IMAGE}:${TAG}"
echo "  - ${FRONTEND_IMAGE}:${TAG}"
