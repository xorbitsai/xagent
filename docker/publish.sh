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

echo "Building and pushing images with tag: ${TAG}"
echo "Target platforms: ${PLATFORMS}"

docker buildx inspect >/dev/null 2>&1 || docker buildx create --use --name xagent-builder

# Build backend
echo "Building backend image..."
docker buildx build \
  --platform "${PLATFORMS}" \
  -f "${REPO_ROOT}/docker/Dockerfile.backend" \
  -t "${BACKEND_IMAGE}:${TAG}" \
  --push \
  "${REPO_ROOT}"

# Build frontend
echo "Building frontend image..."
docker buildx build \
  --platform "${PLATFORMS}" \
  -f "${REPO_ROOT}/docker/Dockerfile.frontend" \
  -t "${FRONTEND_IMAGE}:${TAG}" \
  --push \
  "${REPO_ROOT}/frontend"

echo "Images published successfully:"
echo "  - ${BACKEND_IMAGE}:${TAG}"
echo "  - ${FRONTEND_IMAGE}:${TAG}"
