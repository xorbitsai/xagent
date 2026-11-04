#!/usr/bin/env bash
# Build and push Docker images to Docker Hub
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REGISTRY="xprobe"
BACKEND_IMAGE="${REGISTRY}/xagent-backend"
FRONTEND_IMAGE="${REGISTRY}/xagent-frontend"
TAG="${1:-latest}"
GIT_COMMIT="${XAGENT_GIT_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)}"
GIT_COMMIT="${GIT_COMMIT:-local}"
DEFAULT_PACKAGE_VERSION="${TAG#v}"
if [[ ! "${DEFAULT_PACKAGE_VERSION}" =~ ^[0-9]+([.][0-9]+)*([a-zA-Z0-9.+-]*)?$ ]]; then
  DEFAULT_PACKAGE_VERSION="0.0.0+${GIT_COMMIT::12}"
fi
PACKAGE_VERSION="${XAGENT_PACKAGE_VERSION:-${DEFAULT_PACKAGE_VERSION}}"
XAGENT_VERSION="${XAGENT_VERSION:-${TAG}}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
PUSH="${PUSH:-${CI:-false}}"
CACHE="${CACHE:-true}"
WRITE_CACHE="${WRITE_CACHE:-${PUSH}}"
INLINE_CACHE="${INLINE_CACHE:-${PUSH}}"
INLINE_CACHE_FROM_TAG="${INLINE_CACHE_FROM_TAG:-latest}"
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-${REPO_ROOT}/.docker-build-cache}"
REGISTRY_CACHE="${REGISTRY_CACHE:-${PUSH}}"
BACKEND_CACHE_IMAGE="${BACKEND_CACHE_IMAGE:-${BACKEND_IMAGE}:buildcache}"
FRONTEND_CACHE_IMAGE="${FRONTEND_CACHE_IMAGE:-${FRONTEND_IMAGE}:buildcache}"
CACHE_MODE="${CACHE_MODE:-max}"
BACKEND_CACHE_MODE="${BACKEND_CACHE_MODE:-${CACHE_MODE}}"
FRONTEND_CACHE_MODE="${FRONTEND_CACHE_MODE:-${CACHE_MODE}}"

BACKEND_CACHE_FROM=()
BACKEND_CACHE_TO=()
FRONTEND_CACHE_FROM=()
FRONTEND_CACHE_TO=()
BACKEND_BUILD_ARGS=(
  --build-arg "XAGENT_VERSION=${XAGENT_VERSION}"
  --build-arg "XAGENT_PACKAGE_VERSION=${PACKAGE_VERSION}"
  --build-arg "XAGENT_GIT_COMMIT=${GIT_COMMIT}"
  --build-arg "XAGENT_BUILD_TIME=${XAGENT_BUILD_TIME:-}"
)

if [[ "${CACHE}" == "true" || "${CACHE}" == "1" ]]; then
  if [[ "${INLINE_CACHE}" == "true" || "${INLINE_CACHE}" == "1" ]]; then
    BACKEND_CACHE_FROM+=(--cache-from "type=registry,ref=${BACKEND_IMAGE}:${TAG}")
    FRONTEND_CACHE_FROM+=(--cache-from "type=registry,ref=${FRONTEND_IMAGE}:${TAG}")
    if [[ "${INLINE_CACHE_FROM_TAG}" != "${TAG}" ]]; then
      BACKEND_CACHE_FROM+=(--cache-from "type=registry,ref=${BACKEND_IMAGE}:${INLINE_CACHE_FROM_TAG}")
      FRONTEND_CACHE_FROM+=(--cache-from "type=registry,ref=${FRONTEND_IMAGE}:${INLINE_CACHE_FROM_TAG}")
    fi
    BACKEND_CACHE_TO+=(--cache-to "type=inline")
    FRONTEND_CACHE_TO+=(--cache-to "type=inline")
  fi
  if [[ "${REGISTRY_CACHE}" == "true" || "${REGISTRY_CACHE}" == "1" ]]; then
    BACKEND_CACHE_FROM+=(--cache-from "type=registry,ref=${BACKEND_CACHE_IMAGE}")
    FRONTEND_CACHE_FROM+=(--cache-from "type=registry,ref=${FRONTEND_CACHE_IMAGE}")
    if [[ "${WRITE_CACHE}" == "true" || "${WRITE_CACHE}" == "1" ]]; then
      BACKEND_CACHE_TO+=(--cache-to "type=registry,ref=${BACKEND_CACHE_IMAGE},mode=${BACKEND_CACHE_MODE}")
      FRONTEND_CACHE_TO+=(--cache-to "type=registry,ref=${FRONTEND_CACHE_IMAGE},mode=${FRONTEND_CACHE_MODE}")
    fi
  else
    mkdir -p "${LOCAL_CACHE_DIR}"
    if [[ -f "${LOCAL_CACHE_DIR}/backend/index.json" ]]; then
      BACKEND_CACHE_FROM+=(--cache-from "type=local,src=${LOCAL_CACHE_DIR}/backend")
    fi
    if [[ -f "${LOCAL_CACHE_DIR}/frontend/index.json" ]]; then
      FRONTEND_CACHE_FROM+=(--cache-from "type=local,src=${LOCAL_CACHE_DIR}/frontend")
    fi
    if [[ "${WRITE_CACHE}" == "true" || "${WRITE_CACHE}" == "1" ]]; then
      BACKEND_CACHE_TO+=(--cache-to "type=local,dest=${LOCAL_CACHE_DIR}/backend,mode=${BACKEND_CACHE_MODE}")
      FRONTEND_CACHE_TO+=(--cache-to "type=local,dest=${LOCAL_CACHE_DIR}/frontend,mode=${FRONTEND_CACHE_MODE}")
    fi
  fi
fi

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
echo "Build cache enabled: ${CACHE}"
echo "Build cache export enabled: ${WRITE_CACHE}"
echo "Inline image cache enabled: ${INLINE_CACHE}"
echo "Inline image cache fallback tag: ${INLINE_CACHE_FROM_TAG}"
echo "Registry cache enabled: ${REGISTRY_CACHE}"
echo "Backend cache export mode: ${BACKEND_CACHE_MODE}"
echo "Frontend cache export mode: ${FRONTEND_CACHE_MODE}"

docker buildx inspect >/dev/null 2>&1 || docker buildx create --use --name xagent-builder

echo "Building backend image..."
docker buildx build \
  --platform "${PLATFORMS}" \
  -f "${REPO_ROOT}/docker/Dockerfile.backend" \
  -t "${BACKEND_IMAGE}:${TAG}" \
  "${BACKEND_BUILD_ARGS[@]}" \
  "${BACKEND_CACHE_FROM[@]}" \
  "${BACKEND_CACHE_TO[@]}" \
  "${BUILD_OUTPUT_FLAG}" \
  "${REPO_ROOT}"

echo "Building frontend image..."
docker buildx build \
  --platform "${PLATFORMS}" \
  -f "${REPO_ROOT}/docker/Dockerfile.frontend" \
  -t "${FRONTEND_IMAGE}:${TAG}" \
  "${FRONTEND_CACHE_FROM[@]}" \
  "${FRONTEND_CACHE_TO[@]}" \
  "${BUILD_OUTPUT_FLAG}" \
  "${REPO_ROOT}/frontend"

if [[ "${BUILD_OUTPUT_FLAG}" == "--push" ]]; then
  echo "Images published successfully:"
else
  echo "Images built successfully:"
fi
echo "  - ${BACKEND_IMAGE}:${TAG}"
echo "  - ${FRONTEND_IMAGE}:${TAG}"
