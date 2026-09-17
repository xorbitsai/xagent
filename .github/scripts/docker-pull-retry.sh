#!/usr/bin/env bash
# Pull one or more images, retrying transient registry failures.
#
# CI jobs that drive a real Docker daemon must pre-pull every image their tests
# need. Otherwise the pull happens inside a test fixture, and a Docker Hub
# timeout or an anonymous pull rate limit surfaces as a pile of confusing
# APIError test failures instead of a clearly-named setup failure.
#
# Usage: docker-pull-retry.sh IMAGE [IMAGE...]

set -euo pipefail

ATTEMPTS="${PULL_ATTEMPTS:-3}"
BACKOFF_SECONDS="${PULL_BACKOFF_SECONDS:-15}"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 IMAGE [IMAGE...]" >&2
  exit 2
fi

for image in "$@"; do
  pulled=false
  for ((i = 1; i <= ATTEMPTS; i++)); do
    if docker pull "$image"; then
      pulled=true
      break
    fi
    if [ "$i" -lt "$ATTEMPTS" ]; then
      echo "Pull attempt $i for $image failed; retrying in ${BACKOFF_SECONDS}s..."
      sleep "$BACKOFF_SECONDS"
    fi
  done
  if [ "$pulled" != true ]; then
    echo "Failed to pull $image after $ATTEMPTS attempts" >&2
    exit 1
  fi
done
