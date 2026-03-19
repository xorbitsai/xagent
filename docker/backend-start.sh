#!/usr/bin/env sh
set -eu

# Ensure venv executables are found
export PATH="/opt/venv/bin:$PATH"

# Start backend (xvfb-run is handled by image ENTRYPOINT)
exec /opt/xagent/deploy/entrypoint.sh

