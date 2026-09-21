#!/usr/bin/env bash
set -euo pipefail

cd "${XAGENT_INSTALL_ROOT:-/opt/xagent}"

# Compose-backed shared execution needs one stable private key across the web,
# Agent workers, and background services.  An explicit environment value wins;
# otherwise the image points this at a shared, persistent secret volume.
if [[ -z "${ENCRYPTION_KEY:-}" && -n "${XAGENT_ENCRYPTION_KEY_FILE:-}" ]]; then
  ENCRYPTION_KEY="$(python -m xagent.web.encryption_key_bootstrap "${XAGENT_ENCRYPTION_KEY_FILE}")"
  export ENCRYPTION_KEY
fi

# Docker Compose uses this image for Celery worker and scheduler services too;
# execute those overrides without an unnecessary Xvfb. A standalone Agent
# worker can use browser automation, so its override continues through display
# initialization below.
if (( $# > 0 )) && [[ "${XAGENT_TASK_EXECUTION_ROLE:-combined}" != "worker" ]]; then
  exec "$@"
fi

# Start Xvfb for headless browser/screenshots (avoid xvfb-run wait-for-USR1 hang)
if [[ -z "${DISPLAY:-}" ]]; then
  export DISPLAY=":99"
  # Start Xvfb in background (no xauth needed for our use)
  Xvfb "$DISPLAY" -screen 0 1920x1080x24 -nolisten tcp >/tmp/Xvfb.log 2>&1 &
  # Give Xvfb a moment to become ready
  sleep 0.2
fi

if (( $# > 0 )); then
  exec "$@"
fi

# Start FastAPI server directly
# Database initialization and migrations will be handled by init_db() in app.py
BACKEND_PORT="${BACKEND_PORT:-8000}"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
echo "Starting Xagent web server on ${BACKEND_HOST}:${BACKEND_PORT}..."
exec xagent-web --host "$BACKEND_HOST" --port "$BACKEND_PORT"
