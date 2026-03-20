#!/usr/bin/env bash
set -euo pipefail

cd /opt/xagent

# Start Xvfb for headless browser/screenshots (avoid xvfb-run wait-for-USR1 hang)
if [[ -z "${DISPLAY:-}" ]]; then
  export DISPLAY=":99"
  # Start Xvfb in background (no xauth needed for our use)
  Xvfb "$DISPLAY" -screen 0 1920x1080x24 -nolisten tcp >/tmp/Xvfb.log 2>&1 &
  # Give Xvfb a moment to become ready
  sleep 0.2
fi

# Start FastAPI server directly
# Database initialization and migrations will be handled by init_db() in app.py
BACKEND_PORT="${BACKEND_PORT:-8000}"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
echo "Starting Xagent web server on ${BACKEND_HOST}:${BACKEND_PORT}..."
exec xagent-web --host "$BACKEND_HOST" --port "$BACKEND_PORT"
