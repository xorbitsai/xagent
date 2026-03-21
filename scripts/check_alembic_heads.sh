#!/bin/bash
# Check Alembic has exactly one head

set -e

HEADS=$(alembic heads | grep -c " (head)" || true)

if [ "$HEADS" -eq 1 ]; then
    echo "Alembic: Single head confirmed"
    exit 0
else
    echo "Alembic: Expected 1 head, found $HEADS"
    echo "Run 'alembic heads' to see all heads"
    exit 1
fi
