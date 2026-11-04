#!/usr/bin/env bash
# Check the Alembic revision graph: exactly one head, no duplicate revision IDs,
# and every down_revision resolvable.
#
# Alembic does not enforce any of this on its own: a branched graph is legal and
# `get_heads()` returns every head without complaint, while a duplicate revision
# ID only produces a warning. So each invariant is asserted explicitly here.
#
# Regression coverage: tests/migrations/test_check_alembic_heads.py
# Set ALEMBIC_CHECK_PYTHON to run alembic under a specific interpreter (the tests
# use it to pin their own). It names an interpreter rather than a whole command
# line on purpose: the value goes straight into a quoted array element, so it
# needs no word splitting and stays correct when the path contains spaces.

set -uo pipefail

if [ -n "${ALEMBIC_CHECK_PYTHON:-}" ]; then
    ALEMBIC_CMD=("$ALEMBIC_CHECK_PYTHON" -m alembic)
elif command -v uv >/dev/null 2>&1; then
    ALEMBIC_CMD=(uv run alembic)
else
    ALEMBIC_CMD=(python -m alembic)
fi

stderr_file=$(mktemp)
trap 'rm -f "$stderr_file"' EXIT

if ! heads_output=$("${ALEMBIC_CMD[@]}" heads 2>"$stderr_file"); then
    echo "Alembic: could not read the revision graph." >&2
    echo "A down_revision most likely points at a revision that is not on disk." >&2
    sed 's/^/  /' "$stderr_file" >&2
    exit 1
fi

if grep -q "present more than once" "$stderr_file"; then
    echo "Alembic: duplicate revision ID." >&2
    grep "present more than once" "$stderr_file" | sed 's/^/  /' >&2
    echo "Two migration files declare the same revision; give one a new ID." >&2
    exit 1
fi

head_count=$(printf '%s\n' "$heads_output" | grep -c " (head)")

if [ "$head_count" -eq 1 ]; then
    echo "Alembic: single head confirmed"
    exit 0
fi

echo "Alembic: expected exactly 1 head, found ${head_count}." >&2
printf '%s\n' "$heads_output" | sed 's/^/  /' >&2
echo "Merge the branches with '${ALEMBIC_CMD[*]} merge heads'." >&2
exit 1
