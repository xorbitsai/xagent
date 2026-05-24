# Scripts

Operational helper scripts for local development, upgrades, and release checks.
Run commands from the repository root unless a script says otherwise.

## `backfill_uploaded_files.py`

Registers files already present under the configured uploads directory into the
`uploaded_files` table. Useful after upgrading from an older build that wrote
files to disk but did not create database records for them.

Common usage:

```bash
PYTHONPATH=src python scripts/backfill_uploaded_files.py --dry-run
PYTHONPATH=src python scripts/backfill_uploaded_files.py
PYTHONPATH=src python scripts/backfill_uploaded_files.py --user-id 1
```

## `check_alembic_heads.sh`

Verifies that Alembic currently has exactly one migration head. This is a quick
CI/local guard for accidental migration branching.

Common usage:

```bash
PYTHONPATH=src scripts/check_alembic_heads.sh
```

## `convert_trace_checkpoint_messages.py`

Converts historical checkpoint trace rows from legacy inline payloads to refs
backed by blob tables. It currently covers `snapshot.context.messages`,
`snapshot.pattern_state.tool_ledger`, and `snapshot.context.metadata`. This is
an internal storage migration only: API/runtime readers still decode refs back
into normal checkpoint objects.

The script is dry-run by default. Use `--execute` to write changes.

Common usage:

```bash
# Preview all configured database rows without writing changes.
PYTHONPATH=src python scripts/convert_trace_checkpoint_messages.py

# Convert all historical checkpoint rows.
PYTHONPATH=src python scripts/convert_trace_checkpoint_messages.py --execute

# Convert only one task.
PYTHONPATH=src python scripts/convert_trace_checkpoint_messages.py --execute --task-id 467

# Resume after a known trace_events id or process in smaller batches.
PYTHONPATH=src python scripts/convert_trace_checkpoint_messages.py --execute --start-id 100000 --batch-size 100
```

Notes:

- The configured database comes from `DATABASE_URL`, or the normal xagent
  default database path if `DATABASE_URL` is unset.
- Existing refs rows are skipped, so it is safe to rerun.
- New writes from the upgraded app already use refs and will be skipped.
- Row-level conversion failures are logged and counted, then the script
  continues with later rows. A run with any row errors exits non-zero.
- For SQLite, the database file may not physically shrink until you run
  `VACUUM` after conversion.

## `export_openapi.py`

Exports the FastAPI OpenAPI schema to JSON.

Common usage:

```bash
PYTHONPATH=src python scripts/export_openapi.py -o openapi.json
```
