# Task attachment retention rollout

This is the foundation slice of [#1086](https://github.com/xorbitsai/xagent/issues/1086).
It records prospective detachment and makes explicit reuse atomic. **It does not
start a detached-file sweep or promise that retained bytes have been erased.**

## Lifecycle contract

- Manual deletion and scheduled conversation expiry use `task_deleted`.
- Failed-create compensation uses `task_create_failed`, only for files actually
  bound to the compensated task. Uncommitted bindings are rolled back first.
- Deletion clears `task_id` and records both markers in the same transaction.
  Trace-only expiry and dry runs leave attachments unchanged.
- An authorized owner holding the stable `file_id` can explicitly reattach a
  retained file. Binding clears both markers atomically. Another task, owner,
  or a cleanup claim winning the race causes the existing bind failure contract
  to apply; task creation returns HTTP 409 and rolls back.
- Detached files are hidden from both default and uploads-only lists, public
  routes addressed by registered UUIDs, and implicit filename/configuration-based reuse. Explicit owner
  reuse still applies the normal authorization and materialization checks.
- A raw task DELETE benefits from `ON DELETE SET NULL`, but does not record
  provenance. Supported task deletion must call `purge_task_rows` with a reason.
  Historical unmarked rows remain ambiguous and are preserved.

## Deploy and roll back

1. Back up the database and run migration
   `20260929_task_attachment_detachment` online before deploying new code.
   Offline SQL generation fails explicitly. PostgreSQL adds the replacement
   foreign key as `NOT VALID`, commits the DDL, then validates it. If interrupted,
   rerun the migration; existing columns and an unvalidated FK are supported.
   The migration stops before changing the schema when an upload references a
   missing task and reports the affected upload and task IDs. Restore missing
   task rows from an authoritative backup, or correct each `task_id` only when
   its intended task is established, then rerun the migration. Preserve the
   upload IDs and rows while reconciling them. Do not clear `task_id` or invent
   detachment markers: that would turn ambiguous history into apparent drafts
   or claim a deletion provenance the database does not establish.
   SQLite rebuilds the upload table and requires a maintenance window.
   Historical migration-only schemas without the metadata-owned `tasks` table
   keep their absent task FK. This preserves partial-schema migration behavior;
   `create_all` does not repair an existing table's missing FK. When completing
   such a schema manually, create the parent tables and reapply this revision
   before enabling writers. Normal empty application installs stamp head and
   create the complete metadata schema instead.
2. Deploy all task/file writers and retirement workers together, or pause
   deletion during the rolling deployment. Older writers do not record
   provenance, so a mixed deployment cannot guarantee complete classification.
3. Keep detached collection disabled until the next slice lands. The existing
   task-less share-upload collector excludes marked rows, so its 48-hour
   creation-time policy cannot consume a detached file's future retention window.
4. Prefer rolling back application code while retaining the additive schema.
   Pause deletion and binding first. A schema downgrade refuses to discard any
   non-null detachment marker; export and reconcile those obligations before
   deliberately clearing them. Downgrade restores the task FK's `CASCADE` action.

## Remaining delivery and operational limits

The intended detached retention window is seven days from `detached_at`, with a
deployment override. That clock and its sweep cadence, batch budget, eligible
index, KB/pending-ingest protections, and retry completion accounting belong to
the next PR. No such environment setting or detached cleanup is active here.

The third PR covers managed legacy/local-only disposition and an inventory of
ambiguous historical rows. Do not backfill a deletion reason solely because
`task_id IS NULL`: legitimate unsent drafts share that shape.

Known existing cleanup gaps remain tracked in #1086: durable/local/preview
failure recovery and historic orphan disposition. Account/team erasure,
including uploads removed by the user cascade, is tracked separately in
[SaaS #1576](https://github.com/xorbitsai/xagent-saas/issues/1576).
This slice does not establish that account closure or all historical-file
cleanup is complete.
