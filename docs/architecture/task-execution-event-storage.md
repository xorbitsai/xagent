# Execution-event storage: migration stage 3.1

This stage adds storage primitives only. No Web, channel, runner, checkpoint,
or historical-reader path writes or reads these events in production yet.

## Version boundary

`tasks.conversation_storage_version` defaults to `1` (legacy), both in the
ORM and in SQL. A database CHECK pins it to `1`: the event-backed runtime is
not available in this release. Activating another version requires a later
migration **and** the complete writer/reader routing from stages 3.2–3.3.
This is a conversation-storage version, not an agent runtime version.

Existing rows and inserts from older applications retain legacy behavior.
The migration adds columns with inline CHECKs, avoiding a SQLite tasks-table
rebuild and its inbound foreign-key hazards. PostgreSQL still takes an ALTER
TABLE lock; rollout needs a short lock window, not a zero-lock assumption.
The new indexes apply only to the new, initially empty event table.

## Event envelope

`task_execution_events` records one fact with:

- A generated stable `event_id` and a task-local `sequence`.
- An explicit nonempty `scope_id` (`root` or a stable child execution scope).
- Optional `run_id`, `turn_id`, `assistant_message_id`, and `tool_attempt_id`.
  An assistant tool-call batch and each tool attempt have distinct identities.
- A nonempty producer-supplied `idempotency_key`, unique within task and scope.
  Keys for per-run facts must include the run/attempt identity; user acceptance
  keys can instead identify the durable input command across delivery retries.
- A `kind`, positive `payload_version`, JSON payload, and occurrence timestamp.

The envelope can carry message/attachment data, full tool results, input
application facts, recovery state and references. Event-specific schemas and
runtime identity producers belong to stage 3.2; this store does not invent
missing run state or infer event kinds from existing Trace rows. No history
backfill is performed here.

Payloads use the existing JSON sanitizer before persistence, including the
PostgreSQL JSONB code-point policy. They are not clipped or summarized.
This normalization is not an authorization or client-disclosure policy.

## Transaction contract

`append_task_execution_event_no_commit` stages a fact in the caller's Session;
the caller commits or rolls back alongside its business state. It first
locks the task with an UPDATE, then checks idempotency and allocates the next
`conversation_event_sequence` in the same transaction. Task `updated_at` is
preserved. No writer can commit a later task sequence past a pending append;
a rollback rolls back the sequence allocation as well as the event.

A retry with the same key and normalized fact returns the original event ID,
sequence and first occurrence timestamp. Reusing a key for different content
or correlation raises `ExecutionEventConflict`. Callers must finish their
transaction, including on errors; this helper never commits or rolls it back.

`load_task_execution_events` requires task and scope, and reads by sequence
with a page size of 1–100 (default 100). Out-of-range sizes raise `ValueError`
before querying the database. It uses the caller's transaction snapshot; another
connection cannot see uncommitted events, while read-your-writes in the same
Session is intentional. Authorization remains the future calling service's
responsibility. Neither helper is an externally exposed endpoint.

Only the append interface is provided; there is no edit or pruning API.
Task deletion cascades at the database foreign key. Stage 3.2 must integrate
the full application lifecycle before production events are enabled.

## Validation

The migration tests cover SQLite and PostgreSQL upgrade/downgrade, old SQL
inserts, retained chat rows and foreign keys, create_all parity, and offline
SQL generation. Store tests cover uncommitted visibility, rollback, concurrent
commit ordering and replay, conflicting identities, scope pagination, JSON
payload fidelity and database constraints.

Code rollback may retain the additive schema. Alembic downgrade deletes the
new event table and counter: it is suitable for this unwired stage, not a way
to revert an event-backed task after later stages have been activated.
