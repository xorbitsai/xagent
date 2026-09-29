# Deployment changes

## LanceDB memory compatibility

The LanceDB dependency is bounded: `lancedb>=0.32.0,<0.38` in `pyproject.toml`.
The bound is deliberate. Persistent memory reaches storage only through the
admission primitives, and those depend on LanceDB table, metadata and commit
semantics that no minor release is contracted to preserve, so an unbounded
range would let an untested minor be installed under a runtime that fences
memory off when it disagrees.

CI exercises three representatives of that range:

| Version | Why |
| --- | --- |
| `0.32.0` | The declared minimum. |
| `0.33.0` | The version pinned in `uv.lock`, which is what a default install resolves to. |
| `0.37.1` | The newest supported minor. |

Those three are tested, not every intervening release. Raising the upper bound
means testing the new minor against the admission matrix first and moving the
`pyproject.toml` bound, the lock and this table together.

## 2026-08-11 — New public-task File Operation isolation

### Deployment impact

New widget and shared-link tasks use a server-owned policy marker to restrict File Operation access to the task owner and exact task. Existing private tasks and historical public tasks remain unmarked and keep their previous behavior.

A2A-, SDK-, and trigger-created tasks also remain unmarked and retain legacy owner-wide File Operation behavior. Protect runtime API keys and externally callable trigger credentials accordingly.

A mixed-version deployment is unsafe after new public task creation starts. An older worker does not enforce the marker. Gate widget and shared-link task creation until all API and task-execution workers run the new version.

### Scope limitation

This rollout isolates only the File Operation tool family. Other tools that read paths from the shared task workspace, including image, audio, PowerPoint, video, and SSH upload tools, continue to use the existing workspace resolver and owner-wide external directory roots. MCP roots, sandbox access, shell and Python execution, knowledge-base operations, and preview/download authorization are also unchanged.

Do not treat this rollout as complete public-file isolation. Restrict those tools separately when a public deployment requires a task-wide boundary across every file-capable tool.

Isolation is task-level, not agent-level: delegated agents within the same task inherit the parent task identity and share that task's File Operation file set.

### Prerequisites and configuration

This change has no database migration, backfill, new environment variable, dependency, or infrastructure requirement.

Before deployment, inspect existing `Task.agent_config` values for `__xagent_file_operation_access_version`. Use the query for the configured database:

```sql
-- PostgreSQL
SELECT id
FROM tasks
WHERE agent_config -> '__xagent_file_operation_access_version' IS NOT NULL
LIMIT 1;
```

```sql
-- SQLite with the JSON1 extension
SELECT id
FROM tasks
WHERE json_type(agent_config, '$.__xagent_file_operation_access_version') IS NOT NULL
LIMIT 1;
```

Both queries must return no rows. If either query returns a task, stop the deployment. Select an unused internal key before you continue.

### Deployment and migration steps

1. Gate new widget and shared-link task creation.
2. Deploy the same application version to all API and task-execution workers.
3. Make sure that no old worker can receive a newly created public task.
4. Re-enable widget and shared-link task creation.

Do not backfill historical tasks. Marker absence is the compatibility boundary for this rollout.

For a future marker version, first deploy readers that accept both the current and
new versions while writers still emit the current version. Only change writers
after every API and task-execution worker accepts the new version. Never replace
the current accepted version in one step because persisted tasks must remain
readable throughout the rollout.

### Verification and monitoring

Create one widget task and one shared-link task after the rollout. Make sure that each task can use its own uploaded file.

Make sure that each task cannot use a same-owner file from another task by file ID. Repeat the check with a raw path.

Monitor task execution errors for File Operation policy failures. A failure on a newly created public task can indicate a malformed marker or missing task/owner authority.

### Per-task emergency remediation

If one task must recover before its policy inconsistency can be repaired, quiesce that task, remove only `__xagent_file_operation_access_version` from its `agent_config`, and rebuild or restart its execution. This opts that task out of exact-task isolation and restores legacy owner-wide File Operation access. Treat the change as an audited security exception because it reintroduces same-owner cross-task access for that task.

### Rollback

Gate new widget and shared-link task creation before rolling back any worker. Roll back all API and task-execution workers together. Do not re-enable public task creation while versions are mixed.

Marked tasks do not remain isolated when executed by an older worker. Keep public execution gated during rollback, or complete the forward rollout before those tasks resume.

## 2026-08-18 — Owner-aware builtin OAuth storage

### Deployment impact

The `user_oauth` table gets a nullable `resource_owner_key` column, and existing rows keep a null value. This foundation explicitly scopes non-Gmail OAuth consumers to that ordinary namespace. Gmail behavior remains unchanged here; the immediately following Gmail lifecycle release installs its complete owner boundary before any release can create actor-owned credentials.

Two partial unique indexes replace `uq_user_provider_account`. One index protects ordinary rows. The other index separates actor-owned namespaces. Standard SQL null semantics permit duplicate identities when `provider_user_id` is null. This behavior applies to ordinary and actor-owned rows.

PostgreSQL is the only supported production database, including self-hosted production installations. SQLite is supported only for local development and CI. Startup and migration fail before schema creation on other dialects.

The `users` table is an application-metadata table and must exist before this revision runs. Do not use bare Alembic to initialize an empty application database. Normal application startup stamps the empty database before it creates metadata-owned tables.

On a repository-produced legacy schema with no `user_id` FK, this revision installs `user_oauth.user_id -> users.id ON DELETE CASCADE`. A valid existing cascade can use any constraint name. If a non-cascade `user_id` FK exists, do not run or retry this migration. Have a database operator restore exactly one cascade FK before retrying. If this drift is not repaired, a same-named FK can block the repair or a differently-named FK can remain beside the cascade.

The migration fails when `users` is absent. It also fails when an owner-aware schema does not have the required cascade.

If bare Alembic was run against a genuinely empty database, the command can stop at `20260818_seed_stripe_mcp_app` after earlier revisions created a partial schema. Do not create `users` manually and retry. For a disposable database, delete and recreate it, then initialize it through normal application startup. For a non-disposable database, keep workers stopped and restore the pre-attempt backup, or have a database operator inspect and restore one coherent schema before startup.

On PostgreSQL the migration creates the replacement indexes transactionally before removing the old unique constraint. A failed statement rolls back the complete schema transition. If a same-name relation causes the failure, an operator must inspect and remove or rename that relation before retrying `alembic upgrade head`. `ADD COLUMN` and the non-concurrent index builds hold table locks until the transaction commits and can block both reads and writes to `user_oauth`. Pause every OAuth operation that accesses this table for the migration window, and monitor lock wait time instead of assuming the pause will be short.

On local SQLite, the migration rejects globally colliding owner-index names before it rebuilds the table. Stop local processes that use the database before this rebuild. If the local data must be preserved, create a verified backup. SQLite DDL can commit independently of Alembic's outer transaction.

If the SQLite migration process exits after the rebuild starts, keep local processes stopped. Retry `alembic upgrade head` once with the same release.

The migration completes only an unambiguous interrupted index-installation state. The `resource_owner_key` column must have its expected nullable `VARCHAR(512)` definition. The `uq_user_provider_account` constraint must be absent. Zero, one, or both owner-aware indexes can exist. Each existing owner-aware index must have the expected definition.

The migration creates only the missing owner-aware indexes. If both valid indexes exist, the migration makes no schema change before Alembic records the revision. Do not resume local processes until both indexes pass the verification below.

If that retry reports an invalid schema, do not continue automatically. For a disposable local database, delete and recreate it through normal application startup. For a non-disposable local database, restore the verified backup or repair one coherent schema manually. Never use a table that lacks the old constraint and the two verified owner-aware indexes.

If the retry reports a leftover `_alembic_tmp_user_oauth` table, do not remove or rename either table automatically. For a disposable local database, delete and recreate it. Otherwise, compare both tables and restore one coherent `user_oauth` table before retrying.

The normal application-startup migration path disables SQLite foreign-key enforcement around batch rebuilds. It rejects new foreign-key violations before commit. The standalone `alembic upgrade head` path does not provide that guard. If you use the standalone command, record `PRAGMA foreign_key_check;` and `SELECT count(*) FROM gmail_watch_states;` before and after migration. Do not resume local processes if the foreign-key result gains a row or the watch-state count changes. A valid `ON DELETE CASCADE` can remove child rows without leaving a foreign-key violation.

If the migration reports `UserOAuth schema is partially owner-aware`, do not resume local processes. For a disposable local database, delete and recreate it. Otherwise, restore the last complete backup or repair one coherent schema before retrying `alembic upgrade head`.

If SQLite reports that an owner-aware schema name exists before migration, query `sqlite_master` for that name. Identify its relation type, owning table, and definition. After you create a backup, remove or rename only the unrelated colliding table, index, or view. Then retry `alembic upgrade head`. If either database reports `owner-aware UserOAuth schema has incorrect indexes`, do not use the database. Compare the index columns, uniqueness flags, and predicates with the definitions below. Repair or remove incorrect indexes before you retry the migration.

### Prerequisites and configuration

This change has no new environment variable or dependency. Keep every future actor-OAuth caller disabled; this release does not expose a production path that creates actor-owned rows.

### Deployment and migration steps

Use the PostgreSQL procedure for production. Use the SQLite procedure only for local development. CI uses automated migration coverage.

#### SQLite local development

1. Stop local processes that use the database.
2. If the local data must be preserved, create a verified backup.
3. Update the local application files.
4. Record `PRAGMA foreign_key_check;` and `SELECT count(*) FROM gmail_watch_states;`.
5. Run `alembic upgrade head` one time.
6. Run both queries again. Make sure that the foreign-key result has no new row. Make sure that the watch-state count is unchanged.
7. Verify the schema.
8. Resume the local processes.

#### PostgreSQL production

1. Pause OAuth reads and writes that access `user_oauth`, and make sure no long transaction holds a lock on the table.
2. Run `alembic upgrade head` one time. Already-running old workers can continue non-OAuth work while the transactional DDL runs, but an old worker that starts or restarts after the schema revision advances will fail startup because it does not recognize the new revision. Prevent old-version restarts and autoscaling during this window, or ensure every replacement starts from the owner-aware image.
3. Resume ordinary OAuth writes after the migration commits.
4. Roll every API and task worker to the owner-aware version.
5. Verify the schema and make sure no old worker remains before a later release enables actor-owned rows.

Do not backfill `resource_owner_key`. A null owner identifies an ordinary credential.

### Verification and monitoring

Run this query after the migration:

```sql
SELECT count(*)
FROM user_oauth
WHERE resource_owner_key IS NOT NULL;
```

The result must be zero.

On PostgreSQL, verify both partial unique index definitions:

```sql
SELECT
  c.relname,
  i.indisunique,
  pg_get_expr(i.indpred, i.indrelid) AS predicate,
  pg_get_indexdef(i.indexrelid) AS definition
FROM pg_index i
JOIN pg_class c ON c.oid = i.indexrelid
JOIN pg_class t ON t.oid = i.indrelid
JOIN pg_namespace n ON n.oid = t.relnamespace
WHERE n.nspname = current_schema()
  AND t.relname = 'user_oauth'
  AND c.relname IN (
    'uq_user_oauth_ordinary_account',
    'uq_user_oauth_actor_account'
  );
```

The query must return both rows with `indisunique = true`. The ordinary row must index `(user_id, provider, provider_user_id)` with `resource_owner_key IS NULL`; the actor row must index `(user_id, resource_owner_key, provider, provider_user_id)` with `resource_owner_key IS NOT NULL`.

For local SQLite, run `PRAGMA index_list('user_oauth');` and `PRAGMA index_info('<index-name>');`. Inspect `sqlite_master.sql`. The ordinary index must use `WHERE resource_owner_key IS NULL`. The actor index must use `WHERE resource_owner_key IS NOT NULL`.

Run `PRAGMA foreign_key_list('user_oauth');`. Require exactly one FK on `user_id`. This FK must target `users.id` with a `CASCADE` delete action. On PostgreSQL, inspect the `user_oauth` constraints. Require the same single cascade before you enable actor-owned rows.

Verify existing cloud-storage and builtin OAuth connections. Confirm that seeded non-null-owner test rows do not appear in ordinary catalog or token paths.

### Rollback

Because this release cannot create actor-owned rows, the downgrade remains available after ordinary rollout. The downgrade keeps the `user_id -> users.id ON DELETE CASCADE` FK. The previous application model also requires this cascade.

1. For PostgreSQL, stop all production workers. For SQLite, stop local processes that use the database.
2. If local SQLite data must be preserved, create a current database backup.
3. For a non-disposable SQLite database, run `PRAGMA integrity_check;` against the backup. Record `SELECT count(*) FROM gmail_watch_states;`. The integrity result must be `ok`.
4. Run `alembic downgrade 20260818_seed_stripe_mcp_app`.
5. Run `alembic current`. The command must report only `20260818_seed_stripe_mcp_app`. The Stripe catalog seed remains installed.
6. For SQLite, run `PRAGMA integrity_check;` and `PRAGMA foreign_key_check;`. Require `ok` and no foreign-key violations.
7. For a non-disposable SQLite database, run `SELECT count(*) FROM gmail_watch_states;`. Require the count that step 3 recorded.
8. For SQLite, inspect `PRAGMA table_info('user_oauth');`. The result must not contain `resource_owner_key`.
9. For SQLite, inspect `PRAGMA index_list('user_oauth');` and each `PRAGMA index_info('<index-name>');` result. One unique index must cover `(user_id, provider, provider_user_id)`.
10. For PostgreSQL, deploy the old version. For SQLite, return to the previous local application version.

SQLite can commit each schema operation separately during a batch-table rebuild. If a downgrade fails, do not retry against the changed database. For a disposable local database, delete and recreate it through normal application startup. Otherwise, restore the verified backup. Make sure that `alembic current` reports the owner-aware revision before you retry the downgrade.

The migration refuses the downgrade if a non-null owner row exists. If a caller created such a row, disable that caller. Revoke and remove the credential with an approved procedure. Then retry the downgrade.

## 2026-08-24 — Gmail ordinary-owner fence

### Scope

This release adds no schema, dependency, environment variable, or cleanup state. It restricts Gmail watch and trigger code to ordinary OAuth rows.

Mailbox release now calls Gmail `users.stop` before Pub/Sub cleanup. Each release can add one Gmail API request.

Actor-owned Gmail credentials remain available to builtin MCP tools. Gmail provisioning, renewal, callback, trigger, and release paths reject these rows.

### Prerequisites and configuration

Keep every actor-owned credential writer disabled during this rollout. The owner-aware OAuth migration above must already be current.

### Gmail trigger binding contract

`oauth_account_id` has three states:

1. An absent key is a persisted legacy binding. Its mailbox (`resource_id`) must be non-empty.
2. A positive integer or ASCII decimal string is an explicit binding. It must match a same-user ordinary Gmail account.
3. Any other present value is invalid. New API requests reject it. Persisted invalid bindings fail closed, are marked failed, and prevent mailbox teardown until repaired.

Provisioning resolves a legacy binding only when its mailbox matches exactly one same-user ordinary Gmail account. It does not add an account ID to the stored legacy configuration. A missing or ambiguous match fails closed.

New or edited Gmail trigger configurations must use an explicit account ID. You can re-enable a persisted legacy trigger without editing its configuration. To repair an unavailable legacy binding, replace it with the matching ordinary Gmail account ID.

Do not remove an invalid key to repair a trigger unless it is a confirmed legacy mailbox binding. Do not use `0`, `null`, booleans, floats, or non-decimal strings as Gmail account IDs.

### Deployment and migration steps

1. Deploy this release to every Gmail API, callback, trigger, dispatcher, and worker process.
2. Make sure that no older process remains.
3. Run the ownership query below.
4. If the result is not zero, keep actor credential writers disabled. The fence does not clean invalid watches. Track the approved cleanup path in [issue #1652](https://github.com/xorbitsai/xagent/issues/1652).
5. Enable actor credential writers only after the result is zero.

### Verification and monitoring

Run this query before actor credential writers become active:

```sql
SELECT count(*)
FROM gmail_watch_states AS watch
LEFT JOIN user_oauth AS account ON account.id = watch.oauth_account_id
WHERE account.id IS NULL
   OR watch.user_id <> account.user_id
   OR account.provider <> 'gmail'
   OR account.resource_owner_key IS NOT NULL;
```

The result must be zero. Existing ordinary Gmail watch and trigger tests must also pass before deployment.

A callback with only invalid or actor-owned trigger bindings is acknowledged as unknown and does not advance the Gmail history cursor. Restore a valid ordinary trigger binding before callback processing can continue.

If a trigger reports `Gmail trigger has an invalid OAuth account binding`, replace its `oauth_account_id` with the matching ordinary Gmail account ID. Remove the key only for a confirmed legacy mailbox binding. Do not change actor-owned credentials or watch rows as part of this rollout.

### Rollback

Keep this fence during an actor-feature rollback. Disable actor credential writers before you roll back another actor layer.

The watch-ownership query above detects invalid watch bindings. Before you revert this fence, also run:

```sql
SELECT count(*)
FROM user_oauth
WHERE provider = 'gmail'
  AND resource_owner_key IS NOT NULL;
```

Both query results must be zero only when you revert this fence. The second query proves that no actor-owned Gmail credential remains. This includes credentials without watch state. Actor-owned credentials can exist while the fence remains. A normal actor-feature rollback does not revert this release.

## 2026-08-26 — PostgreSQL 17 default for the bundled Compose database

### Deployment impact

The bundled `postgres` service in `docker-compose.yml` now defaults to `postgres:17-bookworm`, aligning self-hosted deployments with the PostgreSQL 17 major that production runs and CI validates against. There is no Alembic revision, no schema change, and no application-code change. This is a data-directory migration only.

PostgreSQL never upgrades a data directory across major versions. A `postgres_data` volume initialized by PostgreSQL 16 does not open under a PostgreSQL 17 server: the server exits with `FATAL: database files are incompatible with server` without modifying the directory, restart-loops as `unhealthy`, and `backend`, `worker`, and `scheduler` never start because they wait for `service_healthy`. The v16 data stays intact, so pinning the previous tag restores service. Treat an unplanned upgrade of an existing v16 deployment as a full outage until that happens or the migration below completes.

Fresh installations are unaffected, because they initialize directly under v17. So are deployments whose `DATABASE_URL` points at an external or managed PostgreSQL, because the bundled service is not in their path.

### Prerequisites and configuration

`POSTGRES_IMAGE_TAG` is new in this release. It sets the tag of the bundled `postgres` image, defaults to `17-bookworm`, and is documented in `example.env`. Set it to `16-bookworm` to keep an existing v16 volume running until it is migrated.

Before migrating, confirm the deployment uses the bundled `postgres` service rather than an external database, and identify the Compose-prefixed volume name with `docker volume ls | grep postgres_data`. Reserve a maintenance window: the database is unavailable from the point writers stop until verification passes, and `nginx` and `frontend` keep serving errors to users for that whole window.

Never run `docker compose down -v` or remove the live `postgres_data` volume. Both destroy the database irreversibly and neither is an upgrade step. Removing the separate `_pg16_backup` copy that the runbook creates is a different operation and is expected.

The runbook is a general method, not a procedure tuned to any one deployment. Use the backup and verification process you already trust for this database, and rehearse the migration against a copy before running it on production.

### Deployment and migration steps

The executable commands are the [PostgreSQL major version upgrade (16 to 17)](../docker/README.md#postgresql-major-version-upgrade-16-to-17) runbook in `docker/README.md`; each numbered step below is one runbook step, in order. A deployment that uses a sandbox runtime overlay must keep its `-f` overlay arguments on every Compose command, or `backend`, `worker`, and `scheduler` come back without the overlay.

1. Pin `POSTGRES_IMAGE_TAG` to `16-bookworm` and confirm the deployment is healthy on v16.
2. Stop `backend`, `worker`, and `scheduler`, leaving `postgres` running.
3. Run your backup process, verify its result, and record the values you will compare after the restore.
4. Stop the stack without `-v` and copy the v16 volume to a separate volume, so rollback never depends on the dump alone.
5. Remove the v16 data directory from the live volume, guarded on the dump being complete and the copy being v16.
6. Unpin `POSTGRES_IMAGE_TAG` and start `postgres` on v17, which initializes a fresh cluster.
7. Restore the dump with `ON_ERROR_STOP=1`, then run the verification below.
8. Start the remaining services only after verification passes.

Keep the v16 volume copy while you are still deciding whether the upgrade held, then remove it with `docker volume rm`.

### Verification and monitoring

After the restore, `SHOW server_version` must report a 17.x version, and `SELECT version_num FROM alembic_version` must equal the value recorded in step 3. The second check is schema-only: a restore that stopped early does not change that value.

Those two establish that the schema arrived. What proves the data arrived depends on the deployment, so run the checks that matter for yours — row counts on the tables you care about, spot checks against recent records, an application smoke test.

Then run `vacuumdb --all --analyze-in-stages` inside the `postgres` container. `pg_dump` does not carry optimizer statistics across a restore, and a restored cluster plans queries badly until they exist.

Before starting the writers, confirm that `docker compose ps` reports `postgres` as `healthy` and that `docker compose logs postgres` shows a startup with no `FATAL` line.

### Rollback

Valid only until step 8 restarts the writers. Until that point a v17 server has never opened the volume copy taken in step 4, so it is still a complete picture of the database and rollback restores it directly instead of replaying the dump.

1. Stop the stack without `-v`.
2. Confirm the preserved v16 volume exists and reports `PG_VERSION` 16, before removing anything. Docker creates a named volume that does not exist, so an unchecked copy-back from a missing volume restores nothing over a directory it has already deleted.
3. Remove the data directory from the live volume and copy the preserved v16 volume back over it.
4. Pin `POSTGRES_IMAGE_TAG` to `16-bookworm`, start the stack, and confirm that `SHOW server_version` reports a 16.x version.

Roll back rather than repair in place when verification fails after a partial restore under v17. A cluster left half-populated by an interrupted restore is not a state to diagnose during an outage. If the v16 volume copy is unavailable, restore the verified dump from step 3 onto a v16 cluster initialized from `16-bookworm`.

After v17 accepts writes the volume copy is stale, and restoring it discards everything written since the cutover. Recovery from that point means taking a fresh v17 backup and reconciling the two, not a copy-back.

## 2026-09-15 — LanceDB FTS index rebuild for the jieba tokenizer

### Deployment impact

The knowledge-base full-text index now builds with the `jieba/default` tokenizer. The tokenizer is written into the index at build time and is never read back out, so an index built before this release keeps segmenting queries the old way until it is rebuilt. Chinese keyword search stays degraded on those tables, and nothing in the running system repairs them: `ensure_indexes` only creates an index that is missing, and the automatic rebuild inside `compact_tables` needs fresh ingestion plus a fragment or version threshold. A knowledge base that is only read from never reaches either, so a quiescent deployment stays on the old tokenizer indefinitely.

Existing deployments therefore need one manual run of the migration below. New installations do not: their first index build already uses the new tokenizer.

Embeddings tables are named `embeddings_<model>`, one per embedding model and shared by every tenant in that database, so the rebuild covers all tenants at once and the table count matches the number of models in use, not the number of collections.

### Prerequisites and configuration

The script talks to the same database as the application, so it must run with the same `LANCEDB_DIR` the backend uses. When `LANCEDB_DIR` is unset, both fall back to `~/.xagent/data/lancedb` inside the process's own home directory — which is the `xagent_data` volume in the container and a different directory on the host. Run it inside the `backend` container, or export `LANCEDB_DIR` explicitly, rather than relying on that default matching.

Rebuilding reads every indexed row of a table and writes a new index, so size the window by row count, not by table count. It does not rewrite the data files and does not compact: compaction remains the ingestion path's job.

The rebuild takes the same per-table lock that compaction uses, so a table being compacted by a concurrent ingestion is reported as unfinished rather than rebuilt. That lock is an advisory lock file inside a local `LANCEDB_DIR`: against a remote URI, a directory that is not local, or when the lock file cannot be created, both the rebuild and compaction proceed unlocked and can overlap. Run this while ingestion is idle, or re-run afterwards for the tables that were busy. Search keeps serving the old index until the new one commits; no maintenance window is required for readers.

### Deployment and migration steps

Deploy the release first, then run the migration once per database:

```bash
# 1. List what would be rebuilt, changing nothing
docker compose exec backend python -m xagent.migrations.lancedb.rebuild_fts_indexes --dry-run

# 2. Rebuild
docker compose exec backend python -m xagent.migrations.lancedb.rebuild_fts_indexes

# Optional: one table at a time
docker compose exec backend python -m xagent.migrations.lancedb.rebuild_fts_indexes --table embeddings_<model>
```

The dry run classifies every `embeddings_*` table as would-rebuild, would-skip (no full-text index on the `text` column, so there is nothing to replace), or unreadable, and the real run acts on that same classification.

Exit codes:

- `0` — every table that needed a rebuild was rebuilt.
- `1` — at least one table was not rebuilt: the rebuild raised, the table's lock was held elsewhere, or the table could not be read at all. The other tables still completed. An unreadable table is classified before any rebuild runs, so `--dry-run` also exits `1` when one is present.
- `2` — the run never started, for example an unreadable database or a `--table` value that is not an embeddings table.

The script is safe to re-run and rebuilding an already-rebuilt index is not an error, so recovery from exit code 1 is to fix the cause and run it again; with `--table` to retry only what failed.

### Verification and monitoring

Each table is logged with its index state before and after, so the run itself is the record: compare `version` and `indexed_rows` in the two lines, and confirm the closing summary reports the table under `Succeeded`. A rebuilt table normally reports `unindexed_rows: 0` afterwards, but treat the `Succeeded` list as the verdict: when the statistics cannot be read the entry carries `stats_error` instead of row counts, which says nothing about the rebuild.

The tokenizer cannot be read back out of a built index, so there is no stored value to assert against. The behavioural check is a Chinese multi-word keyword search against a collection on that table: before the rebuild it returns nothing or unrelated hits, after it returns the documents containing those words.

### Rollback

No rollback path and none needed: the previous index is replaced by one built from the same rows, and the schema, the data files and the row contents are untouched. Reverting the application code leaves the new index in place and searching it with the old tokenizer restores the previous behavior, which is the degraded one this rebuild fixes.

## 2026-09-23 — Retention purge job

The purge that acts on the retention predicate. It starts only when a retention period is configured (see `example.env`), and no deployment configures one yet, so this change is inert on its own.


### Enabling the purge

The job is gated on `XAGENT_CONVERSATION_RETENTION_DAYS` / `XAGENT_TRACE_RETENTION_DAYS` (see `example.env`), and it refuses to start on anything but PostgreSQL — the row lock its eligibility check depends on is compiled away on SQLite, so an assessment there is not a deletion licence. The refusal is logged once and the loop exits; it is not retried, because configuration cannot change under a running process.

**Every retention setting takes effect at process start, and only there.** Per-team overrides, when a deployment layer registers them, are the one exception (see the 2026-09-28 entry). `.env` is read once and nothing mutates the environment afterwards, so changing a period, the dry-run flag or the kill switch requires a restart. The kill switch exists so that stopping expiry does not mean editing the periods — not so that a running sweep can be halted from outside.

Before setting a period anywhere, work through the enablement gate on the retention tracking issue.

### Mixed-version rollouts and rollbacks

A binary older than the `tasks.last_activity_at` migration writes transcript messages without advancing the anchor. That happens during a rolling deploy, while old workers are still running after the backfill, and for as long as a deployment runs such a binary after rolling back past it. Either way the stored anchor can end up older than the task's newest message.

No drain or reconciliation step is needed for this. Before deleting anything, the purge measures each task from the later of the stored anchor and the task's newest message. It reads both under the task's row lock, and a message insert for that task waits on the lock, so a stale stored anchor cannot expire a conversation early, whichever version wrote the message.

The lock covers message inserts only because `task_chat_messages.task_id` has a foreign key to `tasks.id`. Every supported initialization path creates it, but the revision that introduced the table adds it only when `tasks` already existed. Confirm it on the target database before enabling a period; the query must return one row:

```sql
SELECT conname
FROM pg_constraint
WHERE contype = 'f'
  AND conrelid = 'task_chat_messages'::regclass
  AND confrelid = 'tasks'::regclass;
```

A stale anchor still affects `xagent retention preview` and the purge's candidate scan, which read the stored value only. Both can treat such a task as older than the purge will: the preview over-counts, never under-counts, and the scan hands the purge a candidate it then keeps (`skipped_not_due`) or expires only the trace of (`purged_traces`), where the stored anchor alone would have expired the whole conversation. Because the stored anchor is not repaired, the scan keeps selecting such a task on every sweep: it goes on reporting `skipped_not_due`, or `nothing_to_purge` once its trace is gone, until its newest message itself passes the conversation period.

Recommended first run: set the period together with `XAGENT_RETENTION_DRY_RUN=true`, restart, read the audit line, and only then clear the dry-run flag and restart again. A value the parser does not recognise resolves to a dry run rather than a deletion, but do not rely on that instead of checking the log line.

### Multiple web replicas

One purge loop starts per web process, so a deployment running several of them sweeps several times over. That is safe rather than merely tolerated: two purges that select the same task serialize on its row lock. On the conversation path the loser then finds no row and reports the task as not eligible. On the trace path the row survives, so the loser runs its deletes against a trace that is already gone; they remove nothing and it is counted as `nothing_to_purge` rather than as an expiry. Either way no accepted work is lost and no counter claims work that did not happen.

What it costs is duplicated scanning, which is why there is no advisory lock here. If that becomes visible on a large `tasks` table, set the retention variables on one replica only; the loop starts from configuration, so an unconfigured replica starts nothing.

### Verification and monitoring

`xagent retention preview --days N` reports what a period would expire without touching anything. Once the job runs, each *batch* logs one line beginning `retention purge` — a sweep that drains a backlog logs one per page, not one in total — counting `scanned`, `purged_conversations`, `purged_traces`, `skipped_busy`, `skipped_not_due`, `skipped_override_unresolved`, `skipped_active_interaction`, `nothing_to_purge`, `failed` and `cleanup_owed`.

`scanned` is how many candidates the batch selected, not how many proved expirable — the locked assessment can still refuse any of them. `skipped_busy` is a refusal because the task is not quiescent, which also includes a row that vanished between the scan and the lock (normal with more than one replica) and a task with no anchor at all. `skipped_not_due` is a quiescent task whose period has not elapsed under the lock: its newest message is more recent than its stored anchor, or its team's period changed after the scan. `skipped_override_unresolved` is a task whose team's period could not be resolved (see the 2026-09-28 entry). `nothing_to_purge` is a trace-expiry candidate whose trace was already gone by the time the lock was taken — either removed between the scan and the lock, or, for a task whose stored anchor lags its newest message, removed by an earlier sweep. `failed` is a task whose own purge raised: it is logged with its id and traceback, the sweep carries on, and the task is retried on the next pass.

Those field names deliberately differ from the ones sketched on the tracking issue (`eligible / deleted / skipped-busy / external-pending`): `deleted` is split because the two paths delete different things and an operator needs to know which ran, `skipped_active_interaction` names the one refusal that is permanent rather than transient, and `external-pending` became `cleanup_owed`: the external cleanup of the conversations the batch expired, which the job records for the cleanup retry driver rather than performs (see the 2026-09-25 entry below).

A persistently high `skipped_busy` means the batch is dominated by tasks that are not actually quiescent. A persistently high `skipped_active_interaction` means tasks are holding interaction rows that nothing is closing, which is worth investigating on its own — the purge will keep skipping them. Any non-zero `failed` deserves the log line that accompanies it: the purge no longer stops on such a task, so the only symptom is that counter.

Bulk deletion pressures autovacuum and can extend replication lag. Watch `n_dead_tup` on `tasks`, `trace_events` and the two `trace_*_blobs` tables while an initial backlog drains, raise `XAGENT_RETENTION_BATCH_PAUSE_SECONDS` if it is too aggressive, and plan a one-off `pg_repack` afterwards — deleting rows does not return storage to the filesystem.

### Rollback

`XAGENT_RETENTION_ENABLED=false` followed by a restart stops the job without changing the configured periods; unsetting the periods does the same. Neither restores deleted rows — recovery from an over-broad period is a database restore, which is what makes the dry run the step worth not skipping.

This change adds no migration and no index.

## 2026-09-20 — Persistent memory lifecycle enablement

### Deployment impact

Persistent memory now admits its LanceDB storage once, at worker startup, and
publishes a store only if that admission succeeds. Two behaviors change.

The runtime reads its embedding identity from the global memory embedding
authority alone. It no longer falls back to a user's default embedding model or
to whichever embedding model happens to be configured in the model hub. A
deployment with no authority configured runs persistent memory in an ephemeral
in-process store: memory works within a worker's lifetime and is not persisted.

Configuration no longer takes effect online. Changing the authority's provider,
model, endpoint, dimension, or instruct changes the vector space; running
workers stop serving memory and report that a restart is required. Rotating the
credential or changing the retry budget does not change the vector space, so
workers keep serving with their admitted credential until they restart.

### Prerequisites and configuration

The `global_memory_embedding_authority` table and its migration already ship.
No new environment variable, dependency, or infrastructure requirement is
introduced. `MEMORY_SIMILARITY_THRESHOLD` keeps its meaning.

LanceDB support is unchanged: the runtime only reaches storage through the
admission primitives, so the supported range is the one in `pyproject.toml`
described under "LanceDB memory compatibility" above.

### Deployment and migration steps

1. Quiesce memory writers first. Stop every API, task-execution and chat
   worker; do not leave a worker running against the memory LanceDB directory.
2. Configure the global memory embedding authority if persistent memory is
   wanted. The credential must be application- or organization-owned; a
   personal credential is rejected at rest. Never create or change the
   authority while any API or task worker is serving memory: a running agent
   keeps the adapter it was built with until its next hand-off, so a live
   change leaves a window in which the old adapter is still writing, and a
   same-width change silently mixes two vector spaces in one table. These are
   the same steps for a first rollout and for a later vector-space change.
3. Deploy the same version to every API and task-execution worker and start
   them together. Do not roll the fleet: a mixed fleet can have one worker
   writing under a vector space another has not admitted.
4. Confirm the state on each worker with `GET /api/memory/store-info`.

Admission migrates a table at most once. The first admission of a legacy table
(one without scope columns or without a current full-admission marker) scans
every row and atomically overwrites the table with the validated result. That
overwrite replaces whatever the table held when it commits, which is why the
first upgrade, a release that bumps the validator generation, and any
explicit repair must run with every writer quiesced.
Once a table carries the marker, admission is read-only: every later worker
start, respawn or retry only reads the schema and compares the stored vector
identity with the authority, and it never stages, rewrites or overwrites the
table. Publishing the admitted store is read-only as well: the worker opens
the certified table as is, never creates, reshapes or backfills it, and stays
fenced if the table no longer matches what admission certified. Ordinary
memory writes do not remove the marker, so workers can restart while their
siblings keep serving.

### Verification and monitoring

`GET /api/memory/store-info` reports `state`, `mode`, `supports_vector_search`
and a caller-safe `detail`. The states are:

| `state` | Meaning | Operator action |
| --- | --- | --- |
| `ready` | Admitted; memory is serving. `mode` is `vector`, or `text_only` when the stored vectors do not match the authority. | None when `mode` is `vector`. When `mode` is `text_only`, see "Serving in text_only mode" below: memory is writable but vector search is off, and restoring it is offline work. |
| `not_configured` | No authority configured; an ephemeral store is in use. | Configure the authority, then restart every worker. |
| `credential_unavailable` | The stored credential could not be decrypted or failed its verifier. | Re-set the authority, then restart every worker. |
| `retryable_unavailable` | The admission lock was held, or the backend failed transiently. | Stop every API and task-execution memory writer and keep them all stopped until the retried admission has finished, then restart the fleet together. Ordinary memory writes take neither the admission nor the maintenance lock, and a table that was never fully migrated is rewritten by admission, so a retry that runs beside a live writer on such a table can overwrite a row that writer commits. Checking that no process is mid-maintenance is not enough. |
| `restart_required` | The authority no longer describes the stored vector space, or maintenance was left incomplete. | Quiesce, re-embed offline if the existing vectors must be kept, restart every worker together. |
| `blocked_repair` | Storage holds invalid legacy data or an incompatible schema and is fenced off. | Offline repair; see below. |

Memory API routes answer `503` in every state except `ready` and
`not_configured`, with one stable detail that does not distinguish the faults.
`/api/memory/store-info` keeps answering `200` in every state, including when
the database connection pool is exhausted: it then reports the last published
state rather than failing.

Tasks and chats continue to start while memory is fenced off; they run with
memory disabled and an inert store, so nothing reads from or writes to the
storage admission refused.

Where the reason is recorded, and where it is not:

* **Recorded.** The lifecycle state reaches the task's execution metadata
  (`memory_available`, `memory_availability_reason`), which rides into the
  tracing backend and into the execution checkpoint, and it is reported by the
  internal `AgentService` status. That is what to read when asking why a
  particular task ran without memory.
* **Not recorded.** It is deliberately *not* written to the durable `Task` or
  `TaskChatMessage` rows, and it does not appear in the `TaskInfo` or
  task-completion payloads a caller receives. Do not query task records or
  public task payloads for it; use the execution metadata above, or
  `GET /api/memory/store-info` for the worker-wide state.

The operator log carries the detail the API deliberately does not. Search for
`Persistent memory` at `WARNING` and `ERROR`; each non-ready state logs the
specific quiescence and repair steps for that state.

### Serving in text_only mode

`state` is `ready` and `mode` is `text_only` when admission found the stored
vectors were written under a different embedding identity than the authority
now describes. This is a degraded but stable serving state, not a fault, and it
is not the same as "memory works normally":

* Memory is **readable and writable**. Tasks and chats keep using it.
* New notes are stored **without vectors**. They are reachable only by lexical
  search, never by semantic similarity.
* **Vector search is off** for the whole table. `supports_vector_search` is
  `false`, and searches fall back to lexical matching over every row,
  pre-existing and new alike.
* Existing vectors are **left exactly as they are**. The runtime holds no
  embedding adapter in this mode, so no write re-embeds a historical row and
  no write changes the table's vector width. Nothing mixes two vector spaces.

Restoring vector search is offline work, and it does not happen by itself:

1. Decide which vector identity the table should be in. Either point the
   authority back at the identity the stored vectors were written under, or
   keep the current authority and re-embed the data to match it.
2. If re-embedding: quiesce all writers, back up the memory LanceDB directory,
   and re-embed every row offline under the current authority. Do not re-embed
   in place, and do not start a worker to do it.
3. Restart every worker together. Confirm `mode` is `vector` and
   `supports_vector_search` is `true`.

Leaving a deployment in `text_only` indefinitely is a supported choice, as long
as it is a deliberate one: semantic recall stays off until the step above is
done, and every note written in the meantime will need the same offline
re-embed before it becomes semantically searchable.

### Repairing BLOCKED_REPAIR

`blocked_repair` means admission found invalid legacy data (a NULL, empty or
duplicate note id, non-string metadata, or a `user_id` outside signed int64) or
an incompatible schema, and refused to touch the table. Admission never mutates
storage in this state, so the data on disk is exactly what it was.

1. Stop every worker. Repair is offline work.
2. Back up the memory LanceDB directory (`<storage root>/memory_store`, or the
   project-local `memory_store/` when that legacy location is in use).
3. Repair or remove the offending rows against the backup, not in place.
4. Start every worker together and confirm `state` is `ready`.

Do not start a single worker to "test" a repair: a worker that admits
successfully begins writing, and the rest of the fleet has not admitted.

### Rollback

Rolling back is **not** symmetric with rolling forward, and quiescing writers
is not sufficient on its own. The previous version does not read the authority.
It picks its embedding model per user from the model hub, it has none of the
admission checks this release added, and so it will happily open the memory
table under an identity that has nothing to do with the vectors in it. Three
things follow, and all three are silent:

* **Online re-embedding.** If the identity the old code selects has a
  different dimension than the stored vectors, the first ordinary write
  rewrites the whole table to the new width and re-embeds every historical row
  under the new model. This is irreversible and there is no prompt.
* **Mixed vector spaces.** If the identity has the *same* dimension but a
  different model or endpoint, nothing rewrites and nothing complains. New
  vectors are simply written into a different space than the old ones, and
  recall degrades in a way that no state field reports.
* **Silent loss of durability.** On an unsupported provider configuration the
  old manager falls back to an ephemeral in-memory store. Memory appears to
  work and is discarded when the worker exits.

The previous release also lets an authenticated update overwrite a note's
`user_id` and scope metadata, so a note can silently end up owned by another
user or placed in another scope, with values that are well formed. The table
keeps this release's full-admission marker, so the next roll-forward trusts
those rows. No later scan can undo this: nothing records the original owner,
so re-admission, clearing the marker or a forced full rescan all accept the
new values as valid.

So only a **compatible writer** may open the production memory table: a
runtime that enforces the same server-side owner and scope rules on every
write as this release does. The previous release is not one. Unless you roll
back to a build patched to enforce those rules, roll back with memory storage
detached (see below). With a compatible rollback build, still gate the
rollback on the persisted identity, not just on the fleet version:

1. **Quiesce every writer.** Stop all API and task-execution workers. Nothing
   below is safe against a live writer.
2. **Establish what the persisted vector identity is.** Read it from the
   memory table's schema metadata (`xagent.memory.vector_space`), or from this
   release's `GET /api/memory/store-info` before you stop the fleet — a `mode`
   of `vector` means the stored vectors match the current authority.
3. **Establish what identity the previous release would select** for the same
   deployment: the per-user default embedding model, or the model hub's
   configured embedding model, as that release resolved it.
4. **Take a backup and verify it.** Copy the memory LanceDB directory
   (`<storage root>/memory_store`, or the project-local `memory_store/` when
   that legacy location is in use) while writers are still fenced, then verify
   the copy opens and its row count and vector width match the original. An
   unverified copy is not a backup.
5. **Compare the two identities.**
   * **They match** — this proves only that the identities agree, not that
     the previous release will persist anything. Its memory manager resolves
     the embedding model from the model hub (never from the authority) and,
     when it cannot build a persistent store for that row — a failed lookup, a
     provider it does not build a store for, a construction error — it logs
     the error and serves the in-memory store instead. So before rolling back,
     get positive evidence for every identity step 3 found:
     * a model hub row the previous release will select (the user's default
       embedding model, else the first active embedding model visible to the
       user), active and carrying that identity; and
     * a verified start of the previous version, with memory storage detached
       (both memory directory locations moved aside) and no other worker
       running, in which a memory request made as that user (for example
       `GET /api/memory/list`) is followed by `GET /api/memory/store-info`
       reporting `store_type` `LanceDBMemoryStore`, `is_lancedb` `true` and
       the expected `embedding_model_id`. `InMemoryMemoryStore` or
       `is_lancedb` `false` means the fallback, not a persistent store.

     With both, and only for a rollback build that is a compatible writer,
     stop that instance, discard whatever memory directory it created,
     reattach the real one, and redeploy that build to every worker at once.
     Leave the authority row in place: the old code ignores it, and the new
     version needs it on the next roll-forward. Without both, treat the case as
     unprovable and follow the next branch.
   * **They differ, or the persisted identity or the previous release's
     persistence under it cannot be proven** — do **not**
     start the previous version against this data. Keep memory fenced and pick
     one: restore a verified backup that was written under the identity the
     previous release will select, or re-embed the table offline into that
     identity before rolling back. Only then redeploy.

If the rollback build is not a compatible writer, or you must roll the
application back before either of those can be done, roll back with memory
storage detached: move the memory directory aside so the old code starts
against an empty one, and keep the real one out of its reach. Unrelated
functions are unaffected either way; persistent memory is the only thing at
stake.

When you roll forward again, reattach only storage that compatible writers
alone have written: the directory you moved aside, or a verified backup taken
before the rollback. Notes written while memory was detached are not merged
back. Reconcile them separately, from a source you trust, if they matter.

Rolling back does not undo an offline repair, and does not need to.

## 2026-09-25 — Task cleanup obligations and retry driver

Task deletion commits its rows before it releases what those rows located — the task's workspace directory and any runtime-extension state — because the rows are how the resources are found and a rollback cannot restore a removed directory. A release that failed, or a process that died between the commit and the release, used to leave nothing behind but a log line. Each such release is now recorded in `task_cleanup_obligations`, in the same transaction as the row deletion, and deleted once the resource is gone.

### Deployment impact

- Migration `20260925_task_cleanup_obligations` adds the table. It has no foreign keys and waits for no other table.
- Every web process starts a retry driver, whatever the retention settings. On-demand task and account deletion record obligations on any store, SQLite included. The driver claims each obligation with a compare-and-set, so several replicas can run it at once.
- The retention purge now records the workspace and bound extensions of every conversation it expires. It still releases nothing itself, because it holds the task's row lock; the driver releases them afterwards. The purge's audit line gains `cleanup_owed`.
- `on_task_deleted` can now run after the task row is gone: the driver dispatches it for everything the purge records and for any binding whose provider is not registered in the deleting process. The extensions an admin force-deleted past a *failing* provider are recorded but never retried.
- Both deletion endpoints add `external_cleanup_pending` to their response, next to `workspace_cleanup_pending`. It is true when the workspace or any runtime-extension state is still owed. An out-of-tree provider must not need the task row to release its state (see `TaskRuntimeExtensionProvider` in `src/xagent/core/task_runtime.py`).

### Prerequisites and configuration

**Workspace directories must be visible to every replica.** The driver on one replica may retry a removal recorded on another. A replica that cannot see the directory finds nothing, and finding nothing counts as success. A deployment that keeps workspaces on per-node local disks would therefore report leaked directories as removed. Such a deployment needs a host-affinity change before it relies on this record.

`XAGENT_TASK_CLEANUP_RETRY_INTERVAL_SECONDS` (default 300) sets how often an idle driver looks for due obligations. `XAGENT_TASK_CLEANUP_MAX_ATTEMPTS` (default 8) sets the attempt budget. Retries back off exponentially from five minutes, doubling each time (5, 10, 20, ..., 320 minutes for the default eight attempts) up to a six-hour cap the default budget never actually reaches, so it keeps retrying for roughly ten and a half hours in total.

### Verification and monitoring

A batch that claimed anything logs one line beginning `task cleanup retry`, counting `claimed`, `completed`, `retrying`, `exhausted` and `abandoned`.

`xagent retention cleanup-pending` opens the database read-only. It prints how many obligations are still being retried and lists the ones retrying will not finish:

- **`exhausted`** — the attempt budget ran out.
- **`abandoned`** — deliberately not retried. This covers three cases: an admin's force delete, a workspace whose execution scope could not be resolved before its rows were deleted (the unscoped candidates were cleared, but a scoped workspace cannot be located), and a task id that belongs to a live task again. SQLite reuses the highest deleted id, so an old obligation must not remove the new task's directory.

Add `--all` to include the obligations that are still pending. Reconcile each listed row by hand, then delete it from the table.

A runtime-extension obligation whose provider is not registered in this process is not listed as `exhausted` or `abandoned`: it stays `pending` and is rechecked hourly, without spending its attempt budget, until a process that has the provider registered claims it. Such rows show under `--all`, not in the default list.

### Rollback

Rolling back the application leaves the table in place and unread. Downgrading the migration drops it, and with it the record of every cleanup still owed. Export `xagent retention cleanup-pending --all` first.

## 2026-09-26 — Expired-task records

When the retention purge expires a conversation it now leaves a record, so that later readers can report "expired under the retention policy" instead of the not-found a deleted task gets. Nothing reads these records yet; the v1 API and internal surfaces follow separately (#2565). Only the purge writes them, and no deployment configures a retention period yet, so this change is inert on its own.

### Deployment impact

- Migration `20260926_expired_task_tombstones` adds the `expired_task_tombstones` table and three nullable columns: `tasks.traces_expired_at`, `trigger_runs.task_expired_at` and `workforce_runs.task_expired_at`. The columns have no server default, so on PostgreSQL each `ADD COLUMN` is catalog-only: no table rewrite and no backfill on `tasks`.
- Conversation expiry writes one tombstone per expired task, in the purge's own transaction. It holds no conversation content — only the owner, agent, workforce, source, trigger type, visibility and MCP channel-plumbing marker that the read surfaces' access checks need, plus the task's creation and expiry times. Expect roughly 150–200 bytes per expired task including indexes.
- The same transaction sets `task_expired_at` on every trigger and workforce run that pointed at the task, without changing the run's status and without advancing `trigger_runs.updated_at` or `workforce_runs.last_activity_at`.
- Trace expiry stamps `tasks.traces_expired_at` each time it removes a trace.
- User-initiated task deletion is unchanged and writes no record.
- Tombstones are removed with their owner's account. Deleting an agent or workforce clears the tombstone's reference to it, as agent deletion already does for live tasks. How long tombstones are otherwise kept is an open policy decision (#2567); today they are kept indefinitely.

### Rollback

Rolling back the application leaves the table and columns in place and unwritten. Downgrading the migration drops them, which loses the record of which tasks retention expired; readers then fall back to not-found for those ids.

## 2026-09-26 — v1 API reports expired tasks

The `/v1` SDK API now reads the expired-task records above (#2565). Internal web surfaces and run history follow separately. No deployment configures a retention period yet, so no tombstone exists and nothing below is observable until one does.

### Client-visible changes

- Every route addressed by a task id — `GET /v1/chat/tasks/{id}`, `/steps`, `/events`, `POST .../messages`, `POST .../reply` and `POST /v1/chat/files?task_id=` — answers **`410`** with error code **`task_expired`** and `details: {task_id, expired_at}` when the retention policy expired a task the calling key could have seen. A key that could not have seen the task (another agent or workforce, or a task not created through the SDK) keeps getting `404 task_not_found`, and so does a task its owner deleted.
- An events stream that is already open when its task expires closes with a terminal `stream.error` frame with code **`task_expired`** instead of `task_deleted`. Re-attaching then gets the `410`.
- `GET .../steps` has two new fields, `steps_expired` (boolean) and `steps_expired_at` (timestamp or null). `steps_expired: true` means the retention policy removed the task's historical steps, so the list may be incomplete. It can still hold steps from turns taken after the removal.

A client that does not know the new code still sees a 4xx for the REST routes and a terminal error frame on the stream — the same classes of outcome it gets today for a missing or deleted task. The new `/steps` fields are additive.

### Deployment impact

- No migration and no configuration.
- Responses cached by the steps cache before this change are still served for tasks the retention policy never touched. The first read of a trace-expired task after the upgrade re-reads and re-caches it.
- Rolling back makes the API answer `404 task_not_found` again for expired tasks and drops the two `/steps` fields.

## 2026-09-28 — Internal surfaces report expired tasks

The web UI and internal API now read the expired-task records above (#2565), the same way the `/v1` API already does. No deployment configures a retention period yet, so no tombstone exists and this change is inert until one does.

### Client-visible changes

- `GET /api/conversation-logs/{id}` answers **`410`** with `detail: {code: "task_expired", message, task_id, expired_at}` for a task the retention policy expired that was a hidden, external-source conversation (not MCP channel-plumbing) the calling user could have seen live — their own, or any such log for an admin. `GET /api/chat/task/{id}` and `GET /api/chat/task/{id}/status` answer the same `410` for a task the caller could have seen live regardless of source or visibility — their own task, or any task for an admin. Anyone else keeps the same `404` a task that never existed gets, so the answer cannot be used to probe which ids retention expired. Webhook trigger conversation logs are included on the Conversation Logs route; scheduled and Gmail trigger ones are not, since Conversation Logs never served those live either. `PUT /api/chat/task/{id}` and `DELETE /api/chat/task/{id}` are unchanged and keep answering plain `404`: an expired task is never listed, so the UI has no rename or delete control open on one to call either from.
- Conversation Logs detail gains `trace_events_expired_at`, alongside the existing `trace_events`.
- The trigger-run and workforce-run APIs gain `task_expired_at`; `status` is unchanged. Fixing that field's serialization to go through the same UTC-aware formatter as the rest of the API means every trigger-run and workforce-run timestamp field (`created_at`, `updated_at`, `started_at`, `finished_at`, `next_run_at`, `last_run_at`, `completed_at`, and now `task_expired_at`) now always carries an explicit UTC offset (`+00:00`) on SQLite; PostgreSQL's output is unchanged, since it was already tz-aware.
- The web UI shows "Conversation expired on {date}" in run history where it showed a dash or "Conversation unavailable" for a run whose task is gone, and an expiry or trace-removed note on the Conversation Logs page.
- Lists are unchanged: an expired task is simply absent from them, its source counts and agent filter options included.

### Deployment impact

- No migration and no configuration.
- Rolling back returns the previous `404` for these routes and drops `trace_events_expired_at` and `task_expired_at`.

## 2026-09-28 — Per-team retention period overrides

The two retention variables are deployment-wide. A deployment layer that lets a team shorten or extend them (shortening on every plan, extension up to two years) now has a hook for it. Nothing in this repository registers one: unregistered, the purge uses the two variables alone and runs the same scan.

### Deployment impact

- No migration, no index, no new environment variable.
- The purge's audit line gains `skipped_not_due` and `skipped_override_unresolved`. `skipped_not_due` applies with no resolver registered too: a quiescent task whose period has not elapsed under the lock — for example one whose newest message is more recent than its stored anchor — used to be counted as `skipped_busy`. Alerts or dashboards keyed on `skipped_busy` will see it drop by that amount.

### The resolver contract

A deployment layer registers a resolver with `xagent.web.services.task_retention_overrides.set_retention_override_resolver`. It maps each `user_id` whose team has an override to its conversation and trace periods, either of which may be left to inherit the deployment's value.

- An override may set 1 to 730 days. Any other value, including `None`, is refused rather than clamped, and that user's tasks are kept. An unusable value refuses the user even on a leg the deployment left off.
- A resolver that raises, returns something that is not a mapping, or returns a non-`int` user id keeps everything: the batch selects nothing, and a task already selected is counted as `skipped_override_unresolved`. A single user's value that is not a `RetentionOverride` (or an override with an unusable period) refuses only that user, and the batch proceeds for everyone else. Nothing falls back to the global periods, which would expire a team that extended its period early.
- An override never enables a leg the deployment left off, and never starts a purge the variables leave disabled.
- An inherited trace period follows the team's conversation period when the deployment's trace period equals its conversation period — which is what leaving `XAGENT_TRACE_RETENTION_DAYS` unset produces, but also what an operator gets by setting it explicitly equal to the conversation period. Either way, a team extending its conversation period also extends its trace period past the value the operator set. A team shortening its conversation shortens the trace with it, which deletes nothing earlier: conversation expiry removes the whole task, traces included, at that point anyway. Otherwise the trace period stays the deployment's own.
- Overrides are read once per batch for the scan and again per task inside the task's own transaction, just before its row is locked, so a settings change committed before then applies to that task. The resolver therefore runs inside the purge's transaction and must be a read with no side effects, dry run included.

### Verification and monitoring

- A failed resolver read shows in the audit line only as `scanned=0`, which is indistinguishable from an idle batch. The warning `retention override resolver failed` in the log is the signal, and the cursor restarts from the top of the table on the next sweep.
- Refused users are logged once per batch with up to 20 of their ids. A non-zero `skipped_override_unresolved` is also logged as one WARNING per batch (`retention purge kept N task(s) ...`). It has two possible sources: the resolver failing on the per-task reads (the traceback for each task is logged at DEBUG by `xagent.web.services.task_retention_overrides`), or a user whose value became unusable after the scan.
- `xagent retention preview` still counts against the global periods only; it does not apply overrides.

### Rollback

Unregister the resolver (or stop registering it) and restart: the purge returns to the two variables alone. Rows already expired under a team's shorter period are not restored.
