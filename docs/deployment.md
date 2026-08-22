# Deployment changes

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

Two partial unique indexes replace `uq_user_provider_account`. One index protects ordinary rows. The other separates actor-owned namespaces when `provider_user_id` is non-null. Standard SQL null semantics still permit multiple rows with the same actor key, provider, and null `provider_user_id`. PostgreSQL is the only supported production database, including self-hosted production installations. SQLite is supported only for local development and CI. Startup and migration fail before schema creation on other dialects.

The `users` table is an application-metadata table and must exist before this revision runs. Do not use bare Alembic to initialize a genuinely empty application database; normal application startup stamps the empty database before creating metadata-owned tables. On an existing legacy schema, this revision installs the missing `user_oauth.user_id -> users.id ON DELETE CASCADE` foreign key during the owner-aware transition. It fails closed when `users` is absent or when an already owner-aware schema is missing that cascade.

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

For local SQLite, run `PRAGMA index_list('user_oauth');` and `PRAGMA index_info('<index-name>');`. Inspect `sqlite_master.sql`. The ordinary index must use `WHERE resource_owner_key IS NULL`. The actor index must use `WHERE resource_owner_key IS NOT NULL`. Run `PRAGMA foreign_key_list('user_oauth');`. Require a `user_id -> users.id` row with a `CASCADE` delete action. On PostgreSQL, inspect `user_oauth` constraints. Require the equivalent cascade before you enable actor-owned rows.

Verify existing cloud-storage and builtin OAuth connections. Confirm that seeded non-null-owner test rows do not appear in ordinary catalog or token paths.

### Rollback

Because this release cannot create actor-owned rows, the downgrade remains available after ordinary rollout.

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
