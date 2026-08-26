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

## 2026-08-26 — PostgreSQL 17 default for the bundled Compose database

### Deployment impact

The bundled `postgres` service in `docker-compose.yml` now defaults to `postgres:17-bookworm`. This aligns self-hosted deployments with the PostgreSQL 17 major that production runs and that CI validates against.

PostgreSQL never upgrades a data directory across major versions. A `postgres_data` volume initialized by PostgreSQL 16 does not open under a PostgreSQL 17 server. The server exits with `FATAL: database files are incompatible with server` and does not modify the directory, so the v16 data stays intact and the failure is recoverable by pinning the previous tag.

Under `restart: unless-stopped` the container restart-loops as `unhealthy`. `backend`, `worker`, and `scheduler` wait for `service_healthy` and never start. Treat an unplanned upgrade of an existing v16 deployment as a full outage until the tag is pinned back or the migration below completes.

Fresh installations are unaffected. They initialize directly under v17. Deployments whose `DATABASE_URL` points at an external or managed PostgreSQL are unaffected, because the bundled service is not in their path.

This change has no Alembic revision, no schema change, and no application-code change. The migration is a data-directory migration only.

### Prerequisites and configuration

This release adds one environment variable. `POSTGRES_IMAGE_TAG` sets the tag of the bundled `postgres` image and defaults to `17-bookworm`. `example.env` documents it. Set it to `16-bookworm` to keep an existing v16 volume running until it is migrated.

Before migrating, confirm the deployment uses the bundled `postgres` service rather than an external database. Identify the Compose-prefixed volume name with `docker volume ls | grep postgres_data`. For the default project name it is `xagent_postgres_data`.

Reserve a maintenance window. The database is unavailable from the point writers stop until verification passes. `nginx` and `frontend` keep serving during that window and return errors to users. Stop them as well when user-visible errors are unacceptable.

Do not run `docker compose down -v` or `docker volume rm` at any point in this procedure. Both destroy the database irreversibly. Neither is an upgrade step.

### Deployment and migration steps

The executable commands for every step below are the [PostgreSQL major version upgrade (16 to 17)](../docker/README.md#postgresql-major-version-upgrade-16-to-17) runbook in `docker/README.md`. Run one runbook step per numbered step here, in order.

A deployment that uses a sandbox runtime overlay must keep its `-f` overlay arguments on every Compose command in that runbook. A bare `docker compose up -d` recreates `backend`, `worker`, and `scheduler` without the overlay and silently returns the deployment to non-sandboxed operation.

1. Pin `POSTGRES_IMAGE_TAG` to `16-bookworm` and confirm the deployment is healthy on v16 before anything else changes.
2. Stop `backend`, `worker`, and `scheduler`. Leave `postgres` running.
3. Take a dump and verify it is complete. An interrupted `pg_dump` still leaves a plausible file. Record the values that the verification section compares after the restore.
4. Stop the stack without `-v`.
5. Copy the v16 volume to a separate volume, so rollback never depends on the dump alone. Remove any copy left by an earlier attempt before creating it.
6. Remove the v16 data directory from the live volume, unpin `POSTGRES_IMAGE_TAG`, and start `postgres` on v17, which initializes a fresh cluster.
7. Restore the dump with `ON_ERROR_STOP=1`, then run the verification below.
8. Start the remaining services only after verification passes.

Keep the v16 volume copy until the v17 deployment has run to your satisfaction. Then remove it with `docker volume rm`.

### Verification and monitoring

Run the version query after the restore:

```sql
SHOW server_version;
```

The result must report a 17.x version.

Run the remaining queries against v16 before step 4 and against v17 after the restore, and compare the two results:

```sql
SELECT version_num FROM alembic_version;
```

The result must equal the value recorded before the migration. This is a schema check only. A restore that stopped early does not change this value.

```sql
SELECT count(*) FROM tasks;
SELECT count(*) FROM users;
```

Each count must equal the value recorded before the migration. These are the row-level checks. The two checks above do not prove that the data restored.

After verification passes, run `vacuumdb --all --analyze-in-stages` inside the `postgres` container. `pg_dump` does not carry optimizer statistics across a restore, and a restored cluster plans queries badly until statistics exist.

Confirm that `docker compose ps` reports `postgres` as `healthy`, and that `docker compose logs postgres` shows a startup with no `FATAL` line, before you start the writers.

### Rollback

A v17 server never opened the volume copy taken in step 5, so rollback restores that copy directly instead of replaying the dump.

1. Stop the stack without `-v`.
2. Remove the data directory from the live volume.
3. Copy the preserved v16 volume back over it.
4. Pin `POSTGRES_IMAGE_TAG` to `16-bookworm`.
5. Start the stack and confirm that `SHOW server_version` reports a 16.x version.

Roll back rather than repair in place when verification fails after a partial restore under v17. A cluster left half-populated by an interrupted restore is not a state to diagnose during an outage.

If the v16 volume copy is unavailable, restore the verified dump from step 3 onto a v16 cluster initialized from `16-bookworm`.
