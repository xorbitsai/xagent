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

The `user_oauth` table gets a nullable `resource_owner_key` column. Existing rows keep a null value, and every existing OAuth consumer explicitly selects that ordinary namespace.

Two partial unique indexes replace `uq_user_provider_account`. One index protects ordinary rows. The other separates actor-owned namespaces when `provider_user_id` is non-null. Standard SQL null semantics still permit multiple rows with the same actor key, provider, and null `provider_user_id`. SQLite and PostgreSQL are the only supported database dialects for this schema; startup and migration fail before schema creation on other dialects.

On PostgreSQL the migration creates the replacement indexes transactionally before removing the old unique constraint. A failed statement rolls back the complete schema transition. If a same-name relation causes the failure, an operator must inspect and remove or rename that relation before retrying `alembic upgrade head`. Index creation is not concurrent and can block writes to `user_oauth`, so plan a short OAuth-write pause and monitor lock wait time.

On SQLite the migration rejects globally colliding owner-index names before rebuilding the table in batch mode. Stop every worker before this rebuild and keep SQLite quiesced until the migration completes.

If the migration reports `UserOAuth schema is partially owner-aware`, do not start workers. The schema has either `resource_owner_key` and the old `uq_user_provider_account` constraint together, or neither one. Restore the last known complete schema from backup, or have a database operator finish one coherent legacy or owner-aware schema before retrying `alembic upgrade head`. Do not bypass this fail-closed check.

If SQLite reports that an owner-index name already exists before migration, query `sqlite_master` for that exact name and identify its owning table and definition. After taking a backup, remove or rename only the unrelated colliding index, then retry `alembic upgrade head`. If either database reports `owner-aware UserOAuth schema has incorrect indexes`, keep workers stopped and compare both index columns, uniqueness flags, and predicates with the verification definitions below. Repair or remove the incorrect owner indexes under database-operator supervision before retrying the migration.

### Prerequisites and configuration

This change has no new environment variable or dependency. Keep every future actor-OAuth caller disabled; this release does not expose a production path that creates actor-owned rows.

### Deployment and migration steps

Choose the procedure for the configured database.

#### SQLite

1. Stop new OAuth connections and task execution.
2. Stop every API and task worker.
3. Deploy the new application files without starting workers.
4. Run `alembic upgrade head` one time.
5. Start every API and task worker with the new version.
6. Verify the schema and homogeneous worker version.
7. Resume ordinary OAuth connections and task execution.

#### PostgreSQL

1. Pause new OAuth writes and make sure no long transaction holds a lock on `user_oauth`.
2. Run `alembic upgrade head` one time. Existing workers can continue non-OAuth work while the transactional DDL runs.
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

On PostgreSQL, verify that both partial unique index names exist and are valid:

```sql
SELECT c.relname, i.indisvalid
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

The query must return both rows with `indisvalid = true`.

For SQLite run `PRAGMA index_list('user_oauth');` and `PRAGMA index_info('<index-name>');`. Inspect `sqlite_master.sql` to confirm that the ordinary index uses `WHERE resource_owner_key IS NULL` and the actor index uses `WHERE resource_owner_key IS NOT NULL`.

Before restarting Gmail watch processing, run this query on either supported database:

```sql
SELECT count(*)
FROM gmail_watch_states AS watch
JOIN user_oauth AS account ON account.id = watch.oauth_account_id
WHERE watch.user_id <> account.user_id
   OR account.resource_owner_key IS NOT NULL;
```

The result must be zero. A nonzero result identifies a legacy watch whose account owner does not match its user or whose account is not ordinary; repair or remove that watch before rollout.

Verify existing cloud-storage, Gmail, and builtin OAuth connections. Confirm that seeded non-null-owner test rows do not appear in ordinary catalog, token, or trigger paths.

### Rollback

Because this release cannot create actor-owned rows, the downgrade remains available after ordinary rollout. Stop workers, run `alembic downgrade 20260818_seed_jira_mcp_app`, and deploy the old version.

The migration still refuses downgrade if a non-null owner row exists. If an external or future caller created such a row, disable that caller and use an approved credential-revocation and data-removal procedure before retrying the downgrade.
