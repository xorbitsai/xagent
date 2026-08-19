# Deployment changes

## 2026-08-17 — Google Calendar connector scope narrowing (`20260817_narrow_google_calendar_scope`)

### Deployment impact

The built-in Google Calendar connector now requests `calendar.events` instead of the full `calendar` scope. The migration deletes any `user_oauth` row with `provider = 'google-calendar'` whose granted scope still carries the old, full scope (including rows with no scope recorded at all — see the migration's module docstring for why that counts as evidence of the old grant), then removes the `user_mcpservers` row for any user left with no `google-calendar` credential after that. Affected users lose their Calendar connection and must reconnect through the OAuth flow again; there is no in-place upgrade of an existing token's scope.

This is irreversible. `downgrade()` restores the catalog's requested scope but cannot restore a deleted `user_oauth` or `user_mcpservers` row — there is no way to reconstruct a token or a server association this migration removed.

Do not downgrade this revision and re-upgrade it, and do not apply an offline-generated (`--sql`) script for it more than once, once Calendar connections have been made under the narrow scope: the empty/`NULL`-scope branch of the cleanup (see the migration's module docstring) cannot distinguish a genuinely new, narrow grant that omitted its scope echo from a pre-migration broad one, and a replay would delete the new grant too.

Both the online and offline paths run the `user_oauth` revoke as a single, unbatched `DELETE` — this migration commits in one transaction regardless (`transaction_per_migration=True` on PostgreSQL, see `src/xagent/migrations/env.py`), so every revoked row's lock is held until that same commit either way; there is no batching to trade off, and no lock-duration difference between the two paths.

The `user_mcpservers` orphaned-association cleanup, however, is **online-only**. Offline (`alembic upgrade --sql`) generation narrows the catalog and revokes `user_oauth` rows, but does not touch `user_mcpservers` at all — see the migration's module docstring for why (identifying the official Calendar server from row content isn't expressible as one portable SQL predicate the way the scope-content match is). An offline-applied upgrade does not reach the same database state as an online one. If a deployment relies on generating and applying `--sql` scripts for this revision, run the online migration path once against the same database afterward (or apply this specific revision online) if the orphaned-association cleanup matters to you; otherwise stale associations for bare-`google`-only Calendar users will persist indefinitely.

### Scope limitation

Only `provider = 'google-calendar'` rows are touched by the `user_oauth` cleanup. A bare provider-level `google` row (created by the app_id-less connect flow) is never deleted here even if it happens to carry the old scope too, since other Google connectors (Gmail, Drive, Docs, Slides, Sheets) may depend on that same row as their own fallback credential.

That gap is *mitigated*, not fully closed, at the runtime level: `google-calendar` was added to `APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT` (`web/mcp_apps.py`) in the same change, so a bare `google` row is no longer accepted as a Calendar credential regardless of its scope. This has a real, separate consequence covered below: any user whose *only* Calendar connection was ever a bare `google` row is broken by this policy change alone, independently of whether the `user_oauth` cleanup above revokes anything for them. This migration's `user_mcpservers` cleanup step accounts for that -- it runs after the revoke step and removes any Calendar `user_mcpservers` row left with no `google-calendar` credential, whether that's because the row was revoked this run or because it only ever had a bare `google` row and none at all.

This does **not** make a `google-calendar` row's own token trustworthy going forward -- see the migration's module docstring on why a reconnect can still, in principle, receive a token that again carries the old, broad scope. Deploy the data cleanup and the runtime policy change together (they ship in the same migration/PR); reverting one without the other reopens the gap the other was covering.

This migration does not call Google's token-revocation endpoint. The deleted local row simply stops being usable by this application; Google still considers the underlying token valid (subject to its own expiry) until the user revokes app access directly from their Google Account.

### Prerequisites and configuration

No new environment variable, dependency, or infrastructure requirement. The *schema* has no mixed-version ordering constraint: an older worker reading a `google-calendar` row that was already narrowed (or already deleted) behaves the same as it would against any other disconnected Calendar account.

The *code* policy change is a different story and does have a mixed-version window: `google-calendar`'s membership in `APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT` (`web/mcp_apps.py`) takes effect the moment a given worker process starts running the new code, independent of when the migration runs against the database. During a rolling deploy where old and new worker processes run concurrently, an old worker still accepts a bare `google` row as a valid Calendar credential while a new worker rejects it for the same user -- which worker happens to handle a given request decides whether Calendar works for that user in that window. This resolves itself once every worker runs the new code; it is a transient rollout-window inconsistency, not a persistent one, but it is real and should be expected, not assumed away.

Before deployment, get a precise count of affected users -- everyone who will lose Calendar access from either mechanism this migration/policy change touches, as one distinct set rather than two counts to eyeball and sum (PostgreSQL syntax; a SQLite deployment needs `json_extract(s.auth, '$.app_id')` in place of `s.auth->>'app_id'`):

```sql
SELECT DISTINCT user_id FROM (
  -- Users whose google-calendar grant will be revoked: the migration's
  -- exact predicate (NULL/empty scope, or scope still containing the old,
  -- full calendar scope) -- see _revoke_predicate in the migration.
  SELECT user_id
  FROM user_oauth
  WHERE provider = 'google-calendar'
    AND (
      scope IS NULL
      OR scope = ''
      OR (' ' || scope || ' ') LIKE '% https://www.googleapis.com/auth/calendar %'
    )

  UNION

  -- Users whose Calendar server association will be orphaned once the
  -- revoke above runs: they have no google-calendar row that will survive
  -- it, whether because they never had one (bare-google-only, connected
  -- via the app_id-less batch flow) or because their only one is being
  -- revoked by the query above. Matches _calendar_server_ids in the
  -- migration: prefer auth.app_id, fall back to name only for a legacy
  -- row with no app_id key, and only ever among transport = 'oauth' rows.
  SELECT um.user_id
  FROM user_mcpservers um
  JOIN mcp_servers s ON s.id = um.mcpserver_id
  WHERE s.transport = 'oauth'
    AND (
      s.auth->>'app_id' = 'google-calendar'
      OR (s.auth->>'app_id' IS NULL AND s.name = 'Google Calendar')
    )
    AND um.user_id NOT IN (
      SELECT user_id
      FROM user_oauth
      WHERE provider = 'google-calendar'
        AND NOT (
          scope IS NULL
          OR scope = ''
          OR (' ' || scope || ' ') LIKE '% https://www.googleapis.com/auth/calendar %'
        )
    )
) affected_users;
```

This is the exact set of users this migration disconnects from Calendar -- not an over- or under-count -- and is what to base user communication on.

### Deployment and migration steps

1. Deploy the application version carrying this migration; the migration runs automatically as part of normal startup/migration application.
2. Expect up to two log warnings: `Revoked %d google-calendar user_oauth grant(s) ...` and `Removed %d orphaned Calendar user_mcpservers row(s) ...`. Either can fire independently of the other -- a user can be affected by one, both, or neither. Neither log warning covers the mixed-version code-policy window described above.
3. Communicate to affected users (the preflight query above, run once before the deploy) that they need to reconnect Google Calendar.

### Verification and monitoring

Confirm the catalog row was narrowed:

```sql
SELECT oauth_scopes FROM public_mcp_apps WHERE app_id = 'google-calendar';
```

Connect a fresh test account to Calendar and confirm the Google consent screen requests only `calendar.events`, not the full `calendar` scope. A fresh test account only exercises the scope-narrowing half of this change -- it has no prior grant history, so it cannot demonstrate the reconnect-regains-broad-scope risk documented in the migration's module docstring; that risk needs an account that previously granted the old, broad scope.

For any user who reports a broken Calendar connection after this deploys, direct them to reconnect — this is the expected, intended effect of the migration, not a bug.

### Rollback

`downgrade()` restores the catalog's requested scope to the full `calendar` scope, but every `user_oauth` and `user_mcpservers` row this migration removed stays removed — there is no data to roll back to. A rollback only affects what *new* authorizations request; it does not restore anyone's Calendar connection.

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
