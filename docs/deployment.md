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

## 2026-08-18 — Bounded durable attachment uploads

### Deployment impact

Chat attachments now always enter a process-local durable-upload admission gate before object storage registration. The frontend also limits one browser composer to three active attachment requests. This reduces connection bursts from multi-file selections while retaining the existing retryable HTTP 503 contract. The browser limit is demand reduction, not a server security or capacity boundary.

S3 clients now default to standard retry mode with three total attempts when `XAGENT_FILE_STORAGE_OPTIONS` does not provide an explicit retry configuration. There is no database migration, backfill, new dependency, or infrastructure requirement.

### Prerequisites and configuration

The defaults permit four active durable upload registrations per backend process and let a staged request wait 30 seconds for capacity:

```text
XAGENT_FILE_UPLOAD_MAX_CONCURRENCY=4
XAGENT_FILE_UPLOAD_QUEUE_TIMEOUT_SECONDS=30
```

Both variables are optional. The admission gate cannot be disabled; these variables only tune its concurrency and wait time. The deployment-wide maximum is approximately the configured concurrency multiplied by the number of backend processes. Tune the value against the object-storage connection pool and the number of backend processes; it is not a cluster-wide limit.

`XAGENT_FILE_UPLOAD_MAX_CONCURRENCY` bounds active durable registrations only. It does not bound waiting requests, parsed multipart bodies, previews, local staged files, or aggregate staged bytes. Estimate staging capacity from peak accepted upload traffic over `XAGENT_FILE_UPLOAD_QUEUE_TIMEOUT_SECONDS`, including the backend per-file limit and proxy request limits. Configure an upstream upload request-rate or concurrency ceiling when the calculated exposure can exceed available staging resources.

Existing `XAGENT_FILE_STORAGE_OPTIONS` retry settings remain authoritative. When retries are not explicitly configured, the backend uses standard mode with three total attempts. To preserve the current retry behavior while classifying failures, set an explicit `config_kwargs.retries` value under `XAGENT_FILE_STORAGE_OPTIONS` before the rollout.

### Deployment and migration steps

Exception-chain logging, always-on admission control, and the default standard retry policy arrive in one backend artifact; they cannot be deployed as separate stages.

1. Before deploying the backend, tune the admission limits for each process and, if needed, set an explicit `config_kwargs.retries` value matching the current retry behavior.
2. Deploy the combined backend artifact to all backend processes.
3. Reproduce or observe the storage failure. Use the logged `operation`, `backend`, and exception chain to classify it as transient/throttling, pool pressure, configuration/permission failure, or local capacity failure.
4. Repair configuration, permissions, or local capacity when the failure is not transient; do not increase retries for those failures.
5. Verify backend health, complete one single-file upload, and exercise a concurrent upload burst. Confirm that registrations respect the per-process limit and that excess requests either acquire capacity or return the documented retryable `503` after the configured wait.
6. Deploy the matching frontend, then upload more than three files from one composer and verify that requests complete without an unrestricted browser burst.

No maintenance window or data migration is required. Mixed frontend versions are compatible with the new backend because the upload API request and response shapes are unchanged.

### Verification and monitoring

Monitor backend logs for these messages:

- `Durable storage unavailable during upload` identifies `operation=register_local_uploads`, the configured storage `backend`, and the chained provider or filesystem exception.
- `Timed out waiting for durable upload capacity` indicates that the process-local admission queue exceeded its configured wait.

Verify that persistent provider failures still return `503` with `Durable storage is temporarily unavailable`, and that successful attachments remain available when another selected file fails.

### Rollback

The frontend and backend can be rolled back independently because no persisted schema or API shape changed. Rolling back the backend removes admission control and restores the previous default retry configuration. As an operational mitigation before rollback, increase `XAGENT_FILE_UPLOAD_MAX_CONCURRENCY` or supply the previous retry policy through `XAGENT_FILE_STORAGE_OPTIONS`; monitor storage pressure while doing so.
