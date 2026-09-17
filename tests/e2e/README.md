# E2E Tests

## Running The Tests

Run all e2e tests:

```bash
uv run --project . --group test python -m pytest tests/e2e --run-special -q
```

## Shared Execution

Every test under `tests/e2e` enables shared execution. The autouse fixture in
`conftest.py` overrides the legacy suite's local-execution default. Existing
application harnesses use the combined role; the shared execution tests start
separate HTTP/WebSocket web hosts and standalone workers with isolated SQLite
databases and storage directories.

Redis is required. Set `XAGENT_TEST_REDIS_URL` to a disposable Redis instance,
or install `redis-server` so the fixture can start and stop a local instance.
CI supplies a Redis service. Missing Redis fails setup rather than skipping
these tests. Each test uses a separate event namespace and encryption key.

Run the shared execution tests without Docker:

```bash
uv run --project . --group test python -m pytest tests/e2e/test_shared_*.py --run-special -q
```

The model boundary is deterministic, and platform transport calls and sandbox
provisioning are replaced. Public HTTP/API-key/JWT and WebSocket authorization,
durable commands, Redis event delivery, worker startup/shutdown, AgentService,
AgentRunner, checkpoints, file tools, and database projections remain real.
The subprocess hosts also run production background loops instead of inheriting
pytest's automatic startup skips.

| Test file | End-to-end coverage |
| --- | --- |
| `test_shared_execution.py` | WebSocket execute/chat, live guidance deduplication, final-answer streaming and history replay; SDK create/append/poll/SSE and file upload/read/write/download; SDK and A2A reply after replacing the worker; A2A stream/re-subscribe/query/cancel; pause/resume with competing workers; agent webhook/test/scheduled triggers; workforce owner/SDK/widget/share/preview/trigger execution including child delegation; web exit, worker SIGTERM, and recovery/resume after worker SIGKILL. |
| `test_shared_channels.py` | Slack, Feishu, and Telegram callbacks through real SharedChannelTurn execution and forwarded traces; Telegram attachment download, worker transformation, and returned bytes; queued pause and channel deactivation before worker consumption. |
| `test_shared_happy_paths.py` | Agent public chat/reply/follow-up and file transfer; A2A blocking/context continuation; all execution modes and auto branches; combined host; legacy webhook. |
| `test_shared_gmail.py` | Agent and workforce Gmail callback through actual signature verification, mail ingestion, and worker completion. |
| `test_shared_network_recovery.py` | A controllable TCP link interrupts only the test hosts' Redis connections. An in-flight task still persists its result, an unaccepted START leaves no task behind, and clients recover after reconnection. |

In `test_shared_channels.py`, the bot callbacks and their event bridge run in
pytest's process. Spawned web/worker hosts have channel ingress disabled, so
these cases do not exercise a live platform connection or designated-ingress
subscription startup. `test_channel_delivery_recovery.py` separately kills and
restarts an ingress subprocess and verifies persisted reply recovery through the
Feishu renderer, with platform network sends simulated. The four Docker suites
also inherit the shared-execution autouse fixture.

Outside E2E, the suite defaults to local execution; shared service/worker tests
explicitly override that fixture. Passing legacy tests alone does not validate
the shared default.

The existing MinIO and PostgreSQL 17 E2E suites still require Docker; they also
run with shared execution enabled. The SQLite subprocess tests do not replace
those storage/backend-specific checks.

## Shared Execution Happy-Path Matrix

Scope: supported persisted task execution entry points, successful conversation
operations, execution patterns, and host roles. Each row must reach a real final
result or a real waiting checkpoint followed by a successful reply. This is an
API/callback E2E matrix, not browser UI or external-account provisioning coverage.
Execution modes and host roles are checked once across their shared runtime;
we do not multiply every entry by every mode, tool, media type, and deployment.

| Entry / branch | Successful flow | Test |
| --- | --- | --- |
| Owner WebSocket execute | Existing task -> execute -> streamed answer -> persisted history | `test_websocket_execute_stream_and_reconnect` |
| Owner / Agent widget / Agent share chat | Create/auth -> question -> waiting -> reply -> completed -> follow-up -> history | `test_agent_chat_reply_and_followup` (3 cases) |
| Owner / Agent widget / Agent share files | Create -> upload -> chat -> real file read/write -> returned file -> guest-authorized download bytes | `test_chat_file_input_and_output` (3 cases) |
| SDK Agent | Create -> poll/SSE -> completed -> append | `test_sdk_create_append_and_poll_execute_only_in_worker` |
| SDK Agent files | Upload -> create with input -> real read/write -> download bytes | `test_sdk_file_input_tool_output_and_download` |
| SDK / A2A waiting reply | Waiting checkpoint -> reply -> same run completes on a replacement worker | `test_waiting_reply_restores_checkpoint_in_new_worker` (2 cases) |
| A2A blocking send / context continuation | Text send -> completed artifact/query -> JSON data in same context -> new completed task | `test_a2a_blocking_send_and_context_continuation` |
| A2A streaming | Stream -> subscribe -> final artifact/status | `test_a2a_stream_disconnect_and_resubscribe` |
| Workforce owner / SDK / widget / share / persisted preview | Upload input -> create run -> manager delegates to child and transforms file -> download output -> follow-up question -> waiting -> reply -> completed | `test_workforce_manager_and_child_execute_in_worker` (5 conversation cases) |
| Agent webhook / test / scheduled | Authenticated callback or manual test or due dispatcher -> completed task and TriggerRun | `test_trigger_webhook_and_test_reach_terminal_task`, `test_scheduled_dispatcher_runs_task_and_settles_trigger` |
| Workforce webhook / test / scheduled | Trigger -> manager/child execution -> completed task, WorkforceRun and TriggerRun | `test_workforce_manager_and_child_execute_in_worker` (3 trigger cases) |
| Agent / Workforce Gmail | Signed push -> actual OIDC verification -> mail history/message fetch -> completed task/run | `test_gmail_push_runs_to_completion` (2 cases) |
| Legacy webhook | Persisted legacy trigger -> secret verification -> callback -> completed task/run | `test_legacy_webhook_reaches_completed_task` |
| Slack / Feishu / Telegram text | Callback -> completed -> follow-up on same task -> platform final response | `test_channel_callback_runs_remotely_and_returns_answer` (`text`, 3 cases) |
| Slack / Feishu / Telegram reply | Callback -> question/waiting -> user reply -> completed -> platform final response | `test_channel_callback_runs_remotely_and_returns_answer` (`reply`, 3 cases) |
| Slack / Feishu / Telegram files | Platform download -> registration -> worker read/write -> platform file or link with correct bytes | `test_channel_callback_runs_remotely_and_returns_answer` (`file`, 3 cases) |
| Execution patterns | flash, balanced, think; auto chooses direct answer, ReAct, or DAG -> final result | `test_execution_modes_reach_final_result` (6 cases) |
| Shared host roles | Separate web + worker for entry tests; combined host -> SDK task -> result | `shared_app`, `test_combined_host_runs_shared_task` |
| Normal runtime controls | Running message delivered once; pause then resume; A2A cancel | `test_websocket_running_message_is_delivered_once`, `test_websocket_pause_and_resume_with_two_workers`, `test_a2a_cancel_reaches_running_worker` |

Contract boundaries: A2A accepts text/JSON input, not file parts; terminal A2A
tasks continue through a new task with the same context ID. Feishu currently
returns a managed file link rather than uploading an output attachment. Gmail
starts with a connected OAuth account and provisioned watch as fixture data;
Google certificate and mailbox network responses are deterministic, while token
signature/issuer/audience verification and callback processing are real.
Non-persisted builder preview, browser UI, provider OAuth setup, individual tool
integrations, and every backend/media combination are outside this matrix.
Docker MinIO/PostgreSQL 17 validation remains a separate required CI lane.

## File Persistence E2E Expected Behavior

This document maps the expected durable file storage behavior to the e2e tests that cover it.

The durable file storage implementation is a first-phase, local-first persistence
layer with casual consistency across local disk, the `uploaded_files` table, and
S3/MinIO. It does not try to provide distributed transaction semantics. The
normal contract is:

- Local files are preferred when present.
- Durable storage is the fallback when the registered local file is missing.
- DB rows are marked `storage_status = "available"` only after durable storage
  returns object metadata with a checksum.
- Startup repair reconciles DB-registered files into durable storage where local
  bytes or existing durable objects make repair possible.
- Transient durable-storage failures should fail closed and be retried; known
  unrecoverable local+remote data loss is surfaced later as a missing file rather
  than blocking all app startup.

### Expected Behavior And Coverage

#### Startup Sync

- App startup scans DB-registered files and uploads legacy local files to S3/MinIO when durable metadata is missing.
  Covered by: `tests/e2e/test_file_startup_sync_minio.py::test_startup_sync_repairs_only_files_that_need_durable_storage`
- App startup does not overwrite an object that already exists in S3/MinIO when DB durable metadata is complete.
  Covered by: `tests/e2e/test_file_startup_sync_minio.py::test_startup_sync_repairs_only_files_that_need_durable_storage`
- App startup repairs rows whose DB metadata says S3 is available but the remote object is missing, as long as the local file still exists.
  Covered by: `tests/e2e/test_file_startup_sync_minio.py::test_startup_sync_repairs_only_files_that_need_durable_storage`
- App startup skips rows whose local file is missing and remote object is missing, without failing app startup.
  Covered by: `tests/e2e/test_file_startup_sync_minio.py::test_startup_sync_repairs_only_files_that_need_durable_storage`

#### Upload And API Persistence

- User file upload writes the local file, creates an `uploaded_files` DB row, and persists the object to S3/MinIO.
  Covered by: `tests/e2e/test_file_api_minio.py::test_download_and_preview_materialize_uploaded_file_from_minio`, `tests/e2e/test_file_persistence_minio.py::test_task_uploads_agent_outputs_and_startup_sync_persist_to_minio`

#### Download And Preview

- Download can recover a missing local file from S3/MinIO and restore it to the registered local path.
  Covered by: `tests/e2e/test_file_api_minio.py::test_download_and_preview_materialize_uploaded_file_from_minio`
- Preview can serve a file from S3/MinIO when the registered local file is missing.
  Covered by: `tests/e2e/test_file_api_minio.py::test_download_and_preview_materialize_uploaded_file_from_minio`

#### Delete And Access Control

- Deleting a file removes the DB row, local file, and S3/MinIO object.
  Covered by: `tests/e2e/test_file_api_minio.py::test_delete_removes_uploaded_file_from_db_local_disk_and_minio`
- A durable storage cleanup failure fails the delete request with HTTP 503 and
  keeps the DB row, local file, and S3/MinIO object so the delete can be retried.
  Covered by: `tests/e2e/test_file_api_minio.py::test_delete_keeps_db_row_when_durable_cleanup_fails`
- Another user cannot download, preview, or delete a file they do not own, and the S3/MinIO object remains intact.
  Covered by: `tests/e2e/test_file_api_minio.py::test_file_routes_reject_cross_user_access_and_keep_minio_object`

#### WebSocket And Agent File Persistence

- WebSocket task execution persists agent-created output files to DB and S3/MinIO with output workspace metadata.
  Covered by: `tests/e2e/test_file_persistence_minio.py::test_task_uploads_agent_outputs_and_startup_sync_persist_to_minio`
- Chat/WebSocket task execution can materialize a missing local input from S3/MinIO before the agent reads it, then persist derived output to S3/MinIO.
  Covered by: `tests/e2e/test_file_persistence_minio.py::test_chat_task_materializes_missing_local_input_from_minio_before_agent_reads_it`

#### Remote Storage Outage Behavior

- User file upload fails with HTTP 503, removes temporary local files, and leaves no `uploaded_files` row when durable storage write fails.
  Covered by: `tests/e2e/test_file_api_minio.py::test_upload_returns_503_and_rolls_back_when_minio_write_fails`
- Download serves an existing local copy during durable storage outage, but returns HTTP 503 when the local copy is missing and durable storage cannot be read.
  Covered by: `tests/e2e/test_file_api_minio.py::test_download_serves_local_copy_when_minio_read_fails`, `tests/e2e/test_file_api_minio.py::test_download_and_preview_return_503_when_minio_read_fails_without_local_copy`
- Preview returns HTTP 503 when the local copy is missing and durable storage cannot be materialized.
  Covered by: `tests/e2e/test_file_api_minio.py::test_download_and_preview_return_503_when_minio_read_fails_without_local_copy`
- WebSocket output persistence rolls back the output DB row and fails task output normalization when durable storage write fails.
  Covered by: `tests/e2e/test_file_persistence_minio.py::test_websocket_output_persistence_sends_error_and_rolls_back_when_minio_write_fails`

### Supporting Harness Coverage

- E2E JWT helper creates a token with expected user claims.
  Covered by: `tests/e2e/test_app_harness.py::test_build_access_token_contains_user_claims`
- E2E local file seeding creates a physical file and matching `uploaded_files` row.
  Covered by: `tests/e2e/test_app_harness.py::test_seed_registered_local_file_creates_file_and_db_record`
- Scripted LLM JSON fixtures are converted into mock LLM responses, including native tool-call payloads.
  Covered by: `tests/e2e/test_scripted_llm.py::test_load_scripted_responses_converts_enveloped_entries`

### Current Boundaries

- These tests use the real FastAPI app startup path through `TestClient`.
- S3 behavior is exercised against Docker MinIO.
- LLM behavior is deterministic through `tests/e2e/scripted_llm.py` and JSON response fixtures.
- These tests do not cover a full remote S3 outage during app startup; that should be covered by a smaller service/integration test if needed.
- E2E outage tests cover upload, download, preview, delete, and WebSocket output persistence under durable storage failures by monkeypatching the storage layer while using the real FastAPI app, DB, and Docker MinIO configuration.
- Delete outage tests assert fail-closed behavior: user-facing delete does not
  remove the DB row when durable cleanup fails, and stale KB reconciliation can
  retry cleanup later.

### Production Durability Gaps (TODO)

These e2e tests verify the normal durable-storage contract, but they do not prove all production durability failure modes.

- Crash consistency is not covered. For example, a process crash after writing an object to S3/MinIO but before committing the `uploaded_files` row could leave an orphan durable object; a crash after DB metadata is committed but before local cleanup or response completion could leave partial local state.
- Read integrity verification is not covered. The storage layer records checksum metadata on write, but these tests do not assert checksum validation when materializing or downloading from durable storage.
- Concurrent writer and multi-process repair behavior is not covered beyond startup lock acquisition. Tests should cover duplicate uploads, repeated startup sync, and races between startup repair, user download, and delete.
- Object lifecycle cleanup is not covered. A process crash or late failure can
  still leave orphaned S3/MinIO objects, and these tests do not assert cleanup or
  reconciliation for objects that have no committed DB row.
- S3 checksum metadata repair is not covered. If object bytes are written but
  checksum metadata attachment fails, startup repair may see the object but
  refuse to mark the DB row available until metadata can be inspected or the
  object is manually repaired.
