# Execution events: observability runbook

This runbook is for on-call staff running tasks on the V2 execution-event log
(`task_execution_events`, enabled per task by
`Task.conversation_storage_version = 2`). It lists the signals that exist
today, which problem class each one covers, and how to triage a single task
with the admin diagnostic endpoint.

Read-budget signals do not exist yet. Further signals will be added; this
document lists only what the code emits now.

## 1. Turning signals on

### Runtime metrics (OTLP/HTTP)

Metrics are exported only when export is enabled and an endpoint is resolved.

| Variable | Meaning |
| --- | --- |
| `XAGENT_RUNTIME_TELEMETRY_ENABLED` | Explicit on/off. When unset, export is enabled automatically if any OTLP metrics endpoint below is configured. `false` is a kill switch. |
| `XAGENT_OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | Full OTLP/HTTP metrics URL, including `/v1/metrics`. |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`, `OTEL_EXPORTER_OTLP_ENDPOINT` | Standard fallbacks, in that order. The base endpoint gets `/v1/metrics` appended. |
| `XAGENT_OTEL_EXPORT_INTERVAL_MILLISECONDS` (fallback `OTEL_METRIC_EXPORT_INTERVAL`) | Export interval, default 10000. |
| `XAGENT_OTEL_SERVICE_NAME` (fallback `OTEL_SERVICE_NAME`) | `service.name` resource attribute, default `xagent`. |

Set the same variables for every process that should report: the web
process and the worker processes (`XAGENT_TASK_EXECUTION_ROLE=worker`, or the
workers started by the `combined` launcher, which inherit the environment).

Both the web application and the worker entrypoint (`xagent.web.worker`)
start the exporter. In a split deployment the workers record the
execution-side counters (trace writes, write retries), so configure export on
them, not only on web processes. Check the startup log of each process for
`Runtime performance telemetry OTLP export is enabled`; a process that does
not print it is not exporting.

### Per-minute `runtime diagnostics` log line

Every app process (`combined` or `web` role) and every worker logs one INFO
line per 60 s window, with no switch:

```
runtime diagnostics role=worker pid=4242 window_s=60.0 inflight=1 resume_inflight=0 started_1m=2 finished_1m=1 ok=1 failed=0 cancelled=0 gate_held_s=0.0 slot_tenants=1 slot_top=1 loop_lag_max_ms=12.3 loop_lag_samples=0 degradations=-
```

A pid that stops appearing for two windows is gone or has a blocked event
loop. `failed` counts runs that raised or whose runner handled a setup or run
exception; it describes how the execution ended in that process, not the
task's stored status. Field definitions are in
`xagent/web/services/process_diagnostics.py`.

### `/health` degradations

`GET /health` stays `200 {"status": "ok"}` while degraded and adds a sorted
`degradations` list of signal names. It is unauthenticated, so names only; the
detail strings (which can name a task) are in the logs and in the admin
endpoint below. The registry is per process: `/health` on the web process says
nothing about the workers. A worker's active degradations appear, names only,
at the end of its per-minute `runtime diagnostics` line (`degradations=`, `-`
when none).

## 2. Existing signals by problem class

Metric names below are OpenTelemetry instrument names. Log messages are the
literal format strings.

### (1) V2 event commit failures

- `xagent.execution_event.commit.failures`, attributes `site`
  (`trace_fact`, `outbound`, `input_accept`, `terminal_failure`), `error.type`
  (the first cause that is not the persistence wrapper, e.g.
  `OperationalError`, `ExecutionEventConflict`,
  `ToolAttemptAlreadyStartedError`, `TaskLeaseLostError`) and
  `storage="v2"`. One count per failed write, after retries are used up or
  for a failure that is not retryable.
- With it, one ERROR line per failed write:
  `execution_event_commit_failed task_id=... run_id=... storage=v2
  component=execution-events site=... error_type=... sqlstate=... retried=...`,
  plus `tool_attempt_id=...` for tool events. It carries no exception text;
  this is the only ERROR a failed V2 write logs.
- `xagent.execution_event.write.retries`, attributes `site` and `outcome`:
  `retried` (an attempt was rerun), `recovered` (a rerun succeeded) or
  `failed` (a write that was retried at least once ended in failure). Each
  retryable trace-fact attempt also logs WARNING `Trace event commit attempt failed for
  task %s: %s`.
- `xagent.trace.database.writes`, attributes `outcome` (`succeeded`,
  `retryable_failed`, `failed`) and `storage` (`v1`/`v2`).
- After a failed write the run is aborted and settled by the existing failure
  path.

### (2) Unknown outcomes

- `xagent.execution_event.unknown_outcomes`, attributes `operation` (the
  kind) and `storage` (`v1`, `v2`, or `unknown` where the storage version is
  not in scope). One count per occurrence, with one line
  `execution_event_outcome_unknown task_id=... run_id=... storage=...
  component=execution-events kind=... ref=...`. Kinds:
  - `tool_effect`: a run was settled as an unknown tool effect; `ref` is the
    tool attempt id. Reported once, after the settlement committed (a
    settlement that rolls back reports nothing; lease recovery reports it
    later).
  - `turn_commit`, `command_commit`: a turn or command acceptance whose
    COMMIT outcome could not be reconciled; `ref` is the turn id.
  - `finalize_ack_unresolved`: a finalize COMMIT lost its acknowledgement
    (dropped connection) and the read-back could not decide.
  - `resume_finalize_unreconciled`: the same for a resumed run's finalize,
    which has no read-back yet.
  - `settlement_deferred`: settlement could not decide from the checkpoint
    and left the lease to TTL recovery.
  - `finalize_ack_reconciled` (INFO): a lost acknowledgement the read-back
    proved applied. A confirmed success counted for visibility; exclude it
    from alerts on this counter.
- WARNING `Settlement unknown-effect classification unavailable: task_id=...`
  and WARNING `Event recovery unavailable: task_id=...`: the classification
  read itself failed; the degradation `checkpoint_recovery_unavailable` is set.
- When lease recovery fails a task for this reason, the task's `error_message`
  reads `Task execution lease expired with an unknown tool effect; automatic
  replay is unsafe.` (or `... stopped with an unknown tool effect ...` when the
  run settled itself). The admin endpoint reports only `has_error_message`.
- A start fact exists for an attempt id that is started again:
  `ToolAttemptAlreadyStartedError` (`Tool attempt already started; result
  reconciliation required`) is raised rather than re-executing the tool.
- Use the `open_tool_attempts` section of the admin endpoint to see every
  attempt in a task's log that started without a committed outcome.

### (3) Reconnect duplicates and gaps

- `xagent.websocket.trace.events`, attribute `outcome`: `broadcast`,
  `dropped` (no frame produced for the event), `no_audience` (no client
  attached, nothing sent), `integrity_gap`.
- `integrity_gap` means the live display could not be built from the stored
  fact. It comes with WARNING `Live display integrity gap task_id=%s
  event_id=%s` (with traceback) and a client-safe error event to the task's
  clients.
- There is no counter for duplicate delivery or sequence gaps on client
  reconnect. A client attaching later replays history from storage.

### (4) Recovery refusals

- `xagent.execution_event.recovery.refusals`, attributes `operation` (the
  reader: `owner_check`, `checkpoint`, `tool_outcome`), `outcome` and
  `storage="v2"`, with one line `execution_event_recovery_refused task_id=...
  run_id=... storage=v2 component=execution-events reader=... outcome=...`
  (plus `tool_attempt_id=...` for tool outcomes). Levels by outcome:
  - INFO: `lease_mismatch`, `task_ended`, `task_missing`. Normal fencing:
    another owner took over or the task already finished.
  - WARNING: `unknown_tool_effect`, `unavailable`, `page_limit`.
  - ERROR: `corrupt`.
- It counts refused reads, including the classification probes made by the
  lease-recovery sweep, settlement and auto-recovery decisions. A stuck task
  can count once per sweep; it is neither a per-task nor a per-resume count.
- Not reported yet: refusals raised while choosing the checkpoint partition on
  the root read path (lease or run mismatch, `active_run`,
  `superseded_legacy`, a missing task without a lease, stale widening, a
  failed run-tagged probe, session checkout), and a corrupt checkpoint found
  by the setup-time history read. The degradations and resume logs below
  still cover most of them.
- Degradations (names on `/health`, details in the admin endpoint on the
  process that set them): `checkpoint_load_unavailable` (an execution-event
  checkpoint read failed; cleared by the next successful read),
  `checkpoint_recovery_unavailable` (set when a lease-recovery or settlement
  classification read fails; cleared once per recovery sweep and set again by
  any candidate that is still unavailable). Related, less common:
  `checkpoint_decode_fallback`, `checkpoint_legacy_pointer_ambiguous`,
  `checkpoint_pk_anchor_dangling`, `checkpoint_prune_failed`.
- Resume refusals: ERROR `task_id=%s component=resume checkpoint could not be
  read; restoring prior status %s: %s` (the task goes back to PAUSED or
  WAITING_FOR_USER instead of FAILED). The v1 reply endpoint answers 409
  (`TASK_BUSY`, `INTERACTION_NOT_RESUMABLE`) or 503
  (`TEMPORARILY_UNAVAILABLE`); the A2A endpoint returns a matching error.
- Automatic recovery decisions are stored per task: the `auto_recovery`
  section of the admin endpoint shows the `task_auto_recovery` row (`state`,
  `reason`, `state_detail` such as `ineligible:...`) and the latest
  `task_recovery_events`.
- `task_auto_resume_unavailable` is set after three consecutive failed ticks of
  the auto-resume sweeper in that process.

### (5) Read budget

No signals exist yet. Nothing in the metrics, logs or admin endpoint reports
event-read budget consumption or exhaustion. Further signals will be added.

## 3. Triaging one task with the admin endpoint

```
GET /api/admin/execution-events/tasks/{task_id}
```

Admin users only: 401 without a valid token, 403 `Admin access required` for
a non-admin, 404 for an unknown task, 503 when the database cannot be read
(the response never substitutes zeros). It is read-only and never returns
event payloads, the task's `error_message`, the auto-recovery `last_error` or
the recovery events' `detail`; for the two error columns it reports only
whether they are set. The one free-text field left is `degradations`, whose
details are operator messages written by the platform and can name a task
id. All queries are filtered by task id and use aggregates over columns;
every list is bounded and has a `truncated` flag.

The event sections are a snapshot at `task.conversation_event_sequence`, read
first: an append advances it in the same transaction, so every event at or
below it is committed, and events appended while the request runs are left
out. `auto_recovery` is read from its own tables afterwards and can be a
little newer. A busy task therefore looks consistent within one response but
moves between two requests.

| Section | What it tells you |
| --- | --- |
| `storage` | `v2` or `v1`. A V1 task returns the same shape with empty event sections. |
| `task` | `status`, `control_state`, `run_id`, `state_version`; `conversation_event_sequence` (the committed sequence horizon: readers ignore events above it); lease fields `runner_id`, `lease_attempt_id`, `lease_expires_at`, `last_heartbeat_at`. A non-null `lease_attempt_id` does not prove the run is alive; compare `lease_expires_at` and `last_heartbeat_at` with the current time. `has_error_message` says only whether the task carries an error text. |
| `events` | Total, min/max sequence, latest `occurred_at`, and counts per (`scope_id`, `kind`), at most 200 groups, all at or below the snapshot horizon. Events are never deleted and sequences are allocated without gaps, so `max_sequence` equals `task.conversation_event_sequence` and `total` equals `max_sequence`; a difference within one response means the log itself is damaged and is worth escalating. |
| `runs` | Per `run_id` (including `null`): event count, sequence range, and `settled`, true when an `execution_settled` fact exists for that run. Latest 20 by max sequence. An earlier run with `settled` false never wrote a settlement fact. |
| `recovery_states` | Per scope, the highest sequence of a `recovery_state` fact (the checkpoint). Compare with `events.max_sequence` to see how much was written after the last checkpoint. |
| `open_tool_attempts` | Attempts (per scope) with a `tool_execution_start` fact and no `tool_execution_end` or `tool_execution_failed` fact, the same test the recovery reader uses to decide that no outcome was committed. Shows attempt id, scope, run, start sequence and the kinds seen, newest first, up to 100, plus `open_count`. |
| `auto_recovery` | The task's automatic-recovery row (`has_last_error` only; the text is the run's own error, the same content as `error_message`) and the latest 20 recovery events (type, reason, attempt, timestamps; the free-form `detail` is omitted). |
| `degradations` | `active_degradations()` of the process that served the request only. |

### What an open tool attempt means

An open attempt means the tool was started and nothing durable says how it
ended. The tool's external effect (a sent message, a payment, a written file,
an API call) is unknown: it may have completed, partly completed or never
run. Do not rerun the tool or resume the task to "retry" it on the strength of
the log alone. The platform itself refuses to replay such an attempt
automatically; a run that ends this way settles as an unknown tool effect. Check
the effect in the external system first, then decide with the task owner
whether to start a new turn.

Notes on reading `open_tool_attempts`:

- It lists every attempt in the log with this shape, not only those named by
  the latest checkpoint. The recovery verdict is decided from the attempts the
  latest checkpoint still lists as pending, so an old open attempt does not by
  itself block recovery.
- A terminal fact that records an interrupted attempt (`interrupted: true` in
  its payload) is still refused by the recovery reader, but the endpoint does
  not read payloads and cannot show it; such an attempt is not listed as open.
- A later run's settlement pair (key `tool:{attempt}:settlement:...`) uses the
  same kinds, and a settlement end fact closes the attempt like any other end
  fact.
