# Task execution boundary

The first stage of #2306 separates task execution from API route modules while
preserving the current in-process scheduler, lease ownership and response fields.
It does not introduce a runner role or a cross-process task-start protocol.

## Module ownership

- `web/services/agent_service_manager.py` owns task-scoped AgentService creation,
  caching, reconstruction, tools, sandbox attachment and execution.
- `web/services/task_execution.py` owns background execution and resume,
  finalization, output persistence, and process-local background task handles.
- `web/services/task_orchestrator.py` retains turn claims, scheduling and lease
  lifecycle management. It calls execution services directly.
- `web/services/trace_handlers.py`, `task_event_trace_handler.py` and
  `public_trace_events.py` own trace persistence and public event projection.
- HTTP, A2A and WebSocket adapters call these services. The services and the
  persisted-task tracer can load without importing `web.api` routes.

## Event and acknowledgement delivery

Execution publishes task events through `task_events.publish_task_event`.
The WebSocket host registers a sink which delegates to its connection manager.
The existing connection manager still attaches current control-state fields,
serializes messages and handles disconnected sockets. Publisher errors propagate
to the existing execution error handling. Without a sink, live events are not
forwarded; persistence and task execution still proceed. Every such event increments
`xagent.task_events.dropped` with `outcome=no_sink`. The first event without a sink
also logs a warning; repeated warnings are suppressed until a sink is registered.
This distinguishes missing host delivery from a registered host with no connected
clients, which the connection manager records as `outcome=empty`. WebSocket sink
registration remains at module import and is covered by a fresh-process test.

Resume accepts an optional delivery callback instead of a WebSocket and client
message ID. The WebSocket adapter binds these using `make_delivery_notifier`.
The callback reports the same accepted/rejected outcome at the same point as
before, after the corresponding durable delivery work. A2A and REST do not need
a socket callback.

## Invariants retained in this stage

- Turn claims still commit the exact run and runner lease before local scheduling.
- Heartbeat startup, cancellation draining, fenced settlement and TTL recovery
  retain their existing ordering.
- Background task handles, AgentService instances and heartbeat objects remain
  process-local. Resume can still receive already-acquired resources.
- The command dispatcher retains its existing WebSocket control adapters and
  current PAUSE/RESUME/CANCEL/MESSAGE protocol.
- Agent/tool configuration retains its existing optional request context. This
  stage changes ownership and delivery dependencies, not configuration semantics.

## Following stage

The process split needs a durable START contract, atomic acceptance plus enqueue,
runner-side lease acquisition, effect receipts and defined queued response fields.
It must replace process-local resume handoffs with persisted inputs and checkpoint
reconstruction. Command consumption and runtime initialization then move behind
process roles. Live events need cross-process transport (#1413); the local sink
alone does not provide it. Request-derived runtime values must be captured as data
before crossing that boundary.

Regression coverage includes the existing start/resume, ownership, cancellation,
output and trace tests, plus a subprocess test rejecting API route imports while
loading execution services and constructing the persisted-task tracer.

## Dormant START protocol

`services/task_start_protocol.py` defines version 1 of the durable input for
CREATE and APPEND. This is a protocol-only step: no API produces START, no
runner consumes it, and existing acceptance/scheduling remains unchanged.
`TaskCommandKind.START` reuses `task_execution_commands.kind` (a string column),
so this step needs no schema migration. The current dispatcher excludes START,
including targeted immediate dispatch. An unfinished START still blocks later
commands for the same task; it does not block unrelated tasks. Do not enable a
producer before a consumer is implemented.

The command envelope retains task/actor identity, immutable owner subjects,
target run/version and the existing `(task_id, command_id)` unique identity.
The START command ID is the turn ID, also used by the persisted user message.
Its versioned JSON payload contains:

- accepted `run_id`, `state_version`, `turn_id` and CREATE/APPEND kind;
- transcript `message` and optional separate `execution_message`;
- authorized `file_ids`, optional `before_message_id`, timezone and
  `force_fresh` (APPEND only).

Only these fields are accepted. In particular, version 1 does not encode
arbitrary execution context, trigger metadata, actor authorization policy,
connector secrets, live runtime objects or leases. Producers requiring these
inputs must not silently drop them or use this version until their explicit
handoff contract is implemented. File IDs are references, not a guarantee that
the runner can access the file bytes. Stored task/Agent configuration and file
metadata are loaded at execution time; a serialized Agent or setup snapshot is
not part of START. Resume retains its separate, currently local contract.

A future producer must perform its existing authorization and business-state
CAS, reserve the accepted run, persist the transcript, bind files, and stage
START in **one transaction**, without acquiring an execution lease. It must
preserve applicable Workforce projections in that transaction as well.
`stage_task_start_command` verifies the exact accepted RUNNING run/version and
absence of lease metadata, then delegates to `stage_task_command` as the final
write. It does not itself accept a turn or commit. The caller must roll back
all acceptance writes on failure or a conflicting payload. On SQLite the
caller's acceptance write owns the writer lock; on PostgreSQL the acceptance
CAS already holds the task row lock, and staging explicitly requests it again. Calling staging in a later transaction is not this contract.

`read_task_start_command` strictly decodes the JSON and checks its run and turn
against the immutable command envelope. It does not authorize execution or
check the current task state. The eventual consumer must acquire the exact run's
lease and start its heartbeat before executing, and finish START processing
after the local scheduling handoff rather than waiting for the whole run.
The accepted public run identity is retained; adding a new client-visible queue
status is not required by this protocol. RUNNING continues to mean an accepted,
active turn, including the interval before a runner claims it; lease ownership
distinguishes execution admission. This preserves the existing client-visible
status contract. A distinct QUEUED status would require coordinated changes to
status storage, API/client handling, acceptance and the staging precondition,
and the transition performed when the runner acquires its lease.

This step adds no effect receipts and no safe replay guarantee for a START whose
execution outcome is unknown. The existing generic retry behavior must not be
assumed safe for START when wiring the future consumer. Process roles, producer
migration, consumer admission, cross-process events and credentials remain
subsequent work.
