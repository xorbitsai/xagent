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
