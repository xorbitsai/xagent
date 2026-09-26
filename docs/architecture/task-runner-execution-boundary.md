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

## Uncertain input delivery

The injection checkpoint is the acceptance boundary. Reading or preparing a
message can fail before any write; such a failure is not an unknown write.
After a write starts, a failed acknowledgement requires an authoritative
read-back. If acceptance cannot be determined, the existing delivery receipt
records `outcome_unknown` and the owned execution pauses. While the process
lives there is no automatic reinjection or execution restart. Already generated
answers remain in history; a genuine execution failure retains its original
diagnostic.

A crash is different: a durable command retry whose delivery row is still
pending posts the message again under the same `turn_id`. The runner reconciles
that `turn_id` against the checkpoint it rebuilds from, so a turn that was
written replays instead of being applied twice, and one that was not written is
applied once.

That replay needs a runtime that can reconcile the `turn_id`: the command's own
run, or a run that is live in this process. When the task's run has changed
since the command targeted it and no such runtime exists, an earlier attempt
may have accepted the message as a new turn and started a run for it before
crashing. That input is at-most-once: the retry does not run it again.
It settles the command as accepted with an unknown outcome, advances the
delivery row to `dispatched` ("do not resend", not "applied"), and leaves the
task in its recovered state. The sender gets the `outcome_unknown` delivery
frame; with no reachable origin connection the notice is also published
task-wide. A same-id resend is answered from the command's stored result, so
it reports the unknown outcome instead of success. A new-turn claim that finds
the turn already in the transcript (`TaskTurnAlreadyAccepted`) settles the
same way instead of failing on the unique index.

Rows that no owner can settle any more are reconciled by lease recovery,
which never redrives the turn. Recovering an expired lease advances that
task's `pending` user rows to `dispatched` in the recovery transaction, and a
periodic sweep does the same for rows older than one lease TTL. Both require
a quiescent task: an appendable status (never PENDING or WAITING_FOR_USER), no
pause or resume request in flight, no live lease, no pending or processing
command on the task, and no failed command with the row's `turn_id`.

While the fenced run is still live, input that arrives after an unknown write
never queues behind it: the fenced context rejects it as not accepted, and the
client resends it under a new id.
The reverse order can leave a resume pending. Suppose message A is accepted and
its handoff is waiting for the current run, and message B then becomes
unknown. The finalizer keeps `resume_requested`, and A's handoff still
acquires the lease, because acquisition checks status and run, not the control
state. It resumes from the checkpoint, so A is delivered, and B is part of the
resumed context only if its write landed. B's client was already told its
outcome is unknown, and B is never reinjected.

The runtime records acceptance separately from later tracing and notification
work, so an exception after a confirmed write cannot mean “not accepted”.
Cancellation before a write and cancellation during a write have different
acceptance outcomes. The old context remains fenced against stale checkpoint
writes. Once its execution has exited, an explicit deferred input or resume
loads durable state again.

While that fence is up, a later input that would have to write into the fenced
context (a live message that interrupts the run, or any input while the old
execution is still active) writes nothing and returns `rejected_retryable`.
Its own outcome is known: it was not accepted. It is never deferred and never
schedules a resume, because either would restart the fenced run without the
user's decision. Only the original uncertain write is reported as
`outcome_unknown`.

`classify_injection` in `core.agent.runner` is the single place that turns an
attempt into an `InjectionDisposition`. It reads the recorded attempt
evidence, the returned outcome, and any escaped error or cancellation.
Recorded acceptance wins over a later error. A read-back that proves the
write absent (`UserMessageInjectionRejectedError`) is not accepted in the
same way as a fenced rejection, but it lifts the fence. Each entry point only
maps the disposition:

| Disposition | WebSocket live | Deferred | A2A / SDK | Shared command |
| --- | --- | --- | --- | --- |
| `accepted` | dispatched | dispatched, resume | scheduled | `accepted` |
| `defer` | deferred resume | fails (no checkpoint) | not resumable | `not_resumable` |
| `not_accepted_retryable` | delivery failed, resend with a new id; no task failure | delivery failed, resend with a new id; task paused if fenced, else restored | prelease restored; error carries `retryWithNewId` / `retry_with_new_id` | `busy` with `retry_with_new_id` |
| `unknown` | `outcome_unknown` | `outcome_unknown`, paused | outcome unknown | `unknown` |
| `failed_before_write` | existing error handling | existing | existing | existing |

When a cancellation or lease loss lands after acceptance, the delivery is
still recorded as dispatched and the interruption then follows its normal
handling; it is never paused as an unknown input. A shared reply that was not
accepted keeps its stored answer: repeating the same A2A `messageId` or SDK
`command_id` replays "not accepted, resend with a new id" without a second
injection.

An explicit cancel (A2A `tasks/cancel`, an external cancel, or task deletion,
all through `BackgroundTaskManager.cancel_task`) wins over the unknown-input
pause. The manager records that intent before it cancels, so a deferred
resume whose delivery is `outcome_unknown` settles as FAILED (cancelled), and
the delivery stays `outcome_unknown` and is never resendable. Other
cancellations, such as shutdown or lease loss, keep the pause.

The fence also rejects every later checkpoint of the old run, including the
ones taken after tool steps. Tool calls that complete between the uncertain
write and the stop therefore leave no durable record, and an explicit resume
reloads the earlier checkpoint and may run them again. Tools with external
side effects can repeat. This is an accepted cost of the at-most-once input
contract; a per-step intent log together with tool side-effect classification
is the intended remedy.

A reply timeout can also mean that an accepted command is still queued or being
processed. It is not evidence of a failed injection. For shared execution,
clients can repeat the same request identity to observe the existing command;
they must not automatically create a new identity to resend the input. A2A's
shared `commandId` identifies the internal deterministic command, whereas its
nonshared error correlates with the original `messageId`. Nonshared SDK replies
return a correlation ID, not a new durable deduplication guarantee. Check task
state before deciding whether to resume or send new input.

## Context cache lifetime

`ContextManager` is a process-wide cache keyed by the execution id, which stays
the same across runs and owners. It may hold a context only while a run of that
execution is active in this process, while an injection holds it, or while it
is fenced by an `outcome_unknown` write. Otherwise the checkpoint is
authoritative: another process may have extended it since this one last ran the
task. The last user of an idle context evicts it (`AgentRunner.run` on exit and
each injection on return), and the next input restores from the checkpoint. A
context restored that way belongs to no run, so a live input on it returns
`defer` and the caller takes the deferred path. A reader that started before an
eviction discards its snapshot and reads again, because the evicted context's
last write may postdate that read.

Before a completed run publishes its result, the runner writes one more
checkpoint (`run_end_tail`) when the context changed after the pattern's last
checkpoint, for example the delivered answer it appended. The write reuses that
checkpoint's pattern state, is skipped for fenced contexts and for waiting or
interrupted results, and is best effort: a failure is logged and the result
stands.
