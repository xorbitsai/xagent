# Execution-event storage and writers (stages 3.1–3.2)

`task_execution_events` is the fact source for explicitly created version-two
test tasks. Production task creation still defaults to version `1`; Web,
channels and existing tasks retain legacy routing. There is no public switch
or automatic conversion of existing conversations. Production activation waits
for the event readers in stage 3.3 and the rollout in stage 3.4.

## Commit boundaries

| Boundary | Durable facts | Transaction / compatibility |
| --- | --- | --- |
| Web and channel input, live-message claim | `input_accepted` with original text, attachments and turn identity | Existing acceptance transaction; chat row derived from the event |
| Delivery transition | `input_delivery_changed` | Existing monotonic delivery update, same transaction |
| Control command / interaction | `command_accepted`, `interaction_requested`, `control_state_changed` | Existing permission checks, command identity, state-version and lease fences |
| Runner/runtime and delegated execution | Runtime events, tool start/result/error, `recovery_state` | Strict writer before observers; legacy Trace/checkpoint rows derived in the same transaction |
| First input application | `input_applied`, referring to the proving recovery event | Same transaction as the complete recovery state; acceptance alone does not imply application |
| Outbound messages and Web streams | Message or stream envelope, including protocol identity | Commit before WebSocket broadcast; protocol shape stays unchanged |
| Normal, resumed and channel settlement; orchestrator error settlement | `assistant_message`, `execution_settled` | Existing fenced result/lease transaction; rollback also rolls back transcript and events |

The runtime bridge consumes events at their producer boundary, before Trace
observers. It does not import historical Trace rows into the fact log. The
ordinary database Trace callback becomes a no-op on this path, retaining its
checkpoint read interface only for the transition. Console, WebSocket and
exporter failures cannot invalidate a committed fact. A failed fact commit
raises `ExecutionEventPersistenceError` and stops execution before broadcast.

Recovery events contain the complete existing execution snapshot: context,
messages, adopted summaries, pattern state, planning state and pending work.
Their payloads are not clipped to Trace display limits. Runtime LLM payloads
are committed before the observer-side normalization. The JSONB sanitizer
still applies at the storage boundary; unserializable facts fail rather than
becoming a successful write of a diagnostic placeholder.

## Identity and ordering

The envelope has a generated `event_id`, task-local `sequence`, explicit
`scope_id` (`root` or the stable delegated execution ID), and optional run,
turn, assistant-batch and tool-attempt identities. Append requires a producer
idempotency key unique within task and scope. Reusing it for a different fact
raises `ExecutionEventConflict`; replay preserves the first ID and timestamp.

For version-two ReAct execution, each adopted tool batch and each call attempt
receive separate IDs before the `after_llm` recovery state is committed. Those
IDs survive restoring pending calls. Provider call IDs and DAG step IDs are
not treated as globally unique attempt IDs. Tool facts retain full results.
A previously started attempt is blocked from blind execution: until event
recovery reconciles its result, an uncertain external side effect must not be
repeated. This conservative block is intentional during the writer-only stage.

Appends and pre-append attempt decisions use the same task-row UPDATE lock,
including on SQLite. This serializes sequence allocation, replay and rollback.
Delivery writers acquire the task lock before the message update to preserve
lock order. Runtime and outbound writes reject a replaced bound lease.

The temporary chat projection has a nullable, unique `execution_event_id`.
Legacy rows keep NULL; replaying one canonical message produces one compatible
chat row. Final assistant message keys include run and state-version identity.
No table independently chooses authoritative message content on the new path.

`load_task_execution_events` requires task and scope, and reads by sequence
with a page size of 1–100 (default 100). Out-of-range sizes raise `ValueError`
before querying the database. It uses the caller's transaction snapshot; another
connection cannot see uncommitted events, while read-your-writes in the same
Session is intentional. Authorization remains the future calling service's
responsibility. Neither helper is an externally exposed endpoint.

## Migration and rollback

The new migration permits storage versions `1` and `2`, keeping SQL and ORM
creation defaults at `1`. On SQLite it replaces the version column, whose old
CHECK guarantees all pre-migration values are `1`; it does not rebuild `tasks`
or trigger inbound cascade deletes. PostgreSQL replaces the CHECK. Existing
rows, attachments and public protocol fields remain unchanged.

The legacy readers still consume derived chat / Trace / checkpoint records in
this stage. Removing those readers and implementing event-based reconstruction
belongs to 3.3; version-two tasks must remain test-only until then. There is no
mixed-history fallback or legacy backfill in this change.

Code rollback for legacy production tasks can retain the additive schema.
Schema downgrade refuses when version-two tasks exist, so it cannot silently
reclassify authoritative event history as legacy data.

## Validation

SQLite and PostgreSQL tests cover migration with inbound foreign keys,
unchanged defaults, downgrade refusal, atomic acceptance/settlement, complete
recovery payloads, actual ReAct batch identity, failure-before-broadcast,
observer failures after commit, ownership fences and uncertain-attempt replay.
Existing runner, checkpoint, channel, command, interaction and Trace tests
exercise the unchanged legacy path.
