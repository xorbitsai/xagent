# Execution-event readers: stage 3.3-A

Status: reader prerequisites and source audit, based on main `90660939a`.
This document distinguishes existing facts from the readers still to be built.
Production task creation remains V1. The stage 3.3-B implementation is described
below; C covers model context and D covers display/Trace below. Production routing
is stage 3.4.

## Scope and compatibility

This stage inventories the actual consumers and event payloads, adds bounded
pagination, and records an event coordinate for the existing root transcript
summary watermark. It does not implement the four readers, convert existing
tasks, create an ID mapping table, or remove compatibility writes. No database
schema changes. V1, public protocol shapes, and existing `payload.data` remain
unchanged. The stable release reference is `v0.8.1`; the integration baseline
above also includes the subsequently merged stage 3.2 writers.

Events are the source for conversation/execution content. Task ownership,
permissions, file authorization, command delivery, and interaction CAS retain
their existing business tables. A V2 reader must not silently recover missing
content by reading legacy chat/Trace/checkpoint rows.

## Actual reader inventory

Paths below are relative to `src/xagent/`. Listed functions and query sites
are source evidence, not claims that a V2 branch already exists.

| Surface | Current entry/consumer | Current content source | V2 requirement / next stage |
| --- | --- | --- | --- |
| WebSocket live | `web/services/task_event_trace_handler.py`, outbound handler in `task_execution.py` | Runtime notifications after persistence; deltas are ephemeral | Share durable event conversion with replay; retain stream identity (D) |
| WebSocket reconnect | `web/api/websocket.py` history builder | Chat rows plus Trace rows; separate max-row cache keys | Fixed event horizon, canonical message ownership, safe conversion, cache keyed by event progress and relevant business state (D) |
| Model transcript | `task_setup_snapshot.load_task_setup_snapshot_sync` → `chat_history_service.load_task_transcript_window` | Chat rows, compaction Trace, legacy watermark | Event messages and positioned summaries; preserve input exclusion, attachments, budgets (C) |
| Model execution context | Same setup → `task_execution_context_service.load_task_execution_recovery_snapshot_sync` | Tool/failure/skill Trace summaries | Event-derived context; failure represented once; reuse selected-skill business loading (C) |
| Tool-pair reconstruction candidate | `task_conversation_context_service.load_task_conversation_context_sync` | Chat + tool Trace | **Not wired into the production caller**; reuse suitable rendering rules, not its whole query pipeline (C) |
| Runner resume | `core/agent/runner.py:_load_latest_checkpoint` → `TraceCheckpointStore` → `DatabaseTraceHandler.load_latest_checkpoint` | Trace checkpoint rows and referenced blobs | State event plus later committed facts; preserve run/execution/owner fences (B) |
| Setup reconstruction | `task_setup_snapshot.load_task_reconstruction_snapshot_sync` | Trace rows and DAGExecution plan | Reconstruct equivalent inputs from events; inspect adopted DAG state, not just chat (B/C) |
| Interaction display/answer | `task_interaction_read.get_pending_interaction_question`, `task_interaction_service._resolve_read_direction_anchor` | Interaction business row plus Trace anchor, legacy question fallback | Resolve V2 state event, retain answer CAS and authorization; no V2 legacy question fallback (B/D) |
| REST V1 task snapshot | `web/api/v1/tasks.py` snapshot builder | Root Trace rows and task/interaction state | Same safe event conversion and ordering (D) |
| Conversation logs | `web/api/conversation_logs.py` list/detail builders | Chat, compaction/LLM Trace, file records | Messages, counts, activity, usage and compaction notices, not just detail text (D) |
| Monitor | `web/api/monitor.py` usage/activity queries | Root Trace events | V2 event-backed aggregation without double-counting compatibility rows; V1 retained (D) |
| Workforce/delegated detail | `web/api/workforces.py` | Trace filtered by build/source | Explicit child scope and existing authorization; avoid leaking child context into root history (D) |
| Retention/cleanup | `task_retention_purge.py`, checkpoint pruning and trace-message storage | Trace/checkpoint content and references | Verify event-referenced blobs survive compatibility cleanup; business retention still applies (B/E) |

Stage E must block legacy **content** reads across these surfaces. Continuing
to use a delivery control field is not permission to load chat text through it.
The complete removal of that control dependency belongs to stage 3.5.

## Event envelope and concrete producer contracts

All events have task-wide `sequence`, `event_id`, `scope_id`, `kind`,
`payload_version`, `payload`, occurrence/creation timestamps and nullable
run/turn/batch/attempt identities. The physical row `id` is not a task cursor.
`run_id` is not universally present: runtime writers use the current lease
when available. Recovery must check the snapshot execution identity and
business run partition; NULL does not mean “the current run”.

| Kind / family | Actual producer | Payload available to readers |
| --- | --- | --- |
| `input_accepted` | `stage_chat_message_no_commit` | Flat message: user/role/content/type/interactions/attachments/turn/delivery/source ID; acceptance run retained on handoff |
| `assistant_message` | Same writer for settlement transcript | Same flat message shape; separate from outbound events |
| Outbound messages / `final_answer_start/end/error` | `_persist_agent_outbound_event` | `data`, `protocol_event_id`; occurrence identity in metadata/envelope where supplied; no durable delta facts |
| Runtime events, including tool and LLM events | `ExecutionEventTraceAdapter` → `_save_trace_event` | `data`, `step_id`, `protocol_event_id`, `event_type`, `parent_event_id`; LLM fact complete, tool credentials redacted |
| `recovery_state` | Same adapter for readable checkpoint types | Runtime envelope; `data.snapshot` contains execution ID, context, pattern state and label; checkpoint-format/version fields remain in data |
| `input_applied` | `stage_applied_inputs_no_commit` | `turn_id` plus `recovery_event_id`; emitted alongside the state that contains that input |
| `input_delivery_changed` | `stage_delivery_fact_no_commit` | `turn_id` and status; delivery is not proof of application |
| `command_accepted` | `stage_task_command` | command ID/kind/payload/actor/target state version; request acceptance, not execution success |
| `control_state_changed` | `apply_task_control_transition` | Existing control snapshot including run/state version/status |
| `execution_settled` | Settlement, cancellation, lease recovery writers | Outer task status plus result; A2A cancellation may be outer FAILED and inner `status=cancelled` |
| `interaction_requested` | `stage_interaction_request` | interaction ID, kind/protocol/origin/request/idempotency key/expiry; current answer/slot state stays in interaction business row |
| `action_end_compact` | Runtime compaction through authoritative adapter | Summary, legacy watermark, retained context refs and compaction metadata in `data`; optional transcript event coordinate added below |

No consumer should infer the meaning of every event from a generic `content`
field. For example, `execution_settled` is not another assistant chat bubble,
and a tool's `result` is not the full context snapshot.

An actual ReAct integration regression already establishes this sequence:
`recovery_state(label=after_llm, pending_tool_calls=[...])`, then tool start,
then tool result, with the same attempt and assistant-batch identities.
That proves the saved call can be associated with its result; it does **not**
prove that every ReAct/DAG crash window is already recoverable by a new reader.

## Identity and cursor decisions

| Identity | Owner / meaning | Reader rule |
| --- | --- | --- |
| `event_id` | UUID of one durable occurrence | Store identity; never regenerate on replay |
| `sequence` | Committed position within a task | Order/filter within explicit task and scope; gaps within a scope are normal |
| `protocol_event_id` | Existing runtime/outbound protocol identity | Preserve on real-time and historical conversion; distinct from event UUID |
| `turn_id` | Accepted user-turn occurrence | Link acceptance/application/current input; do not deduplicate by text |
| `assistant_message_id` | One assistant tool-call batch | Group calls before contiguous tool results |
| `tool_attempt_id` | One execution attempt | Pair start/result, reject blind re-execution of unknown effects |
| `run_id`, snapshot execution ID, `scope_id` | Different lifecycle partitions | Validate together; do not substitute one for another |
| `TaskChatMessage.id` | Legacy transcript row coordinate | Remains legacy; never cast to sequence |
| `TaskInteractionRequest.resume_event_id` | Protocol identity of recovery anchor | B can resolve `payload.protocol_event_id` with task/run/execution checks; current reader still uses Trace PK |

`before_message_id` is stored in the strict internal `TaskStartPayload` and
forwarded through start consumers and setup. The same accepted command carries
`turn_id`. For create/append/channel turns with a persisted message, C can
resolve the current acceptance by turn identity and exclude it from prior
history. Empty/non-transcript and `existing` starts retain their explicit
no-cutoff semantics. Do not reinterpret or mutate old queued command payloads.

### Frozen V2 display identity (D implements this contract)

These are decisions for **new V2 tasks**, not an in-place reinterpretation of
already exposed V1 messages. V1 keeps its exact chat IDs and replay IDs. An
existing task must never switch ID namespaces during a reconnect. Migrating
previously exposed V1 history requires a separate stage 3.6 identity contract.
A does not enable a V2 endpoint or change a response today.

| Surface | V2 identity | Stability / ownership |
| --- | --- | --- |
| Conversation-log message `id` | The owning message fact's positive integer `sequence` | Task-local, scoped by the enclosing task; remains an integer. Never a chat PK or an event-table PK. Gaps are valid. Retention must not renumber retained facts. |
| User bubble | `msg-user-turn-{turn_id}` | Live acceptance and history refer to the same accepted occurrence; repeated identical text with different turns stays distinct. The accepting fact owns the log row. |
| Non-stream assistant bubble | `execution_message_{event_id}` | The owning flat `assistant_message` or visible outbound fact; no new ID on replay. |
| Stream bubble | Persisted `data.message_id` (`final_answer_…`) | Start/delta/end/error share this ID. It is **not** `stream_id`, event UUID, protocol event ID, or the tool-batch `assistant_message_id`. |
| Trace / outbound wire event `event_id` | `payload.protocol_event_id` | Preserve the original occurrence ID across live and replay; distinct stream boundary events still have distinct wire IDs. Flat message facts without a protocol event use `execution_message_{event_id}` for their synthesized replay envelope. |

The log API still lists transcript messages rather than individual stream
boundaries. A final settlement's `assistant_message` owns its integer log ID;
a stream start, delta, completion trace, or `execution_settled` does not add
another log message. Visible question outbounds already own the fact referenced
by their chat projection; that projection is not a second message. Subsequent
question/answer occurrences are not collapsed merely because their text matches.
A standalone visible outbound without a chat projection remains a live/replay
message, not a new log row; D preserves the existing log inclusion policy.

**Final stream versus settled answer:** the engine's root completion trace
stores `data.result.stream_message_id`; the companion AI trace stores the same
link. D associates that completion with the success settlement and its terminal
assistant fact within the same non-null `run_id`, root scope and settlement
interval (after the preceding `execution_settled`, through this settlement).
The single root completion's explicit stream link is the alias for that
terminal message. Display the settled answer by updating the existing stream
bubble under that same `message_id`; the assistant fact remains the log owner.
Completion/AI traces are Trace entries, not extra bubbles. A non-stream result
has no alias and uses `execution_message_{event_id}`. Live terminal notifications
carry the exact run/state-version snapshot captured by their settlement owner;
a newer task state cannot select their transcript. Pre-lease operation errors
have no execution settlement and retain their existing business-notice path.
Other streams are distinct
attempts; never merge them by text or by being the latest stream.

This association requires a unique root completion in that interval and a
matching durable stream end. Missing/ambiguous provenance is an integrity gap,
not permission to choose a similar message or assume a NULL run is current.
D must surface that condition before claiming V2 equivalence; experimental old
V2 data with insufficient provenance cannot silently use legacy rows. New V2
activation is gated by this contract. Public integer type preservation does not
claim equality with an experimental compatibility chat row's ID.

**Interrupted streams and bounded reads:** only committed start/end/error
boundaries are recoverable; deltas are ephemeral. At horizon H, a start with no
terminal boundary can restore an empty in-progress placeholder while its run
is still owned. After ownership ends it is an interrupted attempt, never a
completed answer. An error restores a failed attempt; an end restores the full
answer even if settlement is later than H. Once the matching settlement enters
a later horizon it updates that bubble rather than appending another. No partial
text can be invented on cold replay. A warm reconnect preserves text already
received and renders it alongside the interrupted label. Mutable ownership status is read separately
from H. A stream without a matching completion link never suppresses a terminal
message. D tests live/reconnect reconciliation using these exact rules.

```text
sequence  kind                    stable identity / presentation
   10     input_accepted          turn T -> msg-user-turn-T; log id 10
   11     final_answer_start      message_id S -> one provisional bubble
          delta (not stored)      S -> update that live bubble
   12     final_answer_end        S -> full answer available on replay
   13     task_completion         explicit stream_message_id S; Trace only
   14     assistant_message       same run/settlement interval -> update S;
                                  log id 14, not another bubble
   15     execution_settled       closes interval; no bubble
```

The exact numeric IDs above are examples; UUIDs and turn IDs are never parsed
into sequence numbers. Before settlement a stream has no transcript log ID.
D's converter must use the same ownership rules for live and historical input;
this table is its acceptance contract, not a claim that conversion is installed.

## Fixed committed horizon (implemented in A)

`load_task_execution_events(..., after_sequence=A, through_sequence=H)` reads
`A < sequence <= H`, within task and scope, in sequence order. The optional
bound defaults to `None`, preserving existing live-tail behavior and page-size
validation (1–100). H=0 produces an empty history. Reaching/passing H produces
an empty page. The caller obtains a committed H once and carries it over every
page; the query does not acquire ownership or secretly recapture H.

The caller must use an appropriately fresh scalar query/transaction for H,
not a stale ORM identity-map attribute. A history query returns stored rows;
existing authorization remains at the service boundary. A session with its own
uncommitted writes must not use this function to publish “committed” history.
State/permission consistency and reconnection subscription handoff remain
consumer responsibilities. The cursor bounds append-only history, not a
snapshot of mutable task/interaction business tables or concurrent deletion.

```text
Task sequence:      1 root   2 child   3 root | 4 root (later commit)
Captured H:                               3 |
Root page 1:         1
Root page 2:                           3
Live tail after H:                                  4
```

## Root transcript summary coordinate (implemented in A)

For a new root `action_end_compact` with a nonblank summary and positive integer
legacy watermark, resolve the exact same-task chat row through
`execution_event_id` while writing. Add to the fact envelope, outside `data`:

```json
{
  "data": {"summary": "Earlier conversation", "watermark_message_id": 7},
  "transcript_watermark": {
    "scope_id": "root",
    "event_id": "the-original-message-fact-uuid",
    "sequence": 19
  }
}
```

Numbers and UUID text above are illustrative. Tests deliberately make row ID
and sequence differ. The coordinate names the **transcript prefix** that the
existing watermark describes. It is not proof that all tool/runtime facts
through sequence 19 were summarized. C must separately decide the coverage of
its model-history representation; it must not discard every earlier event.

The old `data` and legacy Trace projection remain unchanged. The optional
field is additive within payload version 1; its absence on older facts is
meaningful, not interpreted as zero. No old event is rewritten. Replay reuses
the original optional coordinate (or its original absence), even after the
chat projection is gone. Existing idempotency validation still rejects a
changed source payload.

A first write whose claimed valid watermark has no same-task root fact fails
and rolls back rather than committing an invented position. Missing/invalid
watermarks or blank/no summaries produce no coordinate. Child compactions do
not inherit a root coordinate. V1 never enters this new path.

The lookup is intentionally a **transitional writer** dependency. C must carry
an event-native coverage coordinate through model setup and compaction before
stage 3.5 removes the old projection. This patch makes current summary facts
positionable without old content at read time; it does not complete C's native
context pipeline or promise every older V2 summary is usable.

## Remaining reader obligations, not hidden writer completeness claims

| ID | Finding and disposition | Required evidence before that step completes |
| --- | --- | --- |
| B1 | State and tool result facts exist; resume still loads old checkpoint and blindly repeating an existing attempt is blocked | B must reuse committed result, stop on result unknown, and cover each supported ReAct/DAG transition with crash-window tests |
| B2 | Interaction row has protocol/execution/run anchor fields, but resolver requires Trace PK and schema still links to Trace | B replaces content lookup while retaining CAS; stage 3.5 decides removal of compatibility anchor storage |
| C1 | Existing setup snapshot has no storage-version field and still returns legacy transcript watermark | C routes from the task version and transports explicit event coverage; do not silently change integer meaning |
| C2 | Compacted-input data sufficiency is characterized below; earlier application facts and accepted-turn fingerprints survive | B/C implement event selection and reconciliation; acceptance/delivery must never be treated as application |
| D1 | The V2 message/stream/log identity contract is frozen above; live/history still use legacy conversion | D implements that contract and tests endpoint/reconnect equivalence, including fixed-H incomplete streams and final overlap |
| D2 | Full event payloads include internal snapshots/LLM data; child scope queries are separate | D retains safety gates, trace size limits, file authorization and root/child isolation |
| E1 | Compatibility data is still required by control operations/anchors | E blocks legacy content reads across four consumers; stage 3.5 separately removes remaining control-storage dependence |

## Fact sufficiency evidence completed in A

The new characterization tests use real producers and database adapters on
SQLite and PostgreSQL, then delete compatibility chat/Trace content before
examining persisted facts. They do not install a test reader as production code.

| Scenario | Producer and event-only evidence | Consequence for B/C/D |
| --- | --- | --- |
| Inject two accepted turns, then compact away the first | `AgentRunner.inject_user_message` with `TraceCheckpointStore` persists each candidate context and `input_applied` together. Each application links to the earlier recovery state's UUID; compacted state retains the ordered accepted-turn fingerprints. Restoring `ExecutionContext` from the compacted fact recognizes a retry without another write. | Scanning only the newest message list is insufficient. Use prior application facts plus accepted-turn state; do not synthesize application from acceptance. |
| Adopt a two-step DAG; first step finished, second step has a tool call | Real `DAGPattern.run` persists adopted plan/dependencies, first result, active child context/pattern state, and parent/child execution frames in `dag_after_llm`. Existing `DAGPattern.load_state` restores them without calling the planner. | Recovery needs no old DAGExecution plan or Trace body for these data. B still owns the production lookup and resume integration. |
| Crash horizons before tool start, after start, after result | Pending call's batch/attempt IDs match the later durable start and complete result. Three event prefixes distinguish not-started, result-unknown and result-available. | B must stop on unknown effects and reuse a known result; available data is not proof that the future reader already does so. |
| Same DAG inside a delegated build | Run through `ExecutionEventTraceAdapter(build_id=…)`; all facts use that scope while DAG child frames retain their execution identities. | Build scope and nested DAG frame ID are different namespaces; root queries must not absorb child facts. |
| Live unfinished/failed/completed streams and terminal settlement | Real `PatternRuntime`, outbound bridge, `TraceEventCallback` and managed finalizer preserve wire IDs, stream IDs, explicit completion link and common lease run after compatibility content deletion. Two equal user texts retain separate accepted identities. | D has the durable inputs needed for the frozen identity rules; deltas are deliberately unavailable. |

No additional producer field or database schema is needed for these characterized
paths. The accepted-turn registry is bounded by the runner's existing limit;
A does not promise unlimited fingerprint replay from that registry alone.
Append-only `input_applied` facts preserve older application evidence even after
messages or registry entries are compacted/evicted. The test covers ordinary
injection before compaction, not retroactively importing a compacted V1 snapshot
into V2. In-place conversion remains unsupported here.

This closes A's data inventory and representative sufficiency checks. It does
not certify every future recovery transition: B must still test actual cold
resume, selection/fences, concurrency and failure windows through its new reader;
C/D must test their actual consumers. Any newly demonstrated missing fact must
be fixed at its producer before activation, never by rerunning an external
effect, regenerating an adopted plan or reading legacy content as a fallback.

## Focused verification

Run on SQLite and disposable PostgreSQL:

```sh
python -m pytest tests/web/services/test_task_execution_event_store.py tests/web/services/test_execution_event_read_contract.py tests/web/services/test_task_execution_event_writer.py -q
```

Set `XAGENT_TEST_POSTGRES_URL` to a disposable local server to include PostgreSQL.
The new cases cover a later commit between pages, root/child sequence gaps,
inclusive/zero horizons, unchanged live tail, exact summary boundary,
projection-independent retry, unresolved boundary rollback, older-envelope
replay, child isolation, and unchanged V1 behavior. Existing writer cases cover
real ReAct batch persistence, recovery-state/input linkage, cancellation,
atomicity, stale attempts, and complete-fact/limited-projection behavior.

The additional cases above cover compacted-turn replay, adopted DAG/child state,
tool crash horizons, and live/settled identity data after legacy-content deletion.
Passing A's tests does not certify the not-yet-implemented four readers.


## Stage 3.3-B: event-backed recovery

For V2 tasks, `ExecutionEventTraceAdapter` reads `recovery_state` instead of
legacy Trace checkpoint content. It selects the latest state in the existing
root run/execution partition or delegated build/execution scope, bounded by the
committed task sequence. The root reader retains the existing partition-widening
policy and its recheck, including terminal unknown-tool-effect verdicts; both root
and child readers verify a bound lease again
before returning, including current terminal task status: coordinator cancellation
can retain the lease identity and can commit after the captured horizon.
A completed/failed settlement after the state prevents root
resume, including cancellations recorded as failed settlements.

Before a saved state reaches the scheduler, every pending ordinary tool attempt
is checked, including active DAG steps and Auto's selected ReAct/DAG child. Auto
state with a non-empty child but no recognized decision is rejected as corrupt;
empty pre-decision and final-answer states remain readable. A committed result is reused through
the existing ReAct result application path; no second tool invocation or tool
start/result fact is emitted. A start without a confirmed result, including an
interrupted attempt, raises `UnknownToolEffectError`. Lease-expiry recovery
settles that run as FAILED through its existing fenced transaction; it does not
leave an expired RUNNING lease for endless retries. The task error and settled
fact distinguish an unknown tool effect from a generic lease-expiry failure.
This includes pausing or
stopping V2 while a tool is in flight: cancellation does not prove that an
external side effect was rolled back, so that pending attempt cannot be resumed
automatically. The context reader may show a failure observation to the model;
that observation is not confirmation of the external outcome. V1 replay behavior
is unchanged. Manual reconciliation of unknown effects is outside this stage.
A database read failure reports unavailable (retryable); an unsupported recovery
payload reports corrupt. Unavailable lease reads report an operational degradation
for the sweep. None falls
back to legacy content. Control-message retries retain their existing occurrence
identity and outbound reconciliation path.

Interaction anchors resolve the stored protocol event ID against a root recovery
fact with the same task, run and execution. Answer authorization and interaction
CAS are unchanged. The legacy Trace FK remains identity-only compatibility
control storage until stage 3.5; it is not used to load checkpoint content.
Waiting-question fallback reads outbound facts. Lease-expiry recovery eligibility
also uses state facts. Setup reconstruction uses the adopted state as its history
presence input; the runner still owns partitioned state restoration.

The focused recovery suite exercises actual `AgentRunner.resume`, root and child
ReAct/DAG crash windows, completed/failed/waiting/unknown tool outcomes, settlement,
lease replacement during a read, interaction anchors and compacted input retries.
Deleting legacy Trace/checkpoint content does not prevent these recovery reads.
An accepted input is not applied merely because its acceptance fact exists;
application and compacted retry identity remain in the saved context.

This is a recovery boundary, not whole-task isolation certification: the model setup switch is described in C below, and UI/Trace reads await D. Compatibility writers, control fields,
production defaults, checkpoint formats, database schema and V1 remain unchanged.

## Stage 3.3-C: event-backed model context

V2 setup now reads root event facts at one captured committed horizon. The
snapshot carries the task's storage version and a separate
`conversation_event_watermark`; V1 keeps `conversation_watermark` as a chat row
ID. The background runner, manager reconstruction and compatibility hydration
entry points preserve that distinction. No schema or production-default change.

Only accepted turns with an `input_applied` fact enter prior model history.
Their position is the application sequence, so an input accepted before a
summary but applied afterward survives in the suffix. The current start's
accepted turn supplies the cutoff when available. Existing queued integer
`before_message_id` values retain their meaning through an identity-only join
on `execution_event_id`; no legacy chat content is loaded. Starts that have no
transcript cutoff keep their existing no-cutoff semantics. Root queries exclude
build scopes, recovery snapshot bodies and raw LLM request/response facts.

Ordinary tool calls pair by `tool_attempt_id` and group by
`assistant_message_id`: one assistant declaration followed by contiguous tool
observations, regardless of completion order. The recent window retains eight
calls plus the rest of the boundary batch. Results use the existing runtime
sanitization, formatting and model token-compaction pipeline instead of the old
240-character Trace preview. No tool is invoked to reconstruct history. A start
without an outcome is explicitly unknown, including its external effect; this
model observation does not relax B's refusal to resume an unknown effect.
Question outbounds supply their transcript content. Failed/cancelled settlements
supply one outcome instead of a safe failure placeholder plus another summary.
Normal waiting, paused and interrupted settlements do not add a failure outcome,
including after a successful continuation.
Selected-skill loading continues through the existing business loader.

The existing newest-first budget of 16 historical image references is shared by
message attachments, tool-result references and retained summary references.
File bytes still pass through the normal authorization/materialization boundary.
V1 transcript and Trace context behavior remains unchanged.

Native summaries persist `data.model_context_watermark` with root scope,
event UUID and sequence. This is the historical prefix installed at setup,
including its tool history, rather than a reinterpreted chat ID. Compaction
passes the coordinate through without inventing a later position. Consequently
current-run facts beyond that prefix remain a conservative replay suffix; the
coordinate does not claim the summary exclusively describes that prefix.
Child contexts do not inherit its root coverage claim, and request context
cannot overwrite it. The writer validates the coordinate against the same task
and root scope without consulting chat projections.

A's older `transcript_watermark` remains transcript-only: it cannot suppress
earlier tool facts. Summaries with no coordinate cannot replace a known prefix;
the reader replays facts without regenerating the summary. Malformed required
facts or invalid coverage fail explicitly, without a legacy-content fallback.
Display/Trace conversion is described in D below; whole-task isolation remains E work.

### Follow-up: bound model-context payload loading

The current reader pages queries and batches anchor lookups, but retains every
matching event payload before applying summary coverage and the tool window.
Thus payload bytes read and setup memory still grow with lifetime history even
when compaction covers old facts. This is a non-blocking follow-up to stage C.
Select the applicable summary and retained batch identities first, then load
only required payloads at the same fixed committed horizon. Selection must keep
inputs applied after the summary boundary even when accepted before it, tool
batches with outcomes after the boundary, and complete boundary batches. Verify
bounded payload loading against large histories as well as equivalent model
messages for these cross-boundary cases.


## Stage 3.3-D: event-backed display and Trace

New V2 tasks now use one safe event converter for live notifications and fixed-H
history. WebSocket replay, REST task steps, conversation-log messages and activity,
monitoring aggregates, and explicitly scoped delegated-agent details read execution
facts. V1 retains its existing readers and public identities; production creation
still defaults to V1. No schema change, in-place conversion, or compatibility-write
removal is part of this step.

The frozen identities above apply to both live and replayed frames. V2 envelopes
also expose `execution_sequence`, the owning fact's task-local sequence. The chat
reducer uses this coordinate to prevent an older replayed stream start/end from
replacing a newer full answer or its settled file links. Ephemeral deltas do not
carry a durable sequence and cannot append after a durable terminal frame. This
coordinate is independent of business `state_version` and transport ordering.
A stream start without an end/error restores an empty running placeholder while
its run is owned, or an explicit interrupted state after ownership ends. Completion
and companion AI traces remain timeline entries. The settlement message supplies
the final bubble, under its explicit stream alias when present. Shared-stream
reconciliation uses those same settled message facts instead of V2 `Task.output`.
It re-sends durable content to repair dropped frames, while isolating an invalid
task's display from other connected tasks. A completed interval with a stream end
requires its matching completion link; missing provenance never creates a second
answer identity.

Root views exclude child scopes. The worker inspector reads only its requested
scope after the existing task authorization checks, selects delegated occurrences
before public redaction, and retains safe failure events even when redaction
removes their source metadata. Ordinary V2 tasks expose this inspector through
`/api/chat/task/{task_id}/agent-executions/{worker_task_id}`, authorized for the
owner or an admin; workforce runs retain their existing access checks. The open
inspector refreshes an active child and stops on terminal status or close.
Business tables still own authorization, control,
interaction CAS, file records and file-byte materialization. Existing error,
checkpoint, audit-only and sensitive-tool normalization remains in the display
pipeline; recovery snapshots and raw LLM bodies are not chat content.

V2 log IDs are message sequences, counts/activity use the same message inclusion
policy, and compaction notices are ordered at their event position. Monitoring
selects V1 traces or V2 root facts once per task, excluding V2 compatibility rows.
User-scoped monitoring filters each source before the union; conversation activity
aggregates only the candidate tasks selected by the list's existing filters.
Task-level token counters remain existing business metadata. Queries page facts at
a captured horizon; total history payload loading is not claimed to be bounded
independently of lifetime history. Whole-task legacy-read isolation is stage E;
activation, removal and conversion remain stages 3.4–3.6.
