# Shared worker deployment and upgrade

An unconfigured `combined` process serves HTTP/WebSocket requests and executes
accepted tasks locally, so wheel installs can start without Redis. Published
backend images are preconfigured with `XAGENT_WORKER_COUNT=2` and therefore use
shared execution with Redis. Configuring `XAGENT_WORKER_COUNT`, or selecting the
`web` or `worker` role, enables shared execution for compatibility with existing
deployments. `XAGENT_SHARED_TASK_EXECUTION_ENABLED` can explicitly pin either
mode. Celery workers handle background jobs and are not substitutes for the
standalone Agent worker.

## Combined process launcher

Production web and worker hosts must also share the exact
`XAGENT_SECRET_ENCRYPTION_KEY` (or legacy `SECRET_KEY`) for tool credentials.
The image persists a separate tool key in the secrets volume. For existing
default/short keys, follow [the offline upgrade procedure](tool-credential-key-upgrade.md)
before starting upgraded replicas; runtime decryption never tries a public key.

Set `XAGENT_TASK_EXECUTION_ROLE=combined` and `XAGENT_WORKER_COUNT=4`, then run
`python -m xagent.web` (or `xagent-web`) to launch one web process and four
standalone Agent worker processes. The Docker backend uses this same entrypoint.
Shared execution must be enabled. Redis, the database, encryption key and file
storage configuration are inherited by every child process.

Open-source backend images built from the current source default to two Agent
workers. When `ENCRYPTION_KEY` is absent, the image entrypoint creates a private
Fernet key in the persistent `xagent_secrets` volume and every Xagent service
reuses it. Back up this volume with the database: deleting or replacing the key
makes encrypted runtime values unreadable. An explicit `ENCRYPTION_KEY` always
takes precedence. The checked-in Compose file uses fixed release tags, so this
default takes effect after its backend image is bumped to a release containing
this change. Set `XAGENT_WORKER_COUNT=` and
`XAGENT_SHARED_TASK_EXECUTION_ENABLED=false` together to opt that Compose
deployment back into single-process local execution.

The web process completes normal startup, including database migrations, before
workers start. It serves HTTP/WebSocket and designated channel ingress without
executing Agent tasks. Workers use the existing durable task queue and lease
routing. The count controls processes, not task concurrency or even distribution.
When sandboxing is enabled, configure a stable `XAGENT_SANDBOX_WORKER_ID` for this
launcher: the web process retains that ID and workers use `<base>-worker-1`
through `<base>-worker-N`. Use distinct base IDs for concurrent launchers sharing
one sandbox namespace, and keep them unchanged across restarts.

SIGINT or SIGTERM stops the whole group and gives children 30 seconds to shut
down before terminating remaining processes. An unexpected child exit stops the
other children and fails the launcher; restart the launcher to recover the group.
An unset or empty count retains the existing single-process combined behavior.
The count must be a positive integer and cannot be combined with `--reload`,
`web`/`worker` roles, or disabled shared execution. Direct ASGI server invocations
do not run this CLI launcher; use the existing separate web/worker deployment
when managing processes externally.

## Runtime diagnostics line

Every app process (`combined` or `web` role) and every worker logs one INFO line
per minute, including minutes with no work:

```
runtime diagnostics role=worker pid=4242 window_s=60.0 inflight=1 resume_inflight=0 started_1m=2 finished_1m=1 ok=1 failed=0 cancelled=0 gate_held_s=0.0 slot_tenants=1 slot_top=1 loop_lag_max_ms=12.3 loop_lag_samples=0 degradations=- fair_misses=0 fair_fallbacks=0 fair_all_missed=0
```

Read the lines per `pid`. A pid that keeps logging `inflight=0` is alive and
idle. If a pid that was logging is missing for two consecutive minutes, that
process has exited or its event loop is blocked; with the launcher above, an
unexpected child exit also stops the other children, so check whether the whole
group stopped. A `window_s` well above 60 means the event loop was blocked when
the line was due. In shared mode the `web` process only accepts work, so its
line shows liveness and `loop_lag_*` only, with `inflight=0`. `inflight` counts
unfinished executions in that process and differs from the
`xagent.agent_tasks.running` gauge, which also counts finished executions that
are still registered. `gate_held_s` is the number of seconds in the minute that the
process was at `XAGENT_DISPATCHER_MAX_INFLIGHT_PER_PROCESS`. While a process is at
the cap it does not claim commands that would start a new execution there. A process that is full with
nothing queued also reports it, so read it together with `inflight` and
`started_1m`. `gate_held_s` counts only the time at the full limit: with
`XAGENT_DISPATCHER_INTERACTIVE_RESERVE_PER_PROCESS` above 0, batch commands are
held from `limit - reserve` active tasks, and that time is not counted.
`slot_tenants` is how many tenant keys the tasks hold that this process started
through its dispatcher and that are still active, and `slot_top` is how many of
those tasks the largest tenant holds. Executions started outside the dispatcher
are not counted. Both values are read when the line is logged, like `inflight`,
not summed over the minute. They are 0 in a process that does not dispatch task
commands, and without shared execution, where dispatch takes no slot
reservation. `degradations` lists the names of the process's active degraded
signals, sorted and comma-separated, or `-` when there are none. `/health`
reports them only for a web process; a worker has no HTTP endpoint, so this
line is where a worker's degradations appear. `fair_misses` is how many times
in the minute a query for the commands of one tenant key and dispatch class found
nothing this process could claim; that pair is then skipped for two seconds,
unless its tenant key was the only one ranked, in which case every scan asks it
again and each miss counts. It counts lookups, not tenants: dispatch attempts
woken together can each read the skip table before any of them records a miss, so
one pair can be looked up and counted once per concurrent attempt in the same
wake-up. It is 0 when `XAGENT_DISPATCHER_FAIR_SHARE` is off and in a process that
does not dispatch task commands. `fair_fallbacks` is how many dispatch attempts in
the minute were fallback rounds, which take the oldest claimable command instead of
ranking tenants; a fallback round that dispatched nothing counts too, so after a
stretch of attempts that open no slot every attempt counts until one fallback
round dispatches a command holding a slot token. `fair_all_missed` is how many
scans in the minute ranked at least two tenant keys and still fell back to the
oldest claimable command, if any, because no tenant asked had a command the
process could claim or every tenant was skipped or held back; a scan repeated
after a task is set aside counts again. Both are 0 when
`XAGENT_DISPATCHER_FAIR_SHARE` is off and in a process that does not dispatch
task commands. The line is not logged when `XAGENT_LOG_LEVEL` is
above `INFO`.

Runtime metrics (OpenTelemetry, OTLP/HTTP) are separate from this line and off
unless configured (`XAGENT_RUNTIME_TELEMETRY_ENABLED` or an OTLP metrics
endpoint; see `example.env`). Configure them on worker processes as well as web
processes: in a split deployment, execution-side metrics such as trace writes
and event write retries are recorded by the workers, where tasks run.

`XAGENT_DISPATCHER_FAIR_SHARE` is off by default, and with it off the dispatcher
selects commands exactly as before. With it on, a process that has room for a new
execution takes the oldest claimable command of the tenant that holds the fewest
of its slots, instead of the oldest claimable command of all; tenants that hold
as many slots are ordered by their oldest queued command that opens an execution.
Fairness is computed in each process from the slots of that process alone. All
processes read one queue, so on a tie every process prefers the same tenant, the
one with the older oldest queued command. Once that tenant holds one more slot in
a process, a ranking that reads the updated slot counts breaks the tie there and
the other tenant goes first. Across a deployment of N dispatching processes, the
preferred tenant therefore tends to lead a tenant it tied with by one slot per
process, N in all. N is not a strict bound. The dispatch attempts of one process
run concurrently, and each ranks from the slot counts it read just before its
scan, so several attempts can prefer the same tenant before any of them sees the
slot another one took. Fallback rounds, which take the oldest claimable command
whatever its tenant, can add to the lead as well.
Only executions the process started through its dispatcher count for a tenant,
as for `slot_tenants`: a resume started another way, such as one for a task that
waited for an A2A or a user reply, holds a slot of the cap but counts for no
tenant, so its tenant ranks as if it held fewer slots. Between the batch line
and the limit (`XAGENT_DISPATCHER_INTERACTIVE_RESERVE_PER_PROCESS` above 0), when
at least two tenants have queued commands, the tenant holding the most slots in
the process may not take a new one, so a reserve slot can stay empty until the
process is back under the batch line. Controls and commands of tasks the process
already runs still pass, but within one task a cancel or pause queued behind a
held-back start waits with it. A command dispatched by its id is not held back.
The switch accepts `1`,
`true`, `yes` or `on` and `0`, `false`, `no` or `off`, in any case; unset or
blank is off. Any other value counts as off and logs a warning in two places:
where the dispatcher or the classification of a command whose task declared a
dispatch class reads it (both use one parse per value), and where the
`dispatch policy` line parses it, which happens again whenever the dispatch
policy or classifier changes. The value
in use is logged as `fair_share=` on the process's `dispatch policy` line, at its
first dispatch snapshot and whenever the value, the policy or the classifier
changes, so that line tells whether the switch is on; `fair_misses`,
`fair_fallbacks` and `fair_all_missed` are 0 both when it is off and when it is on
with nothing to count.

`POST /v1/chat/tasks` takes an optional `dispatch_class`, `interactive` or
`batch`, which is stored with the task exactly as sent. `XAGENT_DISPATCH_ACCEPT_REQUESTED_CLASS`
is off by default, and with it off the stored value never changes the class of
any command. A command takes the task's declared class only when that setting and
`XAGENT_DISPATCHER_FAIR_SHARE` are both on. This applies to the start, to appended
messages, to manual resumes and to automatic resumes; an automatic resume takes
the declared class before the task source. Answers to a waiting task do not look
at the declaration and are always interactive, and pause and cancel are not
classified. The decision is made when each command is enqueued and is then fixed
on the command, so changing the setting affects only commands enqueued afterwards:
a task whose declaration starts being accepted gets it on its later commands, and
one whose declaration stops being accepted goes back to the source rule on its
next command. The create response reports the class the start command was given
in `dispatch_class`; it is null when the server runs the task in its own process
instead of through the dispatch queue, and it does not describe later commands.

Starts, messages and manual resumes are classified in the process that accepts
the request, and automatic resumes in the process that executes the task, so set
`XAGENT_DISPATCH_ACCEPT_REQUESTED_CLASS` and `XAGENT_DISPATCHER_FAIR_SHARE` to the
same values in every process that accepts commands and every process that executes
them. If they differ, the commands of one task can get different classes. The
setting is read each time a command whose task has a valid declaration is
classified. A stored value that is not a valid class is treated as no declaration
and logged as a warning for each such command; a task with no declaration reads
neither setting. A deployment classifier installed with `set_command_classifier`
receives the stored declaration in the task view and decides for itself whether to
adopt it.

`POST /v1/chat/tasks` also takes an optional `priority`, an integer from -10 to 10
(0 when omitted or null; higher goes first). It is stored with the task and copied
onto the start, appended messages, manual and automatic resumes and answers; pause
and cancel carry 0. A deployment classifier may return its own priority for a
command. With `XAGENT_DISPATCHER_FAIR_SHARE` off the value is stored and changes
nothing. With it on, it only orders commands of the same tenant key and dispatch
class, also when that tenant key is the only one with queued commands: of the
sixteen highest-priority queued commands of that pair, the highest priority, then
the oldest, that can be claimed is taken. When both classes of a tenant key have a
candidate, the batch candidate is compared by its id and the interactive side by the
smallest id among the claimable interactive commands of its window, so with equal
priorities the order is the same as comparing ids. An older command that needs no
new slot, such as a control command or one of a task already running in the process,
still goes first, and a class the dispatcher has set aside for a while is not asked,
so only the other class counts. Answers to waiting tasks are still taken first,
oldest first. The fallback round, the step that takes an older command needing no
new slot, and a scan with no free slot (a process at its limit or on hold) ignore
priority. Low-priority commands of a tenant key can be starved by a steady stream
of higher-priority ones from the same key. If the sixteen highest-priority queued
commands of a tenant key and class all wait for tasks leased by other processes, the
tenant keeps being missed, even when a lower-priority command of it could run, until
a fallback round happens to take it: that round takes the oldest claimable command
of the whole table, which is not necessarily one of this tenant. When every tenant
key asked misses, the same scan takes the oldest claimable command instead; a tenant
key ranked alone is not skipped after a miss, so the next scan asks it again. With
an admission hook installed, the same happens to a tenant whose command at the head
of an admission queue has lower priority than the sixteen behind it, so do not
enable admission together with priorities until admission waiters are ordered by
priority.

## Configuration and upgrade

1. Stop old application processes before upgrading the database. Back up the
   database and retain the existing private encryption key. Do not run old local
   executors alongside shared executors against the same database.
2. Set `XAGENT_SHARED_TASK_EXECUTION_ENABLED=true` (recommended even though a
   worker count or split role also selects it), then configure the same
   `DATABASE_URL`, `XAGENT_REDIS_URL`, private `ENCRYPTION_KEY`, and
   `XAGENT_TASK_EVENT_CHANNEL_PREFIX` on web and execution hosts. Generate a
   Fernet key for a new deployment with:
   `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.
   The empty example value and published development keys are rejected. Preserve
   the key on restart; replacing it makes existing encrypted values unreadable.
3. Mount the same task storage and upload storage at the same paths on all hosts.
   Set `XAGENT_STORAGE_ROOT` and `XAGENT_UPLOADS_DIR` accordingly. Redis alone does
   not share files. Use a distinct event prefix for each deployment: Redis Pub/Sub
   does not isolate channels by logical database number.
4. Start one web/combined process first to apply the application's database
   migrations, then start standalone workers using the same application version.
   The migrations include task command, runtime credential, and channel delivery
   tables. Standalone workers require the migrated schema.
5. Set `XAGENT_TASK_EXECUTION_ROLE=web` for acceptance-only hosts, `worker` for
   `python -m xagent.web.worker`, or `combined` for a host doing both. Start web
   hosts with `python -m xagent.web`.
6. For Telegram, Feishu, or Slack, set `XAGENT_CHANNEL_INGRESS_ENABLED=true` on
   exactly one designated web/combined process. All others keep it `false`.
   This is an operational designation, not automatic leader election. If it is
   disabled everywhere, no bot connection starts; startup logs state this.

Single-process local deployment is the default when no shared topology is
configured. Set
`XAGENT_SHARED_TASK_EXECUTION_ENABLED=false` explicitly if the deployment should
pin that mode, and use the `combined` role. Shared execution requirements do not
apply in local mode.

## Command ownership upgrade

Shared workers process and settle commands under the task coordinator's owner
attempt. The Registry continues renewing task ownership in batches; shared
commands no longer have an independent processing lease or heartbeat.
`retry_available_at` records a business retry delay and is never heartbeat-renewed.
Every shared agent run holds the task lease. `AgentServiceManager.execute_task`
refuses to start a run that has none: the caller must pass the lease it owns,
or let `execute_task` acquire one for a task id. Checkpoints, command results,
and task settlement are all fenced by that lease, so an unleased run would
write them unfenced.

Stop all old executors before applying `20260918_command_retry_at`. The migration
copies pending commands' retry deadlines into `retry_available_at` and preserves
processing commands, results, attempt counters, and input-delivery evidence.
Do not reset processing commands to pending or clear task leases in bulk. On
startup, task ownership follows the existing lease-recovery rules; recovered
commands keep their identity and pass through their existing application/replay
checks. A completed START or reply handoff is not replayed simply because its
execution worker died. Expired RUNNING tasks retain the existing failure/recovery
semantics rather than automatically rerunning external side effects.

All executors against one database must use the same execution mode. The local
`shared=false` mode retains command claims. Restoring an old binary after shared
workers have processed commands is not a supported in-place rollback; stop the
new processes and restore a consistent pre-upgrade backup instead.

## Sandbox ownership

Every concurrent execution process using Docker or BoxLite must have its own
stable `XAGENT_SANDBOX_WORKER_ID`, such as `worker-1` and `worker-2`. Reuse an ID
when restarting its process, and stop the old process before reusing that ID.
Do not copy one ID across replicas or derive it from a new random process ID.
The ID must match `[a-z0-9][a-z0-9_-]*`: lowercase letters, digits, dashes and
underscores, beginning with a lowercase letter or digit. Dots and uppercase
letters are rejected, so validate generated host/pod identities before using them.

Docker also requires a stable deployment `XAGENT_SANDBOX_NAMESPACE`. Container
names, ownership labels, and SQL metadata use a scope derived from both values.
BoxLite uses a separate home directory and SQL metadata scope per worker ID.
The Compose sandbox overlays supply role-specific identities for their fixed
single instances. Additional execution replicas must supply distinct identities.

A replacement worker creates its own sandbox resources; it does not adopt or
clean up another worker's resources. Workspace files remain shared. Existing
unscoped sandbox resources from local mode need operator cleanup after draining
that deployment; changing an identity does not migrate those resources.

The SQL metadata needs the same upgrade cleanup. Old `sandbox_info.name` and
`sandbox_snapshot.snapshot_id` values have no worker namespace prefix. New
stores use `<scope>::<logical-name>` keys, so normal lookups/deletes cannot reach
those old rows; they are not automatically migrated or deleted. After draining
all processes using the old resources and backing up the database, inspect:

```sql
SELECT id, sandbox_type, name, state FROM sandbox_info ORDER BY id;
SELECT id, sandbox_type, snapshot_id FROM sandbox_snapshot ORDER BY id;
```

Match the records to the retired deployment's container/snapshot inventory,
remove those physical resources, then delete only the reviewed metadata rows
by their exact primary-key IDs using your database administration tool. Retain
rows belonging to other deployments or current worker scopes. Do not infer
ownership merely from the presence of `::`: legacy logical names can contain
that separator too. Deleting metadata alone does not remove containers or
snapshot images, and adding a prefix does not transfer resource ownership.

## Channel replies and failure recovery

Accepted channel commands and final-reply destinations are committed together.
The designated ingress checks pending final replies every 30 seconds, including
after restart. A separate delivery claim prevents ordinary concurrent sends;
its 60-second lease is renewed during delivery. Platform failures retry after
30 seconds without running the Agent again. Recovery selects the earliest eligible
retry time, with new replies first, so failing destinations do not monopolize a
batch. After 10 failed delivery attempts the row becomes `failed` and automatic
retries stop; the stored task result remains available. A warning and the
`xagent.channel.delivery` counter record the terminal outcome. Waiting for an
unfinished execution does not consume this budget. Channel access is checked
again before recovery sends a stored answer. Revoked sender access or changed
ownership permanently discards the reply with a logged/counted reason. An
unavailable channel configuration is retried within the same failure budget,
allowing a temporarily disabled channel to recover after it is re-enabled.

The guarantee is at-least-once delivery while access remains authorized and the
platform accepts the reply before the retry budget is exhausted. If the platform receives a reply just
before ingress crashes or its delivery acknowledgement cannot be saved, a retry
may repeat the reply or attachments. Platform sends and database commits cannot
be made one transaction. Reusing the loading message where supported reduces
visible duplicates, but does not eliminate this failure window.

If waiting for execution reaches `XAGENT_TASK_REPLY_WAIT_TIMEOUT_SECONDS`, ingress
posts an accepted/still-processing notice and leaves the final reply pending.
A temporary database read failure retries within the same deadline. Loss of the
progress route disables progress forwarding for that turn after the first
failure; it does not repeatedly delay execution. Final replies use durable
recovery rather than the old in-memory progress route.

Stopping a newly selected task or closing it after failed acceptance leaves it
paused with its uploaded files, so it can be continued. Cleanup applies only to
the original unaccepted selection; it cannot modify a newer replacement run.

## Admission-queued SDK replies

When a host installs an execution admission policy, an SDK or A2A reply is
committed as a durable RESUME_INPUT command together with its admission ticket.
If that ticket is provably waiting for capacity when ingress checks it (the
bucket is full, an older runnable waiter precedes it, or startup pacing has not
yet released it), the V1 reply endpoint answers `202` with `status: "queued"`
immediately instead of holding the request for
`XAGENT_TASK_REPLY_WAIT_TIMEOUT_SECONDS` and returning `504 reply_outcome_unknown`. The response carries the accepted
`command_id`, the `run_id` the task is waiting on, the acceptance
`state_version`, and the task's current `control_state`; the task stays
`waiting_for_user` until a worker reserves capacity and resumes it. Nothing has been validated at that
point: checkpoint readability, state fences, and model availability are still
checked by the worker, so `queued` is not a running result. Repeat the same
`command_id` with the identical reply body to observe the outcome without
injecting the answer again (a different body under the same ID is `409
task_busy`); the replay stays `queued` while capacity is unavailable and
reports the stored outcome (`running`, or the existing reply errors) once the
worker has run it. A queued reply that a cancel or pause control stops before
it runs replays as `409 task_busy` with `retry_with_new_id`: the reply was not
applied. Resend under a new `command_id` only if the task is still
`waiting_for_user`; after a cancel or pause it is not, and a new ID receives
`409 no_pending_interaction`.

That rejected-command mapping applies on every shared-execution host, governed
or not: a reply the worker's handoff rejected before injection (`state_changed`,
`stale_owner`, `stale_claim`, `identity_changed`, `invalid_payload`) also
replays as `409 task_busy` with `retry_with_new_id`, where it previously
replayed as `504 reply_outcome_unknown`. Only failures without a recorded
rejection reason remain unknown outcomes.

Waiting consumes no attempt or defer budget. Replies that are not governed by an
admission policy, and governed replies whose worker is merely slow, keep the
existing wait and `reply_outcome_unknown` semantics. Only SDK replies receive the
queued acknowledgment: A2A ingress has no projection for a queued reply, so a
governed A2A reply keeps the existing wait and `504 reply_outcome_unknown`
(with `accepted: true` and its `commandId`) until an A2A projection is designed,
and the durable reply still runs when capacity opens.

Within a bucket the oldest waiting command runs first, including for prompt
dispatch by ID, and buckets stay independent of each other. A waiter is the
bucket head only while it is its own task's next runnable command and needs a
slot: a command behind any earlier unfinished command of its own task, or
guidance that joins its task's running execution, is not a head and never
delays other tasks' commands in that bucket. Per-task order still applies to
the task's own commands, and a head whose business-retry deadline has not
passed still delays later work in the same bucket.

## Runtime credential lifetime

Run-scoped credentials remain encrypted while a task is paused or waiting for
user input. `XAGENT_TASK_RUNTIME_SECRETS_TTL_SECONDS` is a positive integer,
defaulting to 86400 (24 hours), measured from acceptance. Resume does not extend
this lifetime. Reads reject expired credentials immediately, even before the
cleanup sweep deletes them. Finished and replaced runs are also cleaned up.
After expiration, submit fresh runtime credentials with a new request; an old
run cannot silently reuse expired values.

## A2A first-message retries

Shared A2A requests without `taskId` retain the original acceptance under the
Agent, authenticated owner's stable identity, API key identity, and `messageId`. Repeating the
same text and explicit `contextId` returns the original task's current state,
including a completed or failed state, without creating another task. Reusing
that identity with different text or explicit context returns `INVALID_ARGUMENT`.
A new message ID represents a new input even when its text is identical.
Different API keys have independent retry histories; rotating a key starts a new
history. The public key prefix identifies the key; the secret is never stored in
the receipt. If the first request omits `contextId`, a retry may include the
server-assigned context ID returned for that task. A different context still
conflicts; an explicitly supplied initial context must be repeated unchanged.

Apply `20260919_task_input_receipts` with all old application processes stopped.
The receipt, task, message and START command commit together. Existing inputs
cannot be backfilled because their original message IDs were not retained.
Existing-task continuation and paused/waiting replies keep their current
protocol; local execution with shared mode disabled is unchanged.

Receipts have no heartbeat, processing state or automatic expiry. They retain
only hashed input identity/content and task/command references, not message text
or credentials. Deleting the task or command leaves a receipt tombstone; an
identical retry returns `NOT_FOUND` rather than resurrecting work. Normal current
authentication and task ownership checks still apply to replayed requests.
Dropping this table, including migration downgrade, discards the retry history
and removes the corresponding deduplication guarantee. This is acceptance
idempotency, not an exactly-once guarantee for external tools or message sends.


### Channel input acceptance foundation

The internal `channel_input_acceptance` service supports committing physical-input
receipts, task selection, attachment bindings, the user transcript, START, and the
reply destination in one transaction. Input identity includes the owner subject,
channel, sender, provider scope, and physical message ID. An identical retry uses
the original command; changed content conflicts, and a deleted target is not recreated.

Batch lookup separates new inputs, original commands to replay, and rejected old
receipts. Newly accepted inputs in one batch share one START and use the last new
input's reply destination. Concurrent overlapping batches must look up and partition
again when acceptance reports `ChannelInputBatchChanged`. Lookup batches must be
nonempty and belong to one owner, channel, sender, source, and provider scope.
Lookup and acceptance reject mixed channel/sender/source/scope batches with
`TaskTurnError("input_batch_invalid")` before accessing the database. Lookup also
rejects an owner change while resolving a batch. Attachments
must already be durably staged; acceptance binds their metadata atomically, while
callers remain responsible for compensating unreferenced staged objects after failure.

When a commit acknowledgement is lost, the service first checks whether this
attempt's receipt and command were committed. For a batch, recovery checks only
the primary input receipt and its command identity, relying on atomic acceptance;
it does not scan all secondary receipts for competing acceptances. Confirming
that outcome does not re-authorize the already accepted request against later channel configuration.
Replaying a competing request still requires current authorization. Only receipt
primary-key conflicts trigger repartition; unrelated integrity failures propagate.
`ChannelInputBatchChanged` is a retry signal for the caller's partition loop, not a
user-facing `TaskTurnError`.

Slack, Feishu and Telegram use this foundation in shared execution mode.


### Slack input acceptance

In shared execution mode, Slack messages use durable receipts keyed by the
workspace, channel, sender and physical message timestamp. Message and mention
redeliveries reuse the original task and command, including after ingress restart.
Authorization is checked again before replay. Attachments are staged before task
selection and committed with the receipt, transcript, START and reply destination;
an unavailable attachment prevents acceptance of that input.

Slack keeps its existing progress filtering and three-second tool-status cadence.
Selected display updates share the durable delivery claim with final replies, and
the loading-message timestamp is persisted for observers and recovery. Local
execution and control commands retain their existing behavior.

### Accepted channel command observation and progress

An accepted input can construct a `SharedChannelTurn` using `as_turn()` and call
`observe()` to attach a reply route and wait for the existing command. Observation
does not create another START, transcript message, or receipt. The existing
`execute()` entry point continues to accept new work and uses the same result wait.
The caller closes the observer to release its local event route. Reply timeout
returns the accepted/pending result; final delivery can recover independently.

`DurableChannelProgress` is an explicit sender, not a trace handler. Channel
handlers filter, aggregate and rate-limit trace events before calling `send()`,
so ignored events do not acquire a delivery claim. This does not reduce worker
trace routing or Redis traffic. Final delivery remains independent of filtering.

It sends progress and final replies through the existing
delivery claim. Progress callbacks update `delivery.destination` with platform
loading-message identifiers; settlement persists them under the claim token so
later observers and final recovery reuse that destination. An active claim blocks
both kinds of send. Successful progress releases the claim without a poll delay,
allowing an ordinary final send or recovery to run immediately. Pending notices
and result polling retain their existing delay. Progress can bypass a pending
poll delay when no send has failed, but respects failed-send backoff. Progress-send failures defer further
sends without consuming the final reply retry budget; final-send failures retain
the bounded retry budget. Platform sends remain at least once
when acknowledgement or destination persistence is uncertain.

Worker progress forwarding runs outside the agent's trace dispatch wait. Each
execution allows one in-flight send and up to 128 queued events. Events are sent
in order; a full queue applies backpressure instead of dropping healthy progress.
Connection failures discard the queued backlog and retain the retry policy below.
Normal execution drains progress for up to five seconds before committing its
final result, then cancels any remaining send. Cancellation or execution errors
abort forwarding immediately. Cleanup drains the owned send in all cases.
Final results keep their independent durable delivery path.

Worker progress retries connection failures on subsequent trace events with an
exponential delay from one to thirty seconds, reset after successful delivery.
This includes missing or closed routes, dead ingress hosts and lost acknowledgements,
so replacement observers can resume progress. Already emitted progress is not
replayed. Non-connection errors still disable the progress forwarder.

Commit recovery counts an observed competing receipt before checking current
authorization. A competing batch requests repartition immediately, consistently
with receipt conflicts during flush; the next lookup must still authorize the
caller. Single-message replay also retains its authorization checks.


### Feishu and Telegram batch controls

Feishu and Telegram collect ordinary messages into short batches and continue
processing messages received while the preceding batch runs. Control commands
are handled outside that queue. Feishu supports `/start`, `/help`, `/new`,
`/stop` and `/pause`; state-changing commands authorize the sender before acting.
`/stop` clears pending input and requests a pause while retaining the current
task. `/new` saves a new-conversation selection before clearing queued input and
stopping the old turn; late preparation cannot restore the previous selection.
Messages arriving while a Feishu control command is being authorized wait until
that command finishes; they are then queued in arrival order. A command clears
only earlier pending input. If a new-conversation selection is saved but cleanup
of previous replies fails, Feishu reports that the new conversation is selected.

Telegram retains its task-switching, agent selection and voice behavior. Both
channels also signal a preparing request when a retained previous shared turn
accepts a stop; previously Telegram could miss that newer preparation.

These controls retain the existing per-user conversation scope and active-task
file formats. They do not add cross-chat isolation or durable pending-input
storage. Platform sends already in flight may complete after a control command.


### Feishu input acceptance

In shared execution mode, Feishu identifies each physical input by its configured
channel, authorized sender, chat and message ID. Retries after ingress restart
reuse the original task and START. Overlapping batches replay existing commands
and accept only new inputs; changed or deleted original inputs are reported
without preventing unrelated new inputs from being accepted. Messages from
different chats are accepted in separate batches while retaining the existing
per-user task selection.

Attachments are staged before acceptance and their metadata commits together
with receipts, transcript, START and reply destination. Attachment failure
(including a missing provider file key) prevents acceptance of all new messages
in that contiguous chat group; the user must resend the group together. Later
chat groups continue independently. Progress and final output share the durable
delivery claim and retain the saved loading-message ID across observers.

Database acceptance is authoritative. If saving the local current-task file
fails afterwards, the accepted task remains valid and the user is told so;
retrying the same physical message reuses that task. The task selection remains
in memory, but a restart can lose that unsaved selection. The file format and
per-user conversation scope are unchanged. `/new` still saves its selection
before stopping old work; failure at that earlier boundary leaves old work intact.

Shared ordinary messages predating ingress startup are checked against receipts;
old control commands remain filtered to prevent replaying `/new` or `/stop`.
Controls interrupt unaccepted preparation or request a pause if acceptance has
already committed, and `/new` suppresses the old result. Ingress shutdown detaches
observers without pausing accepted worker tasks, including acceptance that commits
while shutdown is draining. Explicit user controls still take effect during that
drain. Observation failures after acceptance are logged without reporting a failed
request; durable result recovery continues when an active channel bot is running.
Deactivating the channel suspends that delivery recovery until it is active again.
Local execution keeps
its previous behavior. Inputs still waiting in the in-memory queue are not made
durable by this change; external sends remain at least once.


### Telegram input acceptance

Shared Telegram inputs use durable receipts keyed by configured channel, sender,
chat, topic and physical message ID. Attachment fingerprints use `file_unique_id`;
downloads use `file_id`. Replays reuse the original command before downloading or
resolving speech recognition, even when the current Agent or ASR configuration
has changed. Authorization is checked again on replay.

Contiguous chat/topic groups are accepted separately. All new inputs in a group
are rejected together if an attachment cannot be downloaded or a voice message
cannot be transcribed; resend the group together. Later groups continue. Voice
messages are transcribed in message order; regular audio remains an attachment.
Files, transcript, receipt, START and reply destination commit together. A local
conversation-map save failure does not undo an accepted request.

Loading messages, progress and final output use the durable delivery claim.
Recovery can create a loading message when acceptance committed before the
initial send. `/stop` pauses work while preserving its answer; `/new`, `/switch`
and Agent selection suppress abandoned replies. Shutdown detaches observation
without cancelling accepted worker work. Local execution behavior is unchanged.

Shared startup retains Telegram's pending ordinary messages. Commands and text
stop aliases predating startup are ignored. Telegram message timestamps have
second precision, so controls in the same second as startup are conservatively
ignored too. Agent selection and pagination keyboards from previous bot starts
expire; send `/agents` for a new menu. Local startup still clears pending updates.
This does not make aiogram's in-memory queue a durable inbox: messages already
acknowledged to Telegram but not accepted into the database can still be lost
on a crash. Platform sends remain at least once.
