# Shared worker deployment and upgrade

Shared execution is enabled by default. A `combined` process serves HTTP/WebSocket
requests and executes accepted tasks; a `web` process only accepts requests; a
standalone `worker` executes them. Celery workers handle background jobs and are
not substitutes for the standalone Agent worker.

## Combined process launcher

Set `XAGENT_TASK_EXECUTION_ROLE=combined` and `XAGENT_WORKER_COUNT=4`, then run
`python -m xagent.web` (or `xagent-web`) to launch one web process and four
standalone Agent worker processes. The Docker backend uses this same entrypoint.
Shared execution must be enabled. Redis, the database, encryption key and file
storage configuration are inherited by every child process.

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

## Configuration and upgrade

1. Stop old application processes before upgrading the database. Back up the
   database and retain the existing private encryption key. Do not run old local
   executors alongside shared executors against the same database.
2. Configure the same `DATABASE_URL`, `XAGENT_REDIS_URL`, private `ENCRYPTION_KEY`,
   and `XAGENT_TASK_EVENT_CHANNEL_PREFIX` on web and execution hosts. Generate a
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

For a deliberate single-process local deployment, set
`XAGENT_SHARED_TASK_EXECUTION_ENABLED=false` and use the `combined` role.
Shared execution requirements do not apply in that mode.

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

## Runtime credential lifetime

Run-scoped credentials remain encrypted while a task is paused or waiting for
user input. `XAGENT_TASK_RUNTIME_SECRETS_TTL_SECONDS` is a positive integer,
defaulting to 86400 (24 hours), measured from acceptance. Resume does not extend
this lifetime. Reads reject expired credentials immediately, even before the
cleanup sweep deletes them. Finished and replaced runs are also cleaned up.
After expiration, submit fresh runtime credentials with a new request; an old
run cannot silently reuse expired values.
