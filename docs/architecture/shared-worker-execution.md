# Shared task execution foundations

This change contains the acceptance/data and event/recovery foundations for shared task execution. Application startup and task execution remain on the existing path. It does not start a shared worker or the Redis event bridge.

## Acceptance and persistence

`_AcceptedTurn` represents a persisted turn without an execution lease. The existing `_claim_turn_no_commit` still acquires the lease and returns `_ClaimedTurn` inside the same transaction before scheduling. Message and file binding, Workforce projection, commit reconciliation and local scheduling keep their existing contracts. Terminal finalization now also writes `Task.output` in the same transaction as the assistant transcript on the local path. This brings forward the same content (or failure clearing) that `finish_turn` writes later, allowing shared readers to observe terminal status and durable output together.

`task_runtime_secrets` stores encrypted, single-turn connector secrets and auth selectors. Reads verify task, turn, run and the owner's stable subject. Storage, binding and cleanup operations are available to later ingress/worker integration; existing connector runtime paths do not use this store yet. No secret values are added to task APIs or event payloads. The store requires a valid, explicitly configured `ENCRYPTION_KEY` and rejects the published development key, including when copied from `example.env`. Unavailable key configuration returns `connector_runtime_unavailable` before writing and also when reading an existing, correctly scoped row. If a different valid key cannot decrypt that row, the read returns `runtime_secret_unavailable` instead; both errors are HTTP 503, and neither returns credential contents.

These inputs belong to one execution, not the lifetime of a paused conversation. After its lease is released into `paused` or `waiting_for_user`, cleanup removes them. A subsequent resume must supply fresh secret/auth-selector values through ingress; missing required values fail with `runtime_secret_unavailable`. Resuming must not silently reuse or omit an earlier execution's credentials. The integration that first enables staging must wire terminal deletion and the compensation sweep in the same change. There is no TTL or running sweep in this foundation PR.

Nullable `reply_host_id` and `reply_origin` fields identify the creating ingress and its exact socket registration. They are stored when a command is created; duplicate submissions cannot overwrite the route. The migration preserves existing commands and can be downgraded independently.

## Events and recovery

The Redis bridge provides task fanout, origin-bound private replies, socket acknowledgements, bounded deduplication and reconnect notifications. Redis remains a Pub/Sub transport, not a task queue or replay log. A private-reply timeout is delivery-unknown and must not cause command execution to retry. Replies without an origin route remain valid for non-socket ingress; they emit a warning and a `xagent.task.reply.delivery` counter with `outcome=no_route`, without logging the reply content.

State-bearing event enrichment is extracted from the WebSocket API so a future producer can capture the original run/version before transport. Each shared socket has a bounded writer queue; slow sockets cannot block the other recipients. Authorized task audiences can receive persistent state/output snapshots to reconcile missing events.

The frontend handles unavailable/resync notifications and exact-run snapshots. It marks interrupted output, avoids appending tokens to a missing prefix, and replaces the display when complete persisted output is available. Existing local output remains unchanged when these events are absent.

## Activation boundary

`XAGENT_SHARED_TASK_EXECUTION_ENABLED` defaults to **false in this intermediate change**. Application startup refuses `true`: this change does not wire the bridge into application startup and is not a deployable shared execution mode. Treat the flag as process startup configuration; changing it inside a running process is unsupported. Tests explicitly initialize the bridge to exercise its contracts. Reconciliation lifecycle tests also start and stop `ConnectionManager`'s loop explicitly; production startup must wire both the bridge and the reconciliation loop together in the activation change.

`XAGENT_TASK_EVENT_CHANNEL_PREFIX` defaults to `xagent:task-events:v1`. Redis logical DB numbers do not isolate Pub/Sub; separate deployments need different prefixes.

Follow-up changes will implement START handoff and lease fencing, resume/existing-task and ingress adapters, channel bots, and worker/web lifecycle integration. The final integration will enable shared execution by default, after those paths are complete. No intermediate PR should turn it on prematurely.

## Validation

Tests accompany acceptance/rollback, immutable origin routing, encrypted input isolation, migration upgrade/downgrade, event identity and deduplication, ACK semantics, bounded socket queues, and frontend recovery. The Redis reconnect test accepts `XAGENT_TEST_REDIS_URL`; PostgreSQL migration tests accept `XAGENT_TEST_POSTGRES_URL` and create/drop their own disposable database. Missing service URLs skip only those external-service tests.
