# Isolated SQL worker

Production deployments can move SQL tool execution out of the Agent process by
setting `XAGENT_SQL_WORKER_URL`. The Agent keeps ownership of stored connection
credentials, validates each destination against the platform database, and
sends a single query to the authenticated internal worker. Open-source local
installations retain the existing in-process fallback when the URL is unset.

Run the worker with:

```sh
XAGENT_SQL_WORKER_TOKEN="$(openssl rand -hex 32)" \
  python -m xagent.sql_worker
```

The Agent and worker must use the same `XAGENT_SQL_WORKER_TOKEN`. Do not expose
the worker port publicly.

## Required isolation boundary

Run the worker in a separate container or equivalent sandbox. Its environment
must not contain the platform `DATABASE_URL`, Redis/Celery URLs, encryption or
JWT keys, cloud credentials, or a Docker socket. The worker refuses to start if
common platform-secret variables are present. It needs outbound network access
only so it can reach databases explicitly configured by a user.

Use a read-only root filesystem, a small writable `/tmp`, no Linux capabilities,
`no-new-privileges`, and CPU, memory, PID, query-time, row-count, query-size, and
result-size limits. Each query runs in a disposable subprocess; the worker kills
that process at the hard timeout so a database driver cannot keep running in a
background thread.

`XAGENT_SQL_WORKER_QUERY_TIMEOUT_SECONDS` defaults to 30 seconds,
`XAGENT_SQL_WORKER_MAX_ROWS` to 10,000, `XAGENT_SQL_WORKER_MAX_RESULT_BYTES` to
8 MiB, `XAGENT_SQL_WORKER_MAX_QUERY_BYTES` to 1 MiB, and
`XAGENT_SQL_WORKER_MAX_CONCURRENCY` to 4.

Write statements remain supported. A timeout can race with a remote database
commit, so the timeout response explicitly reports an unknown completion state
and callers must not retry it automatically.
