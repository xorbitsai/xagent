# Runtime performance baseline

This is measurement tooling, not a thread-pool or worker isolation change.
`scripts/runtime_baseline.py` drives the real task-create API, opens one WebSocket
per task, sends `execute_task`, consumes events and checks persisted status after
`task_completed`. It independently probes successful password login and `/health`.
No requests are retried automatically.

## Safety and prerequisites

- Use a **dedicated test deployment and account**. Login writes a refresh token;
  using your regular account can invalidate its refresh session. Tasks persist in
  the database and may invoke tools, consume model quota, or cause external side
  effects. Use a controlled model/tool configuration with no external credentials.
- Use a separate storage root/database from daily development. This runner does
  not create accounts, configure models, alter `.env`, start/stop the backend, or
  delete tasks. Credentials are read only from `XAGENT_BENCH_USERNAME` and
  `XAGENT_BENCH_PASSWORD`; it does not load your normal `.env`.
- Run the backend and load generator in separate processes, preferably separate
  hosts for large runs. Record CPU/memory limits and monitor generator CPU too.
- Enable the merged runtime OTel metrics and configure an OTLP metrics endpoint.
  The optional Prometheus URL is a read-only query endpoint, **not** the OTLP URL.
  Restrict it to the test environment: the report captures all `xagent_*` series
  and their labels, which may include deployment identifiers.
- A test account must have capacity for the chosen concurrent tasks. Quota
  rejection is a failure, not evidence that the runtime sustains that load.

## Fixed workload

Create a JSON file with a fixed task payload. Pin all four model slots to a
controlled test model ID, or pin an agent ID and record its exact published
configuration. Do not use automatic model routing for comparisons. For example:

```json
{
  "title": "runtime-baseline",
  "description": "Reply with exactly BASELINE_OK. Do not use tools.",
  "execution_mode": "flash",
  "llm_ids": ["TEST_MODEL_ID", "TEST_MODEL_ID", "TEST_MODEL_ID", "TEST_MODEL_ID"]
}
```

This short scenario validates the harness; it does **not** reproduce the
colleague's multi-round workload. For that scenario, use a controlled local model
service that returns a fixed number of tool calls and a final answer, with fixed
response delay/payload sizes, and a harmless local tool/MCP server. Record these
settings and their source revision in the experiment notes. The runner does not
replace the model or tool implementation. Remote model variability must not be
mistaken for an application regression.

## Run

From the repository root, with the existing Python environment (`aiohttp` is
already a project dependency):

```bash
export XAGENT_BENCH_USERNAME=runtime-benchmark
# Set XAGENT_BENCH_PASSWORD through your local secret manager or a hidden prompt.
python scripts/runtime_baseline.py \
  --base-url http://127.0.0.1:8001 \
  --task-payload /absolute/path/to/task.json \
  --output /absolute/path/to/new-baseline-directory \
  --confirm-test-environment \
  --stages 0,10,50 --duration 120 --tasks-per-minute 12 \
  --probe-rate 1 --task-timeout 300 \
  --prometheus-url http://127.0.0.1:9090
```

The output directory must not already exist. For first validation use
`--stages 0,1 --duration 10 --tasks-per-minute 6`. Only after validating the
configuration, increase to e.g. `0,100,200,400,630`, 62 task arrivals/minute,
and an appropriately long duration/timeout for the controlled workload.

**Stages are concurrency caps, not assertions of achieved concurrency.** At
62 arrivals/minute and a 150-second task lifetime, steady-state occupancy is
roughly 155, not 630. Confirm actual running-task gauges. Raise arrival rate or
controlled task duration to reach higher occupancies; document the change.
The default low rate will not necessarily fill even the default caps.

Each stage schedules task arrivals for `--duration` seconds, then drains all
submitted client operations, while login/health probes continue. Probe and task
connection pools are separate. Arrivals are fixed-rate, with bounded in-flight
work; full capacity or excessive generator scheduling delay produces a `shed`
record instead of an unbounded queue or a catch-up burst. Probe limits are per
operation. Size `--probe-inflight` to cover the anticipated timeout/latency at the
chosen probe rate. Any probe shedding means the requested probe load was not met.

Task duration covers create, WebSocket connection/execution and final status
confirmation. Login/health duration includes connection and response handling.
Scheduler lag is reported separately. Task timeout covers the entire operation;
HTTP requests also have `--request-timeout` (default 60 seconds).

On the first task failure, new task arrivals stop and existing client operations
drain; later stages are skipped. If a task cannot be confirmed terminal, reports
retain its task ID. A failed/timed-out creation can have committed without
returning an ID: `unknown_creations` records this uncertainty. Inspect test-server
state before rerunning. A disconnected WebSocket or cancelled load generator is
**not** a server-side task cancellation. No automatic cleanup/deletion is done.

## Reports and comparison

- `report.json`: config, SHA-256 of the exact payload file, stage timestamps,
  per-attempt timings/outcomes, created and uncertain task IDs, summaries,
  successful task throughput over submission **plus drain**, optional raw
  Prometheus range data. No credentials, tokens, task contents, response bodies,
  or exception messages are saved by the runner.
- `report.md`: compact per-stage p95/p99 and outcome counts. `completed` means
  the runner finished, not that the service passed a performance target.
- Failed/timed-out attempts remain in latency statistics (timeouts are censored
  lower bounds). Shed attempts have no latency sample. Empty percentiles are
  unavailable, not zero. With few samples p99 is effectively the maximum; collect
  hundreds of probes per stage and repeat runs before interpreting tail changes.

Preserve a notes file alongside each run with: server Git SHA, dependency lock
hash, DB engine/version and connection limits, server process/worker count,
CPU/memory limits, test model/tool revisions, latency/round/payload parameters,
OTel export/scrape intervals, and generator host/configuration. Run an unreported
warm-up first and repeat each baseline at least three times under the same setup.
Keep cold-start measurements separate from steady-state comparisons.

Prometheus data is queried per stage at 15-second resolution after a configurable
export settling delay (`--metrics-settle`, default 30 seconds). Increase it to
cover your export/scrape intervals. Missing/error/empty results are **unavailable**,
not zero lag. Short stages may have insufficient samples. Gauges and histograms
retain their instance labels; do not average worker percentiles.
The captured range extends through `metrics_ended_at` (after drain) to retain
late exports; use the separate stage start/end timestamps for workload alignment.

Useful queries (filter `job`/`instance` to the test deployment; names below use
the standard Collector Prometheus suffix translation):

```promql
histogram_quantile(0.99, sum by (le, job, instance) (
  rate(xagent_event_loop_lag_milliseconds_bucket[2m])))
histogram_quantile(0.95, sum by (le, job, instance, operation) (
  rate(xagent_thread_pool_queue_wait_duration_milliseconds_bucket[2m])))
histogram_quantile(0.95, sum by (le, job, instance) (
  rate(xagent_trace_database_commit_duration_milliseconds_bucket[2m])))
```

Compare client login/health tails against event-loop lag, thread-pool queue wait
versus execution, trace serialization/commit, WebSocket broadcast and running
task gauges over the **same** time window. Use host/container tooling separately
for CPU/RSS/GIL profiling; the current runtime metrics are not a CPU profiler.
Only then choose the next optimization. A mock protocol-server smoke test proves
the load runner works, not the XAgent backend's capacity.

## Harness tests

```bash
PYTHONPATH=src:. pytest tests/test_runtime_baseline.py -q
```

Tests use an ephemeral localhost HTTP/WebSocket protocol fixture, not the user's
backend, database, accounts, model services, or monitoring services.
