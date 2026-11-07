# Agent long-context cache benchmark

This benchmark runs Xagent's real ReAct pattern through its native
Xinference adapter. The model audits a synthetic invoice ledger, follows a chain
of tool-generated receipts, retries an injected transient tool failure, calculates
the total with a real arithmetic tool, saves a
JSON audit, and delivers a final answer. A report passes only when the pattern
succeeds **and** the local tool has verified every invoice, receipt, and total.
Text that merely claims to have run a tool cannot satisfy those checks.

The harness uses local test tools and a new output directory. It does not launch
or reconfigure models, load a production database, or provide shell/network tools
to the agent. It consumes inference capacity on the endpoint you select. Run
cases sequentially when testing one model replica.

The pattern runs tools serially and disables the optional repeated-tool decision
calls to isolate the growing-context loop. This is a controlled core-runtime
benchmark, not a full web-service or production workload benchmark.

## Run

Use the checkout's frozen development environment. The optional benchmark also
needs `tokenizers`, which is not required by the production prefix observer.
Provide the serving checkpoint's local `tokenizer.json`; the harness never
downloads a tokenizer or model. An API key or access token can be read from a
private file so its value does not appear in the command line or reports.

```bash
PYTHONPATH=src:. uv run --frozen python -m scripts.benchmarks.agent_cache_benchmark \
  --endpoint https://xinference.example.com \
  --model your-model-uid \
  --api-key-file /secure/xinference-key \
  --tokenizer-json /models/checkpoint/tokenizer.json \
  --context-window 131072 --context-tokens 65536 --rounds 8 \
  --output /tmp/agent-cache-64k
```

For a private/local deployment, configure the existing operator-owned network
policy for that endpoint and its addresses. For example, a local SSH forward
uses `XAGENT_MODEL_PRIVATE_ENDPOINTS=http://127.0.0.1:39984` **and**
`XAGENT_MODEL_PRIVATE_CIDRS=127.0.0.1/32`. These settings are separate from model
authentication. The benchmark does not relax the deployment policy itself.

Suggested cases, each in a fresh output directory:

| Case | Additional/changed arguments | Required evidence |
| --- | --- | --- |
| 64K target | `--context-tokens 65536` | Audit file, retry, growing conversation and cache usage |
| 96K target | `--context-tokens 98304` | Same checks at the larger input |
| 120K target | `--context-tokens 122880` | Same checks with output headroom in a 128K window |
| Resume | `--resume-after 3` | Interrupt, JSON checkpoint/tool-state reload into fresh runtime and pattern, completed audit |
| Compaction | `--compact-after 3` | Actual removal/summary event, completed audit after recovering facts if needed |

`--rounds` counts successful invoice verifications, not model calls. The default
failure is at verification two (`--fail-at 0` disables it). Each tool returns the
next invoice ID and a fresh receipt, so subsequent calls depend on actual tool
results. Compaction cases also expose `read_ledger_invoice`, which reads the
saved ledger if its text is no longer in context. The harness temporarily
lowers the compaction threshold once to exercise the existing compactor; this
is a functional stress case, not a proposed production threshold.

Resume exercises checkpoint serialization and reconstruction of the runtime,
pattern, context, and local tool state. The resume point is retained separately
as `resume-checkpoint.json` while `checkpoint.json` tracks subsequent progress.
It is not a web-service process-crash or
exactly-once external-write test. The checkpoint contains the synthetic ledger;
ordinary event/call reports omit prompt text and credentials.

## Interpret results

`report.json` includes task outcome, verified invoices, retries, compaction and
resume evidence, elapsed time, and per-call serving usage. `audit.json` is the
verified output; `calls.jsonl` and `events.jsonl` are incremental diagnostic
records, retained even if the run fails. Failures exit nonzero and must remain
part of comparisons. Use an external GPU monitor to record memory alongside
the run; the portable harness does not assume a local GPU.

The context target sizes the ledger with the supplied tokenizer and reserves
4096 tokens for instructions and tools. **It is not the exact request length.**
Use each call's provider-reported `usage.prompt_tokens` for the actual length.
The explicit context window must match the launched engine limit, which can
differ from a model family's advertised maximum.

`adapter_first_payload_s` measures time from adapter entry to its first content
or tool-call chunk. It includes client setup when needed; it is not a server
prefill timer or a measurement of the first network byte. Nonstreaming
compaction calls have elapsed time but no first-payload metric. The p95 is a
descriptive nearest-rank statistic over the recorded samples, not an SLO claim.
Repeat complete runs before drawing capacity or performance conclusions.

Compare first calls and later calls separately. A first call is not necessarily
cold: existing engine cache may already contain part of its prefix. Missing
cache usage means unknown, rather than a measured zero hit rate. Prefix-cache
reuse accelerates prefill; output generation and tool time still contribute to
task latency.

## Runtime prefix observation

The runtime automatically attaches a small `prompt_prefix` observation to
`llm_call_start` and its matching end event when a tracer records LLM events.
It complements the existing prompt-composition and cached-token statistics.

The observer compares **runtime** message and schema representations within
the same execution, step, model, call kind and phase. It reports `initial`,
`append_only`, `identical`, `messages_changed`, `messages_removed`, or
`tools_changed`, with the unchanged whole-message count and first changed
message index when a baseline exists. The runtime keeps the 32 most recently
used baselines in memory and emits only the comparison statistics. A fresh
runtime starts with `initial`, including after resume.

These observations precede provider templating and context-reference
materialization. They do not measure token-prefix length, prove cache residency,
or identify the server's eviction reason. An `append_only` request can still
miss an evicted cache; a changed message may retain a reusable token prefix.
Use serving usage and timings to establish the actual effect. Non-JSON runtime
payloads produce `unavailable` without affecting inference.

Unit tests use fixtures and prove report/tool/observation behavior, not GPU
performance:

```bash
PYTHONPATH=src:. uv run --frozen pytest -q \
  tests/core/agent/test_prompt_prefix.py tests/core/agent/test_agent_cache_benchmark.py
```

## Concurrent agents and foreground traffic

The concurrent harness runs independent real ReAct audits together, with a fresh
scope at the beginning of each system prompt. Each audit checks two invoices,
retries a transient error, uses the sum tool and verifies its saved output. The
first successful invoice tool waits for a configured duration; agents alternate
between the listed short/long waits. Sessions, contexts, receipts and output
files remain separate. A fixed number of unrelated foreground requests arrive
at a fixed cadence, independently of how quickly the agents finish.

```bash
PYTHONPATH=src:. uv run --frozen python -m scripts.benchmarks.agent_cache_concurrency_benchmark \
  --endpoint https://xinference.example.com --model your-model-uid \
  --api-key-file /secure/xinference-key \
  --tokenizer-json /models/checkpoint/tokenizer.json \
  --variant lifecycle --agents 4 --max-inflight 2 \
  --context-tokens 65536 --context-window 131072 \
  --tool-delays 2 20 --probes 8 --probe-tokens 8192 \
  --probe-start 5 --probe-interval 10 --output /tmp/agent-cache-concurrent-4
```

Run two-agent and four-agent batches in fresh directories. Keep model weights,
replicas, engine sequence limits, GPU KV budgets and workload arguments identical
between these configurations:

| Variant | Agent behavior | Operator-configured server |
| --- | --- | --- |
| `baseline` | Suppress affinity/lifecycle metadata in this harness only | Native Xavier cache |
| `lifecycle` | Real runner routing scope and automatic lifecycle metadata | Same native Xavier cache |
| `lmcache` | Same client behavior as `lifecycle` | Xavier with LMCache RAM/filesystem tiers |

The variant label does **not** enable Xavier, launch models, select a storage
backend, or introduce a production feature switch. Record server configuration,
source commits and hardware telemetry independently. Optimized audits fail their
protocol check if the deployed adapter does not send the expected metadata or
acknowledge waiting/finished transitions. First agent and foreground calls must
report zero cached tokens. Foreground responses must contain `OK`.

All calls share a FIFO client admission queue with `--max-inflight` slots. Set
this deliberately, for example two slots for two replicas each configured with
one active engine sequence. This measures multiple agent sessions with bounded
inference concurrency, not unrestricted engine admission. Every call records
`admission_wait_s` separately from its model-call duration. Foreground latency
runs from its scheduled arrival through admission and response completion, so
queueing is not hidden. The workload finishes after both audits and the fixed
foreground workload finish. Task throughput uses successful audits divided by
that complete batch duration; failed tasks remain in attempted counts and make
the batch fail. Per-agent durations, raw calls, traces and saved audits remain
available even when another agent fails. A partially failed batch must not be
published as an all-success performance sample.

The optional `--gpu-telemetry-json` accepts a continuously refreshed mapping of
GPU index to rows containing `utc` (timezone-aware ISO 8601), `temperature_C`,
`sw_thermal_throttle` and `hw_thermal_throttle`. Missing/stale telemetry fails
closed. Above `--cooldown-trigger` (default 70 C) or with a thermal flag, new calls
wait and in-flight calls drain; admission resumes at `--cooldown-resume` (default
60 C). Use the same thresholds for all comparison arms and record them; colder
targets may be necessary for sustained concurrent workloads. The harness changes no GPU settings. Any run with
thermal throttling must be retained and excluded from clean timing comparisons.
Use a separate start-temperature gate and external monitor for repeatable runs.

Only shared cooldown intervals with zero in-flight requests are counted in
`cooling_idle_s`; never sum overlapping per-request admission waits. Reports keep
both actual batch wall time and a wait-subtracted experimental measurement.
Neither subtracting waits nor limiting admission establishes production capacity:
extra pauses affect cache lifetime and queueing, and can overlap tool delays.
Small batches report foreground median/max, not a production p95. Repeat batches,
retain failures/exclusions, and compare foreground delay alongside agent throughput.
