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
PYTHONPATH=src:. uv run --frozen python -m scripts.agent_cache_benchmark \
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

The observer compares adjacent **runtime** message/schema representations within
the same execution, step, model and call kind. It reports `initial`,
`scope_changed`, `append_only`, `identical`, `messages_changed`,
`messages_removed`, or `tools_changed`, along with the unchanged whole-message
count and first changed message index. Compaction/decision/retry calls have
separate baselines. Only content digests and scope identifiers are retained in memory; neither hashes nor text
are emitted. A fresh runtime starts with `initial`, including after resume.

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
  tests/core/agent/test_prompt_prefix.py tests/test_agent_cache_benchmark.py
```
