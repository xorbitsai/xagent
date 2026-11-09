# Benchmarks

Optional benchmark launchers and their workload, observation and reporting helpers
live in this package. Run modules from the repository root with `PYTHONPATH=src:.`.

| Entry point | Purpose |
| --- | --- |
| `scripts.benchmarks.agent_cache_benchmark` | One real ReAct invoice audit, cache usage, resume and compaction |
| `scripts.benchmarks.agent_cache_concurrency_benchmark` | Concurrent audits and fixed-arrival foreground requests |
| `scripts.benchmarks.shared_worker_benchmark` | Dedicated PostgreSQL/shared-worker deployment and SDK workload |

See [the agent cache guide](../../docs/agent-cache-benchmark.md) for cache benchmark
commands, controls and interpretation. The shared-worker launcher's module
help describes its separate test deployment prerequisites and retained writes:

```bash
PYTHONPATH=src:. uv run --frozen python -m scripts.benchmarks.shared_worker_benchmark --help
```

Shared-worker workload, observation and report modules are launcher helpers.
Benchmarks consume the inference/deployment resources selected by the operator;
unit tests use fixtures and do not establish serving capacity.
