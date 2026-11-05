# Xinference cache lifecycle

Xagent automatically uses execution-scoped cache hints when a Xinference chat
model advertises `supports_agent_cache_lifecycle: true`. This capability is
available for supported vLLM Xavier hybrid GPU-prefix deployments. Older servers
and other providers continue without lifecycle hints. There is no user switch.

Each execution gets its own cache session. Ordinary and streaming calls carry
the session metadata through the existing Xinference SDK. The runtime reports
`waiting` while tools run without an active LLM call, then `active` when work
resumes. Parallel tools keep the waiting state until the last tool finishes.
Pattern completion, errors, and runner cleanup report `finished`.

Updates use the existing guarded model HTTP session and credentials, with a
500 ms timeout. A missing or unavailable cache service does not fail a tool or
retry a model call. Increasing revisions let Xinference ignore delayed updates.

The initial server policy changes the eviction order of free resident KV blocks.
It preserves waiting prefixes where capacity permits and makes finished
prefixes easier to reclaim. Blocks remain allocatable. This does not promise
additional GPU capacity or remove the need for recomputation under pressure.
Cache hints do not replace `cache_salt` isolation or session-affinity routing.
