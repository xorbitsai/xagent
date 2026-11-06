# Xinference cache lifecycle

Xagent automatically uses execution-scoped cache hints when a Xinference chat
model advertises `supports_agent_cache_lifecycle: true`. Current server support
includes vLLM >= 0.32 Xavier recurrent-hybrid GPU-prefix deployments. A single
replica with this vLLM adapter requires the hybrid mode, `enable_xavier=True` and
`xavier_gpu_cache_bytes=0`; ordinary non-hybrid single-replica deployments do not
support it. Other Xavier adapters can participate when their server advertises
the capability. Older servers and other providers continue without lifecycle
hints. There is no user switch.

Each execution gets its own cache session. Ordinary and streaming calls carry
the session metadata through the existing Xinference SDK. The runtime reports
`waiting` while tools run without an active LLM call, then `active` when work
resumes. Parallel tools keep the waiting state until the last tool finishes.
Pattern completion, errors, and runner cleanup report `finished`.

Updates run in the background through the existing guarded HTTP session and
credentials, with a 500 ms timeout per attempt. A single sender coalesces pending
transitions, so slow cache control cannot delay or lose a completed tool result.
Increasing revisions let Xinference ignore delayed updates. While tools wait,
the sender refreshes `waiting` every 60 seconds using the same revision, before
the server's 300-second expiry.

A failed provider enters a 30-second cooldown. Failures produce one warning per
provider/model, including the HTTP status when available, without logging URLs,
credentials or request payloads. Warning deduplication keeps up to 256 providers.
`finished` gets one additional background retry for transient failures, then
releases the callbacks. Process shutdown or persistent failure can still lose a
hint; the server's expiry provides cleanup in that case. Hints never retry model
or tool execution.

The initial server policy adjusts the eviction order of free resident KV blocks.
It gives waiting prefixes a soft retention preference; finished or expired
sessions return to normal LRU eviction without deleting reusable shared pages.
Blocks remain allocatable. This does not promise additional GPU capacity or
remove the need for recomputation under pressure. Cache hints do not replace
`cache_salt` isolation or session-affinity routing.
