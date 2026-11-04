# Context compaction hardening follow-up

The immediate oversized-message fix intentionally remains small. A later
follow-up should turn compaction into a complete, model-aware budgeting and
recovery pipeline.

## Budgeting

- Represent the compact model's context window, input budget, output budget,
  and safety margin explicitly.
- Use the resolved compact model rather than the main model when their windows
  differ.
- Count the complete provider request, including wrappers, tool calls, message
  content, context references, and output allowance.
- Use the resolved model's tokenizer where available; keep the conservative
  generic tokenizer only as a fallback.
- Verify both the summary request and the post-compaction agent request.

## Recoverability

- Record explicit recovery metadata for tool observations, including whether
  the operation is read-only and the durable arguments or handles needed to
  repeat the read.
- Do not infer recoverability from the message role or tool name alone.
- Preserve file paths, URLs, resource identifiers, artifact handles, and other
  durable locators separately from large raw payloads.
- Never re-run writes, sends, executions, or other state-changing operations
  to recover a dropped observation.

## Oversized unrecoverable messages

- Protect user, system, and assistant messages by default.
- Summarize natural language in bounded chunks along paragraph or semantic
  boundaries.
- Split JSON, code, and tables only at structural boundaries.
- Merge chunk summaries hierarchically while retaining source message IDs for
  auditability.
- Report an explicit blocked state when safe compaction is impossible.

## Result and fallback semantics

- Track which messages a summary covers and which original messages remain
  protected.
- Build the next context from the summary, protected originals, durable
  recovery references, and the latest user request.
- Ensure protected messages do not trigger the same ineffective compaction on
  every turn.
- Allow destructive truncation only for content already represented by a
  summary or proven recoverable.
- Distinguish input overflow, output rejection, unusable summaries, tokenizer
  uncertainty, and unrecoverable content in trace metadata.

## Test matrix

- Every message role and mixtures of recoverable and unrecoverable content.
- One oversized message and cumulative overflow from individually small ones.
- Large tool calls, context references, multilingual text, emoji, code, JSON,
  and tables.
- Different main-model and compact-model windows.
- Multiple protected messages, repeated compaction, checkpoint, and resume.
- Provider input/output rejection and partial chunk-summary failure.
- Invariants that successful compaction preserves user constraints, never
  silently deletes unrecoverable data, fits the target window, and actually
  shrinks the context.
