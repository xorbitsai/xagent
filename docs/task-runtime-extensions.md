# Task runtime extensions

Task runtime extensions let an application distribution attach task-scoped
resources -- a browser target, a sandbox lease, a tenant-owned connector --
without adding provider-specific columns and branches throughout the
open-source agent runtime.

A provider is an object implementing four hooks. The registry is process-wide
and populated once at application startup. Everything a provider needs is
importable from one stable module:

```python
from xagent.task_runtime import (
    TaskRuntimeClientError,
    TaskRuntimeContext,
    TaskRuntimeContribution,
    TaskRuntimeExtensionProvider,
    register_task_extension,
    unregister_task_extension,
)
```

`xagent.task_runtime` is the supported import surface. The underlying
`xagent.core.task_runtime` and `xagent.web.services.task_runtime` modules also
export these names, but the facade is what an out-of-tree package should depend
on.

## The four hooks

| Hook | When it runs | Timeout | On failure |
| --- | --- | --- | --- |
| `on_task_created` | once, after the core `Task` row is committed | 30s | **fail-closed** -- the task is compensated and the create request fails |
| `build_runtime` | every time the task's `AgentService` is constructed or rebuilt | 10s | **fail-open** -- the contribution is logged and dropped |
| `public_metadata` | on task create (advisory) and on the metadata endpoint (authoritative) | 10s | **fail-closed on the endpoint**, degraded on create |
| `on_task_deleted` | on task delete, user delete, and create-compensation | 30s | **fail-closed** -- 503 and the task is preserved |

Every hook may be defined `async def` or plain `def`; the registry awaits the
result if it is awaitable. Hooks are always started on a worker thread, so a
synchronous hook cannot block the event loop.

A provider must expose all four as callable attributes.
`register_task_extension` raises `TypeError` at registration time if any is
missing, rather than failing later inside a request.

### A worked example

```python
from xagent.task_runtime import (
    TaskRuntimeClientError,
    TaskRuntimeContext,
    TaskRuntimeContribution,
    register_task_extension,
)


class ComputerRuntimeProvider:
    async def on_task_created(self, context: TaskRuntimeContext, configuration):
        target = configuration.get("target")
        if target not in _APPROVED_TARGETS:
            # 400 or 403 only. The detail text reaches the client verbatim.
            raise TaskRuntimeClientError(f"Unknown browser target {target!r}")

        # Open a short, operation-local session and close it here.
        session = context.session_factory()
        try:
            session.merge(
                ComputerBinding(
                    task_id=context.task_id,
                    user_id=context.user_id,
                    target=target,
                    state="active",
                )
            )
            session.commit()
        finally:
            session.close()

    async def build_runtime(self, context: TaskRuntimeContext):
        binding = _load_binding(context)  # opens and closes its own session
        if binding is None:
            return None
        return TaskRuntimeContribution(
            tools=(build_computer_tool(binding, workspace=context.workspace),),
            environment=(
                "The task has access to the browser target selected by the user."
            ),
            preferred_input_modalities=("image",),
        )

    async def public_metadata(self, context: TaskRuntimeContext):
        binding = _load_binding(context)
        if binding is None:
            return None
        # Non-secret, JSON-compatible values only.
        return {"target": binding.target, "state": binding.state}

    async def on_task_deleted(self, context: TaskRuntimeContext) -> None:
        # Must be idempotent. See "Deletion and idempotency" below.
        session = context.session_factory()
        try:
            binding = _select_binding_for_update(session, context.task_id)
            if binding is None:
                return  # already released by an earlier attempt
            binding.state = "release_requested"
            session.commit()
            await _release_remote_lease(binding)  # tolerates "already released"
            session.delete(binding)
            session.commit()
        finally:
            session.close()


register_task_extension("computer", ComputerRuntimeProvider())
```

Extension names must match `^[a-z][a-z0-9_]{0,63}$`. Registering a name twice
raises unless `replace=True` is passed.

## Creating a task

The normal task-create request carries provider configuration under
`runtime_extensions`:

```json
{
  "title": "Inspect the selected page",
  "runtime_extensions": {
    "computer": {
      "target": "approved_browser",
      "binding_id": "binding-123"
    }
  }
}
```

Shape violations -- a non-object value, more than 16 entries -- are Pydantic
422s. Registry and semantic violations -- an unregistered name, a
non-JSON-compatible configuration, a configuration over 64 KiB -- are rejected
as HTTP 400 before the task row is created and before any hook runs.

Runtime extensions are rejected outright (400) on the public widget and
shared-link task paths.

The order inside the create request is deliberate:

1. The `Task` row is built, and if `runtime_extensions` is non-empty the
   **binding record is written into it in the same transaction** (see below).
2. `db.commit()`.
3. `on_task_created` is dispatched to each requested provider, in the order the
   registry holds them.

Setup therefore runs after the core row is committed, so a provider can open its
own session and safely reference `context.task_id`.

If one provider's `on_task_created` fails, cleanup (`on_task_deleted`) is
dispatched to that provider and to every provider that already completed, in
reverse order; then the newly created core task and its file bindings are
compensated. Cleanup failures during compensation are logged, not raised -- the
original create failure stays primary.

The response status depends on what the provider raised:

- `TaskRuntimeClientError` -- its `status_code` (400 or 403) and `detail` are
  returned to the client verbatim. This is the only way a provider can talk to
  the caller.
- anything else, including `TimeoutError` -- 503 `Service unavailable`, with the
  real exception logged privately.

## The binding record

Deletion must dispatch only to providers that actually own something for this
task. That set is persisted per task in the existing `tasks.agent_config` JSON
column, under the reserved key `runtime_extension_bindings`, as a sorted list of
extension names. Reusing the JSON column -- the same convention
`execution_scope` and the A2A context id already use -- keeps the binding record
migration-free.

**The record is written optimistically, before any hook runs, in the same
transaction that creates the task.** Over-recording is safe: a provider whose
`on_task_created` never completed still gets `on_task_deleted`, and that hook is
required to be idempotent. Under-recording would silently leak provider-owned
state forever, which is why the write cannot wait for the hooks to succeed.

A missing or malformed record decodes to the empty tuple. Tasks predating this
feature bound to nothing, and a corrupt record must not make a task
undeletable.

## Runtime behavior

`build_runtime` runs whenever Xagent constructs or reconstructs the task's
`AgentService`. Providers are built independently and every successful
contribution is merged, in registry order.

- **`tools`** are appended to the tool list produced by the core registry and
  then pass through the same pipeline as core tools: category sorting, the
  task's `ToolSelectionSpec` name filter, the per-user hook override and
  allowlist filters, sandbox wrapping, and output filtering.
- **`environment`** is appended to the task's system prompt, separated by a
  blank line. It is non-secret system context describing the selected resource
  and how the agent should use it. It must not contain credentials.
- **`preferred_input_modalities`** is an advisory routing hint. See below.

### Tool requirements

A contributed tool must have a non-empty string `name`. Violating this raises
and fails the whole tool build, so validate it in your own code first.

A contributed tool must also have a usable `metadata.category`. A tool whose
`metadata` is `None` or whose `category` is `None` -- the shape you get from a
bare LangChain `@tool` function -- is dropped with a warning rather than taking
down the task's entire tool build, including its core tools.

Contributed tools carry the default `ToolCategory.OTHER`, which is never present
in a configured category set. A `BY_CATEGORIES` selection spec would therefore
drop all of them, so contributed names are passed to the spec as a task-scoped
opt-in, exactly like an `mcp:<server>` scope. Names already claimed by a core
tool are excluded from that opt-in.

### Name collisions

After policy filtering, providers are reconciled in registry order. A provider
whose surviving tool names collide -- with a surviving core tool, with an
already-admitted provider, or with itself -- is **dropped as one unit**: its
tools, its `environment`, and its modality preference all go, so its prompt text
can never describe tools that were discarded. The same happens to a provider
left with no surviving tool at all. Each drop is logged.

Survivors are matched back to their owning provider by **object identity**, not
by name. A tool this factory already rejected must not be able to claim its name
back and evict a different provider's accepted tool of the same name.

### Policy narrowing is not permanent

Tool policy can widen again between turns. Every rebuild therefore re-derives
from the full, pre-policy contribution rather than re-narrowing an
already-narrowed view. A tool filtered out by a restrictive tool policy comes
back -- along with its provider's prompt text -- when the policy widens.

This is implemented with a registry-internal back-reference on the narrowed
view. Which leads to:

### Fields providers do not set

`TaskRuntimeContribution` has three registry-internal bookkeeping fields:
`tool_origins`, `provider_contributions`, and `source_contribution`. **Providers
set only `tools`, `environment`, and `preferred_input_modalities`.**
Normalization discards any `provider_contributions` and `source_contribution` a
provider supplies, and the merge recomputes `tool_origins` from the tools it
actually received.

### `preferred_input_modalities` is advisory

The routing layer distinguishes two sources of modality requirements:

| Source | Kind | Router cannot honour it |
| --- | --- | --- |
| the conversation's own content (image parts, audio parts, context references) | hard requirement | `RouterModalityRoutingError` -- routing fails |
| `TaskRuntimeContribution.preferred_input_modalities` | advisory | logged at INFO, dropped, routing proceeds |

When the installed `xrouter-llm` `RoutingService.route()` accepts modality
preferences, advisory and required modalities are passed together. When it does
not, only the modalities the messages actually carry can fail the request; a
provider's hint is simply not applied. An explicitly selected model is never
rejected because of a provider hint.

### Contribution limits

| Limit | Value |
| --- | --- |
| `runtime_extensions` entries per create request | 16 |
| provider configuration, JSON-encoded | 64 KiB |
| `environment`, per provider and merged | 64 KiB |
| `tools`, per provider and merged | 64 |
| aggregate `public_metadata` response | 256 KiB |

Exceeding a limit inside `build_runtime` drops that provider's contribution
(fail-open). Exceeding one in `on_task_created` or `public_metadata` fails the
request.

## The context object

Providers receive a frozen `TaskRuntimeContext`:

```python
task_id: int
user_id: int                       # always the task OWNER, never the acting admin
source: str | None
session_factory: Callable[[], Any]
workspace: Any | None = None       # populated only during build_runtime
```

They never receive an ORM `Task`, a request object, or a checked-out SQLAlchemy
session.

**`session_factory` is a factory, not a session, on purpose. Providers own every
session they open and must close it inside the hook that opened it.** The
framework deliberately does not retain or instrument provider sessions, so
nothing will close one for you. In particular, **a contributed tool must not
capture a session** -- tools outlive the `build_runtime` call that produced
them, and a retained session is a pooled connection held for the lifetime of the
task. Open a fresh short session inside the tool's own execution instead.

`workspace` is set only for `build_runtime`; it is `None` in the other three
hooks.

## Public metadata

Client-safe live state is available from:

```text
GET /api/chat/task/{task_id}/runtime-extensions
```

```json
{
  "task_id": 42,
  "runtime_extensions": {"computer": {"target": "approved_browser"}},
  "runtime_extensions_status": "complete",
  "runtime_extensions_omitted": []
}
```

Return non-secret, JSON-compatible values only, or `None` to contribute nothing.
A non-mapping return value is an error.

The endpoint is fail-closed: a provider failure returns that provider's
`TaskRuntimeClientError` status (400/403) or otherwise 500. The same metadata is
also attached to the task-create response, but there it is optional decoration
-- the binding is already persisted, so a failure yields an empty mapping and
`runtime_extensions_status: "failed"` rather than failing the create.

If the aggregate response would exceed 256 KiB (less a 2 KiB reserve for the
status fields), later providers are omitted, listed in
`runtime_extensions_omitted`, and the status becomes `"truncated"`.

## Deletion and idempotency

`DELETE /api/chat/task/{task_id}` runs provider cleanup **before** the core rows
are deleted, and dispatches `on_task_deleted` **only to the providers named in
that task's binding record**, in reverse registry order. Providers the task
never bound to are not called, so one broken extension cannot block deletion
deployment-wide.

All bound providers are attempted even after one fails.

| Situation | Result |
| --- | --- |
| every bound provider succeeds | core rows deleted, 200 |
| a bound provider raises or times out | **503, and the task is not deleted** -- retry after fixing the provider |
| the task is bound to a name that is no longer registered | logged as an error; deletion proceeds |
| `?force=true` by an admin | core rows deleted anyway; failures logged loudly as leaked state |
| `?force=true` by a non-admin | 403 |

Blocking deletion forever because a provider was unloaded from the deployment
would be worse than the leak, so an unregistered binding makes the leak loud
instead of fatal. `force=true` is the escape hatch for a chronically failing
provider; it is admin-only and it leaks whatever that provider still holds.

### Why `on_task_deleted` must be idempotent

This is a hard requirement of the contract, and there are four independent ways
the hook gets called more than once for the same task:

1. **Retry after 503.** A single-task delete that fails does *not* narrow the
   binding record. The task is preserved with its full binding set, so a retry
   re-dispatches **every** bound provider -- including the ones that already
   released successfully on the first attempt.
2. **Bindings are recorded optimistically.** The record is written before any
   `on_task_created` runs, so a provider that never actually created anything
   still receives `on_task_deleted`. The hook must treat "nothing to release" as
   success, not as an error.
3. **Create compensation.** A create failure dispatches `on_task_deleted` to the
   provider that just failed, whose own setup may have partially completed.
4. **Core deletion can still fail after cleanup succeeded.** Provider hooks run
   first; if the subsequent row purge fails, the next attempt starts over from
   the hooks.

Concretely: a provider that releases an external lease or sandbox should persist
a provider-side "release requested" state and reconcile it safely on repeated
calls, instead of treating the first release attempt as an irreversible one-shot
action. Returning cleanly when there is nothing left to release is the correct
behavior, not a silent bug.

### Admin user deletion

`DELETE /api/admin/users/{user_id}` walks the user's tasks in keyset-paged
batches and runs cleanup concurrently within a page, skipping tasks with no
binding record entirely.

Unlike single-task deletion, this path **does** narrow each task's binding
record after every page, persisting exactly the extensions that are still held
(`TaskRuntimeExtensionError.unreleased_extensions`, or the empty set on
success). Without that marker, a failure on a later page would leave earlier
pages' tasks with released providers, no DB-visible record of it, and a retry
that re-dispatches cleanup for state that is already gone. Any cleanup failure
aborts the whole user deletion with 503.

## Hook isolation and timeouts

Every hook invocation goes through the same guard. The call is submitted to a
process-wide `ThreadPoolExecutor`, so a synchronous provider cannot block the
event loop, and it is bounded by two timeouts:

| Phase | Timeout | Configurable |
| --- | --- | --- |
| waiting for a free worker thread | 30s | `XAGENT_TASK_RUNTIME_HOOK_QUEUE_TIMEOUT_SECONDS` |
| `on_task_created` execution | 30s | no |
| `build_runtime` execution | 10s | no |
| `public_metadata` execution | 10s | no |
| `on_task_deleted` execution | 30s | no |

The pool size defaults to 8 threads and is set by
`XAGENT_TASK_RUNTIME_HOOK_MAX_WORKERS`. Both env vars take a positive integer;
an invalid or non-positive value falls back to the default. The pool is created
lazily and shut down with the application, and it is recreated on demand so
embedded apps and tests can run more than one lifespan in the same process.

**The per-hook execution timeouts are constants, not configuration.** A provider
that needs longer than its budget must do the slow work outside the hook.

On timeout the hook raises `TimeoutError`, which is handled exactly like any
other provider exception for that hook -- 503 on create, a dropped contribution
on build, 500 on the metadata endpoint, 503-and-preserve on delete. Note that
cancelling a timed-out worker **cannot stop Python code already running in that
thread**: the timeout bounds the request path, not the provider. An `async`
hook does receive normal cancellation.

`SystemExit`, `KeyboardInterrupt`, and a genuine cancellation of the
surrounding task are re-raised rather than absorbed into the per-hook failure
policy, so process control and request cancellation still work through an
untrusted provider.

## Failure policy summary

The asymmetry is deliberate, and it is the single most important thing to
understand before shipping a provider:

- **`build_runtime` is fail-open.** Runtime tools are optional enrichment. A
  broken out-of-tree provider must not prevent every task in the deployment from
  constructing its core tool set. The operator sees an error log and a task
  running without the extension.
- **`on_task_created` is fail-closed.** A binding that was not established must
  not look established.
- **`public_metadata` is fail-closed on its own endpoint**, because metadata is
  that endpoint's entire response, and degraded on task create, because there it
  is decoration on an already-successful binding.
- **`on_task_deleted` is fail-closed for providers that own the task.** The
  alternative -- deleting the core row and orphaning provider state -- is
  unrecoverable, whereas a 503 leaves an idempotent retry available. The cost is
  that a permanently broken provider makes its tasks undeletable, which is what
  the admin `force=true` escape hatch exists for.
