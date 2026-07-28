# Task runtime extensions

Task runtime extensions let an application distribution attach task-scoped
resources without adding provider-specific columns and branches throughout the
open-source agent runtime. A provider is registered once at application
startup:

```python
from xagent.core.task_runtime import (
    TaskRuntimeContext,
    TaskRuntimeContribution,
    TaskRuntimeExtensionProvider,
)
from xagent.web.services.task_runtime import register_task_extension


class ComputerRuntimeProvider(TaskRuntimeExtensionProvider):
    async def on_task_created(
        self,
        context: TaskRuntimeContext,
        configuration: dict,
    ) -> None:
        # Validate the selection and persist a task binding.
        ...

    async def build_runtime(
        self,
        context: TaskRuntimeContext,
    ) -> TaskRuntimeContribution:
        binding = ...
        if binding is None:
            return TaskRuntimeContribution()
        return TaskRuntimeContribution(
            tools=(build_computer_tool(context, binding),),
            environment=(
                "The task has access to the browser target selected by the user."
            ),
            preferred_input_modalities=("image",),
        )

    async def public_metadata(
        self,
        context: TaskRuntimeContext,
    ) -> dict:
        # Return non-secret, JSON-compatible state only.
        return {"target": "browser"}

    async def on_task_deleted(self, context: TaskRuntimeContext) -> None:
        # Make cleanup idempotent.
        ...


register_task_extension("computer", ComputerRuntimeProvider())
```

The normal task-create request accepts provider configuration under
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

Unknown providers and non-object configurations are rejected before the task
is created. Provider setup runs after the core task row is committed so a
provider can open its own short database session and safely reference the task.
If setup fails, Xagent invokes provider cleanup and compensates the newly
created core task and file bindings.

## Runtime behavior

`build_runtime` runs whenever Xagent constructs or reconstructs the task's
`AgentService`. Contributions are merged in provider registration order:

- `tools` enter the standard tool pipeline before selection, per-user policy,
  sandbox wrapping, and output filtering;
- `environment` is appended to the task's system context and must not contain
  credentials;
- `preferred_input_modalities` informs virtual model routing but remains a
  preference, so an explicitly selected model is not rejected.

Providers receive a frozen `TaskRuntimeContext` containing primitive task,
owner, and source values, a task workspace during runtime construction, and a
`session_factory`. They never receive an ORM `Task`, request object, or
checked-out SQLAlchemy session. Providers must use short operation-local
sessions and must not retain them in contributed tools.

Client-safe live state is available from:

```text
GET /api/chat/task/{task_id}/runtime-extensions
```

Task deletion attempts every provider's `on_task_deleted` hook in reverse
registration order. Cleanup errors are logged after the core deletion rather
than making an already-deleted task reappear.
