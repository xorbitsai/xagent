# Trusted memory backend bindings

`xagent.web.services.memory_binding` is an opt-in host extension point. It does
not replace the default memory manager, lifecycle, policy resolver, tools, or
API routes by itself.

Register one synchronous resolver during process initialization:

```python
from xagent.web.services.memory_binding import (
    MemoryManagementBindingRequest,
    MemoryStoreBinding,
    MemoryTaskBindingRequest,
    set_trusted_memory_binding_resolver,
)


def resolve_memory_binding(request):
    if isinstance(request, MemoryManagementBindingRequest):
        principal_id = request.authenticated_user_id
    elif isinstance(request, MemoryTaskBindingRequest):
        principal_id = request.user_id
    else:  # The public resolver never supplies another request type.
        raise TypeError("unsupported memory binding request")

    store, identity, generation = select_existing_store(principal_id)
    return MemoryStoreBinding(
        store=store,
        backend_identity=identity,
        route_generation=generation,
        public_diagnostics={"backend": "host-managed"},
    )


set_trusted_memory_binding_resolver(resolve_memory_binding)
```

Task input is a detached projection of trusted persisted fields. Management
input contains only the authenticated principal. Caller-controlled headers,
query parameters, model text, and ambient context must not select a backend.
The resolver must not retain an ORM session in the returned binding.

`resolve_trusted_memory_binding()` returns `None` only when no resolver is
registered. Once registered, callback errors and invalid returns raise
`MemoryBindingResolutionError`; a consumer must fail closed instead of
selecting the process-wide store. Async consumers use
`resolve_trusted_memory_binding_async()`, which runs the synchronous callback
off the event loop.

Each `MemoryStoreBinding` is a frozen snapshot containing:

- an existing `MemoryStore`;
- opaque `backend_identity` and `route_generation` cache-fencing values;
- an optional `MemoryAccessScope`, bound before the snapshot is returned;
- flat, scalar `public_diagnostics` safe for a host-selected store-info view;
- an optional `memory_tools_factory` that accepts one frozen
  `MemoryToolAssemblyRequest`.

The consumer layer is responsible for resolving once per request or turn,
including identity and generation in its retrieval/cache fingerprint, keeping
an in-flight turn on its original binding, and invoking the factory only for
that selected binding. A missing factory means the existing default memory
tools remain unchanged.

For the local user-isolation wrapper, `MemoryAccessScope` still requires
non-empty dimensions by default. Explicit owner-only access uses
`strict_owner_only=True` with an empty mapping. It sees only dimension-less
notes belonging to that owner and excludes all dimension-stamped notes.
