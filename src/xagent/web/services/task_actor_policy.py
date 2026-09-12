"""Trusted persisted actor references for fresh shared task execution."""

from typing import Any, cast

from ..models.task import Task
from .mcp_runtime import (
    MCPActorAuthorizationPolicy,
    MCPBuiltinOAuthActorPolicyRequiredError,
)
from .task_runtime import (
    MCP_RUNTIME_AUTHORIZATION_POLICY_IDENTITY_KEY,
    MCP_RUNTIME_AUTHORIZATION_POLICY_STDIO_KEY,
    mcp_runtime_authorization_policy_identity,
    mcp_runtime_authorization_policy_required,
)


def bind_shared_actor_policy(
    task: Task, policy: MCPActorAuthorizationPolicy | None, *, is_create: bool
) -> None:
    marked = mcp_runtime_authorization_policy_required(task.agent_config)
    if not marked and policy is None:
        return
    if not marked or not is_create or policy is None:
        raise MCPBuiltinOAuthActorPolicyRequiredError(
            "Shared actor execution requires trusted CREATE acceptance"
        )
    identity = mcp_runtime_authorization_policy_identity(task.agent_config)
    if identity is not None and identity != policy.resource_owner_key:
        raise MCPBuiltinOAuthActorPolicyRequiredError(
            "Actor policy does not match the persisted identity"
        )
    setattr(
        task,
        "agent_config",
        {
            **(task.agent_config or {}),
            MCP_RUNTIME_AUTHORIZATION_POLICY_IDENTITY_KEY: policy.resource_owner_key,
            MCP_RUNTIME_AUTHORIZATION_POLICY_STDIO_KEY: policy.allow_builtin_stdio,
        },
    )


def load_shared_actor_policy(
    task: Task, *, is_create: bool
) -> MCPActorAuthorizationPolicy | None:
    if not mcp_runtime_authorization_policy_required(task.agent_config):
        return None
    identity = mcp_runtime_authorization_policy_identity(task.agent_config)
    stdio = (cast(dict[str, Any] | None, task.agent_config) or {}).get(
        MCP_RUNTIME_AUTHORIZATION_POLICY_STDIO_KEY
    )
    if not is_create or identity is None or type(stdio) is not bool:
        raise MCPBuiltinOAuthActorPolicyRequiredError(
            "Trusted actor execution reference is unavailable"
        )
    return MCPActorAuthorizationPolicy(
        resource_owner_key=identity, allow_builtin_stdio=stdio
    )
