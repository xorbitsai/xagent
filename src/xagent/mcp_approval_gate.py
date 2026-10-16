"""Stable host integration surface for task-scoped MCP approval gates."""

from .core.tools.adapters.vibe.mcp_approval_gate import (
    ConnectorRef,
    GateClassifier,
    GatedCall,
    GateDecision,
    GateToolMatcher,
    MCPApprovalGateRegistration,
    MCPApprovalGateScope,
    MCPApprovalReplayContext,
    ToolCallExecutionContext,
    current_mcp_approval_replay_context,
    has_scoped_approval_gate_for_source,
    mcp_approval_gate_rollback_fence,
    mcp_approval_gate_scope,
    register_mcp_approval_gate,
    unregister_mcp_approval_gate,
)

__all__ = [
    "ConnectorRef",
    "GatedCall",
    "GateClassifier",
    "GateDecision",
    "GateToolMatcher",
    "MCPApprovalGateRegistration",
    "MCPApprovalGateScope",
    "MCPApprovalReplayContext",
    "ToolCallExecutionContext",
    "current_mcp_approval_replay_context",
    "has_scoped_approval_gate_for_source",
    "mcp_approval_gate_rollback_fence",
    "mcp_approval_gate_scope",
    "register_mcp_approval_gate",
    "unregister_mcp_approval_gate",
]
