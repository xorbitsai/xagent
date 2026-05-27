import sys
from types import ModuleType, SimpleNamespace

mcp_module = ModuleType("mcp")
mcp_module.ClientSession = object
mcp_module.StdioServerParameters = object
mcp_types_module = ModuleType("mcp.types")
mcp_types_module.Tool = object
mcp_client_module = ModuleType("mcp.client")
mcp_client_sse_module = ModuleType("mcp.client.sse")
mcp_client_sse_module.sse_client = object
mcp_client_stdio_module = ModuleType("mcp.client.stdio")
mcp_client_stdio_module.stdio_client = object
mcp_client_streamable_http_module = ModuleType("mcp.client.streamable_http")
mcp_client_streamable_http_module.streamablehttp_client = object
mcp_shared_module = ModuleType("mcp.shared")
mcp_shared_httpx_utils_module = ModuleType("mcp.shared._httpx_utils")
mcp_shared_httpx_utils_module.create_mcp_http_client = object
sandbox_helper_module = ModuleType(
    "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper"
)
sandbox_helper_module.load_sandboxed_mcp_tools = object
sandbox_helper_module.should_sandbox_mcp_connection = object
mcp_module.types = mcp_types_module
sys.modules.setdefault("mcp", mcp_module)
sys.modules["mcp.types"] = mcp_types_module
sys.modules["mcp.client"] = mcp_client_module
sys.modules["mcp.client.sse"] = mcp_client_sse_module
sys.modules["mcp.client.stdio"] = mcp_client_stdio_module
sys.modules["mcp.client.streamable_http"] = mcp_client_streamable_http_module
sys.modules["mcp.shared"] = mcp_shared_module
sys.modules["mcp.shared._httpx_utils"] = mcp_shared_httpx_utils_module
sys.modules[
    "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper"
] = sandbox_helper_module

from xagent.core.tools.adapters.vibe.mcp_adapter import MCPToolAdapter  # noqa: E402


def test_build_args_model_handles_optional_array_schema():
    mcp_tool = SimpleNamespace(
        name="gmail_manage_labels",
        description="Manage Gmail labels",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "add_label_ids": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            "required": ["action"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    args_model = adapter.args_type()
    parsed = args_model(action="modify_message", add_label_ids=["TRASH"])

    assert parsed.add_label_ids == ["TRASH"]


def test_normalize_args_by_schema_wraps_scalar_for_array_field():
    mcp_tool = SimpleNamespace(
        name="gmail_manage_labels",
        description="Manage Gmail labels",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "add_label_ids": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            "required": ["action"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    normalized = adapter._normalize_args_by_schema(
        {"action": "modify_message", "add_label_ids": "TRASH"}
    )

    assert normalized["add_label_ids"] == ["TRASH"]
