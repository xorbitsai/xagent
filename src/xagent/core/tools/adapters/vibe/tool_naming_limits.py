"""Provider-imposed limits on generated tool names.

Split out of agent_tool_names.py so that modules needing only this
constant -- including ones imported inside a reduced-dependency sandbox,
like mcp_adapter.py and api_tool_adapter.py -- don't pull in that
module's own concerns (Chinese-name romanization, and whatever
optional/backend-only dependency that romanization needs) just to read
a number.
"""

MAX_AGENT_TOOL_NAME_LENGTH = 64
