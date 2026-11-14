"""The MCP result member a Composio-backed connector reports its count on.

Composio bills each Proxy Execute request, and one connector tool call can make
several. The connector subprocess counts the requests Composio accepted and
puts the count in the MCP tool result's ``_meta`` under this key; the agent
process reads it in ``MCPToolAdapter`` and records one usage row per request.

The key lives here, outside ``media_usage`` (which loads the model layer), so a
connector subprocess can import it cheaply, and under ``core`` so the adapter
can read it without importing the web package.
"""

COMPOSIO_REQUESTS_META_KEY = "xagent/composio_requests"
