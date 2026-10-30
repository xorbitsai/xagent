# FXMacroData MCP

FXMacroData MCP serves official macroeconomic releases, release calendars,
central bank policy rates and FX data for 22 currencies at
`https://mcp.fxmacrodata.com`. It can be connected as a custom MCP server. The
connection is opt-in and does not change Xagent's built-in tools or saved
credentials.

Without an API key, USD releases, the USD release calendar and the USD
indicator catalogue work, and each release becomes readable 15 minutes after it
is published. Other currencies, FX rates and the remaining tools need a key from
https://fxmacrodata.com/subscribe and return a `subscription_required` error
without one.

## Run the example

From a source checkout with Xagent's core dependencies installed (`pip install
-e .`), run from the repository root:

```bash
PYTHONPATH=src python scripts/fxmacrodata_example.py \
  --config docs/examples/fxmacrodata/mcp_server.yaml \
  --currency USD
```

The YAML uses Xagent's `MCPServerConfig` format. The example validates it,
converts it to an effective connection, and discovers tools with
`load_mcp_tools_as_agent_tools`. It calls the resulting agent tools for
`release_calendar` and `latest_announcements` for the requested currency and
prints both results in Xagent's normal MCP result shape. No LLM credentials are
needed to run this tool example.

Keyless results carry a `freemium_delay` object. The example copies its message
into a top-level `notices` list so an agent can say that the data is delayed
instead of presenting it as current. Use `announcement_datetime` for when a
figure was or will be published; a row's `date` is the period the figure refers
to, not the day it was released.

To use a key, set `FXMACRODATA_API_KEY`. The example then sends it as an
`Authorization: Bearer` header. The key is checked before any request, is only
sent to an `https` URL (or a loopback address for local testing), and is
replaced with `[redacted]` in printed results and errors. The YAML itself never
contains a key.

This standalone example does not initialize Xagent's database or modify saved
connectors. Discovery, execution, error handling, cancellation and transport
timeouts use the existing MCP runtime.

## Connect in Xagent

In the Connectors page, add a **Custom MCP** connection with these values:

| Setting | Value |
| --- | --- |
| Name | `fxmacrodata` |
| Transport | `streamable_http` |
| URL | `https://mcp.fxmacrodata.com` |
| Authentication | None, or a header `Authorization: Bearer <your key>` |
| Header | `User-Agent: xagent/fxmacrodata-example` |

Load the server's tools and select the connector for the agent that needs it.
Its tools appear with the MCP server prefix. Removing the connector from that
agent removes its access to these tools.

## Validate the example

With the core dependencies installed, the local protocol test starts a loopback
MCP server and exercises the example command, tool loading, request headers with
and without a key, key redaction, a locked-currency error, the delay notice and
input validation:

```bash
PYTHONPATH=src python -m unittest discover -s tests/scripts -p test_fxmacrodata_example.py
```
