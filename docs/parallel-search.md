# Parallel Search MCP

Parallel Search MCP provides free, keyless web search and page excerpts at
`https://search.parallel.ai/mcp`. It can be connected as a custom MCP server.
The connection is opt-in and does not change Xagent's built-in web search
provider selection or saved credentials. Free-tier requests are subject to
the service's rate limits.

## Run the example

From a source checkout with Xagent's core dependencies installed (`pip install
-e .`), run from the repository root:

```bash
PYTHONPATH=src python scripts/parallel_search_example.py \
  --config docs/examples/parallel-search/mcp_server.yaml \
  --query "Python asyncio TaskGroup documentation" \
  --fetch-url https://docs.python.org/3/library/asyncio-task.html
```

The YAML uses Xagent's `MCPServerConfig` format. The example validates it,
converts it to an effective connection, and discovers tools with
`load_mcp_tools_as_agent_tools`. It calls the resulting agent tools for
`web_search` and, when `--fetch-url` is supplied, `web_fetch`. The JSON output
contains source URLs and excerpts in Xagent's normal MCP result shape. A fresh
session identifier is reused across both calls. No LLM credentials are needed
to run this tool example.

This standalone example does not initialize Xagent's database or modify saved
connectors. It uses Streamable HTTP, no authorization header, and the
`xagent/parallel-search-example` User-Agent from the YAML. Discovery, execution,
error handling, cancellation, and transport timeouts use the existing MCP
runtime. To fetch a source discovered by search, run the example again with
that URL as `--fetch-url`.

## Connect in Xagent

In the Connectors page, add a **Custom MCP** connection with these values:

| Setting | Value |
| --- | --- |
| Name | `parallel_search` |
| Transport | `streamable_http` |
| URL | `https://search.parallel.ai/mcp` |
| Authentication | None |
| Header | `User-Agent: xagent/parallel-search-example` |

Load the server's tools and select the connector for the agent that needs it.
Its search and fetch tools appear with the MCP server prefix, separately from
the built-in search tools. Removing the connector from that agent removes its
access to these tools. This does not select Parallel as the built-in search
provider.

## Validate the example

With the core dependencies installed, the local protocol test starts a loopback
MCP server and exercises the example command, tool loading, search/fetch calls,
request headers, session reuse, and a server error:

```bash
PYTHONPATH=src python -m unittest discover -s tests/scripts -p test_parallel_search_example.py
```
