# Tool capability categories

The builder and tool management page use the categories reported by the backend.
Search-provider credential cards carry the same category as the runtime tools.

| Selector | Display name | Capabilities |
| --- | --- | --- |
| `basic_tools` | Basic Tools | Structured questions, JSON field translation, current time and timezone validation |
| `code_execution` | Code Execution | Python, JavaScript and shell commands |
| `web_tools` | Web Tools | Web search, webpage fetching, web asset downloads and HTTP API calls |

The time tools continue to be intrinsic: any nonempty tool selection includes
them, while an explicit zero-tools selection does not. Their display category
is Basic Tools. The calculator is not registered.

File operations, knowledge bases, browser automation, connectors and the other
specialized categories remain independent. The internal `other` fallback is
not assignable to an agent and is not treated as a Basic Tools grant.

## Upgrade existing configurations

The `20261107_split_tool_categories` database migration rewrites persisted
`agents.tool_categories` and `tasks.agent_config.tool_categories`:

- `basic` becomes `basic_tools`, `code_execution`, and `web_tools`.
- `web_search` becomes `web_tools`.
- Duplicate categories are merged, preserving order. Other categories, connector
  scopes, explicit empty selections and unconfigured (`null`) values are kept.

Existing agents retain code execution and HTTP API capabilities through the
new groups. Network capabilities now follow the complete Web Tools group:
old Basic selections also gain search, and old Web Search selections also
gain HTTP API calls. Builders can adjust the migrated selections normally.

Run the database upgrade and restart application processes and workers together.
The runtime and UI use only the new selectors. Update external templates and
API clients to send these selectors; bundled templates are already updated.
The data migration is not reversed on downgrade, because the merged network
group cannot be split back unambiguously.

Category selection does not replace sandbox isolation or file/network policy.
Code tools continue to follow the deployment's configured sandbox policy.
