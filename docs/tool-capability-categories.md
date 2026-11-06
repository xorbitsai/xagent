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

## Existing selections

Old selectors remain accepted without rewriting stored agents, templates,
execution snapshots or SDK requests:

- `basic` preserves its previous built-in tools: structured questions, JSON
  translation, Python, JavaScript, shell commands and `api_call`.
- `web_search` preserves search, webpage fetching and asset downloading. It
  does not grant `api_call`.

The backend filters these legacy groups by their original tool membership.
The builder shows retained selections as existing permissions and lists their
tools. Saving unrelated changes preserves the selectors; users can explicitly
replace them with the new groups. Only new groups are advertised for new agent
configuration. This prevents search-only agents from gaining HTTP API access
and keeps existing code-enabled agents working.

Category selection does not replace sandbox isolation or file/network policy.
Code tools continue to follow the deployment's configured sandbox policy.
