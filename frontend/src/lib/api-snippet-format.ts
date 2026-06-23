import type { ApiSnippetTarget } from "@/lib/api-snippet-target"

export type ApiSnippetTab = "curl" | "python"

export function formatAgentApiSnippets(
  agentId: number,
  apiTarget: ApiSnippetTarget,
): Record<ApiSnippetTab, string> {
  const baseUrl = apiTarget.baseUrl || "YOUR_API_BASE_URL"

  return {
    curl: `curl -X POST ${baseUrl}/v1/chat/tasks \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "agent_id": ${agentId},
    "message": { "role": "user", "content": "Hello" }
  }'`,
    python: `# pip install "xagent-sdk @ git+https://github.com/xorbitsai/xagent-sdk@v0.3.1#subdirectory=python"
from xagent_sdk import AgentClient

with AgentClient(api_key="YOUR_API_KEY", base_url="${baseUrl}") as agent:
    result = agent.tasks.run(agent_id=${agentId}, message="Hello")
    print(result.output)`,
  }
}
