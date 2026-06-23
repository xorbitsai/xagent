import { getApiSnippetTarget, type ApiSnippetTarget } from "@/lib/api-snippet-base-url"

export type ApiSnippetTab = "curl" | "python"

export function formatAgentApiSnippets(
  agentId: number,
  apiTarget: ApiSnippetTarget = getApiSnippetTarget(),
): Record<ApiSnippetTab, string> {
  return {
    curl: `curl -X POST ${apiTarget.baseUrl}/v1/chat/tasks \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "agent_id": ${agentId},
    "message": { "role": "user", "content": "Hello" }
  }'`,
    python: `# pip install "xagent-sdk @ git+https://github.com/xorbitsai/xagent-sdk@v0.3.1#subdirectory=python"
from xagent_sdk import AgentClient

with AgentClient(api_key="YOUR_API_KEY", base_url="${apiTarget.baseUrl}") as agent:
    result = agent.tasks.run(agent_id=${agentId}, message="Hello")
    print(result.output)`,
  }
}
