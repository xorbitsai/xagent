import { describe, expect, it } from "vitest"

import { formatAgentApiSnippets } from "./api-snippet-format"

describe("formatAgentApiSnippets", () => {
  it("formats runtime snippets with the regional base_url", () => {
    const snippets = formatAgentApiSnippets(42, {
      baseUrl: "https://sg.cloud.xagent.co",
    })

    expect(snippets.curl).toContain("https://sg.cloud.xagent.co/v1/chat/tasks")
    expect(snippets.python).toContain("from xagent_sdk import AgentClient")
    expect(snippets.python).toContain('base_url="https://sg.cloud.xagent.co"')
    expect(snippets.python).not.toContain("Region")
    expect(snippets.python).not.toContain("region=")
  })

  it("formats runtime snippets with an explicit fallback base_url", () => {
    const snippets = formatAgentApiSnippets(42, {
      baseUrl: "https://api.example.test",
    })

    expect(snippets.curl).toContain("https://api.example.test/v1/chat/tasks")
    expect(snippets.python).toContain('base_url="https://api.example.test"')
    expect(snippets.python).not.toContain("Region")
  })
})
