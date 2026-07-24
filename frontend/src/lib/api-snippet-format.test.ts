import { describe, expect, it } from "vitest"

import {
  formatAgentApiSnippets,
  formatWorkforceApiSnippets,
} from "./api-snippet-format"

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

  it("uses a placeholder when the base URL is empty", () => {
    const snippets = formatAgentApiSnippets(42, {
      baseUrl: "",
    })

    expect(snippets.curl).toContain("YOUR_API_BASE_URL/v1/chat/tasks")
    expect(snippets.python).toContain('base_url="YOUR_API_BASE_URL"')
  })
})

describe("formatWorkforceApiSnippets", () => {
  it("targets the workforce runs endpoint with the regional base_url", () => {
    const snippets = formatWorkforceApiSnippets(7, {
      baseUrl: "https://sg.cloud.xagent.co",
    })

    expect(snippets.curl).toContain(
      "https://sg.cloud.xagent.co/v1/workforces/7/runs",
    )
    // Multi-turn / polling reuses the shared chat tasks surface.
    expect(snippets.curl).toContain("/v1/chat/tasks/TASK_ID")
    expect(snippets.python).toContain("from xagent_sdk import WorkforceClient")
    expect(snippets.python).toContain("workforce_id=7")
    expect(snippets.python).toContain(
      'base_url="https://sg.cloud.xagent.co"',
    )
  })

  it("uses a placeholder when the base URL is empty", () => {
    const snippets = formatWorkforceApiSnippets(7, { baseUrl: "" })

    expect(snippets.curl).toContain("YOUR_API_BASE_URL/v1/workforces/7/runs")
    expect(snippets.python).toContain('base_url="YOUR_API_BASE_URL"')
  })
})
