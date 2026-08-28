import { describe, expect, it } from "vitest"

import {
  sanitizeAppIntegrations,
  sanitizeConnectorStatus,
  sanitizeConnectorStatusEntry,
  sanitizeUnsharedConnectors,
  sanitizeUnsharedKnowledgeBases,
} from "./team-sharing-sanitizers"

describe("team sharing response sanitizers", () => {
  it("keeps only well-formed unshared connectors", () => {
    expect(
      sanitizeUnsharedConnectors([
        { type: "mcp", id: 1, name: "GitHub" },
        { type: "mcp", name: "Missing", reason: "unresolved" },
        null,
        "bad",
        { type: "mcp", id: null, name: "Missing id" },
      ]),
    ).toEqual([
      { type: "mcp", id: 1, name: "GitHub" },
      { type: "mcp", name: "Missing", reason: "unresolved" },
    ])
  })

  it("keeps only well-formed unshared knowledge bases", () => {
    expect(
      sanitizeUnsharedKnowledgeBases([{ name: "support" }, null, { name: 42 }, "bad"]),
    ).toEqual([{ name: "support" }])
  })

  it("rejects malformed app integration payloads", () => {
    const app = {
      id: "github",
      name: "GitHub",
      description: "GitHub connector",
      icon: "github",
      server_id: 7,
    }
    expect(sanitizeAppIntegrations([app, null, 1, { id: "missing-fields" }])).toEqual([app])
    expect(sanitizeAppIntegrations({ apps: [app] })).toEqual([])
  })

  it("keeps a connector status entry only when all three fields are real booleans", () => {
    expect(
      sanitizeConnectorStatusEntry({ shared: true, is_owner: false, needs_config: true }),
    ).toEqual({ shared: true, is_owner: false, needs_config: true })
    expect(sanitizeConnectorStatusEntry({ shared: true })).toBeNull()
    expect(
      sanitizeConnectorStatusEntry({ shared: null, is_owner: true, needs_config: false }),
    ).toBeNull()
    expect(
      sanitizeConnectorStatusEntry({ shared: "yes", is_owner: false, needs_config: false }),
    ).toBeNull()
    expect(
      sanitizeConnectorStatusEntry({ shared: true, is_owner: "yes", needs_config: false }),
    ).toBeNull()
    expect(
      sanitizeConnectorStatusEntry({ shared: true, is_owner: false, needs_config: "no" }),
    ).toBeNull()
    expect(sanitizeConnectorStatusEntry("bad")).toBeNull()
    expect(sanitizeConnectorStatusEntry(null)).toBeNull()
  })

  it("keeps only well-formed connector status entries, keyed as given", () => {
    expect(
      sanitizeConnectorStatus({
        "mcp:1": { shared: true, is_owner: false, needs_config: true },
        "mcp:2": { shared: true },
        "mcp:3": { shared: null, is_owner: true, needs_config: false },
        "mcp:4": { shared: "yes", is_owner: false, needs_config: false },
        "mcp:5": "bad",
      }),
    ).toEqual({
      "mcp:1": { shared: true, is_owner: false, needs_config: true },
    })
    expect(sanitizeConnectorStatus("bad")).toEqual({})
    expect(sanitizeConnectorStatus(null)).toEqual({})
    expect(sanitizeConnectorStatus(undefined)).toEqual({})
    expect(sanitizeConnectorStatus([1, 2])).toEqual({})
  })
})
