import { describe, expect, it } from "vitest"
import { normalizeInteractions } from "./app-context-chat"

describe("normalizeInteractions", () => {
  it("preserves a persisted first-chat summary, with only well-formed display fields", () => {
    expect(normalizeInteractions([{
      type: "teammate_summary", field: "teammate_summary", label: "Social posts",
      description: "Draft platform-native posts.", placeholder: "Share a brief.", apps: ["Instagram", null],
    }])).toEqual([{
      type: "teammate_summary", field: "teammate_summary", label: "Social posts",
      description: "Draft platform-native posts.", placeholder: "Share a brief.", apps: ["Instagram"],
    }])
    const [malformed] = normalizeInteractions([{
      type: "teammate_summary", field: "teammate_summary", description: { unsafe: true },
    }])
    expect(malformed).not.toHaveProperty("description")
  })
  it.each(["weekly", 0, false, null])("preserves the canonical suggested value %s", defaultValue => {
    const [result] = normalizeInteractions([
      { type: "text_input", field: "choice", label: "Choice", default_value: defaultValue },
    ])
    expect(result.default_value).toBe(defaultValue)
  })

  it.each([{}, [], Infinity, NaN])("drops malformed suggested values %s", defaultValue => {
    const [result] = normalizeInteractions([
      { type: "text_input", field: "choice", label: "Choice", default_value: defaultValue },
    ])
    expect(result).not.toHaveProperty("default_value")
  })

  it("keeps a connect_apps interaction and passes its apps list through", () => {
    const result = normalizeInteractions([
      { type: "connect_apps", field: "connect_apps", label: "Connect your apps", apps: ["Gmail", "HubSpot"] },
    ])

    expect(result).toEqual([
      {
        type: "connect_apps",
        field: "connect_apps",
        label: "Connect your apps",
        apps: ["Gmail", "HubSpot"],
      },
    ])
  })

  it("drops non-string entries from apps", () => {
    const result = normalizeInteractions([
      { type: "connect_apps", field: "connect_apps", label: "Connect", apps: ["Gmail", 42, null] },
    ])

    expect(result[0].apps).toEqual(["Gmail"])
  })

  it("still filters out an unrecognized interaction type", () => {
    const result = normalizeInteractions([{ type: "not_a_real_type", field: "x", label: "x" }])

    expect(result).toEqual([])
  })

  it("accepts connect_connector and keeps its display data", () => {
    const connector = { interaction_id: "i1", app_id: "cosmos", name: "Azure Cosmos DB", route: "mcp", fields: [] }
    expect(normalizeInteractions([
      { type: "connect_connector", field: "connect_connector", label: "Connect Azure Cosmos DB", live_widget: true, interaction_id: "i1", connector },
    ])).toEqual([
      { type: "connect_connector", field: "connect_connector", label: "Connect Azure Cosmos DB", interaction_id: "i1", connector },
    ])
  })
})
