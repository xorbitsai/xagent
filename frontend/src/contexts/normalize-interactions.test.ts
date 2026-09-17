import { describe, expect, it } from "vitest"
import { normalizeInteractions } from "./app-context-chat"

describe("normalizeInteractions", () => {
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
})
