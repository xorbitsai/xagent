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

  it("drops primitive non-string entries from apps", () => {
    const result = normalizeInteractions([
      { type: "connect_apps", field: "connect_apps", label: "Connect", apps: ["Gmail", 42, null] },
    ])

    expect(result[0].apps).toEqual(["Gmail"])
  })

  it("keeps an {id, name} object entry from apps, not just plain strings", () => {
    const result = normalizeInteractions([
      {
        type: "connect_apps",
        field: "connect_apps",
        label: "Connect",
        apps: [{ id: "gmail", name: "Gmail" }, "HubSpot"],
      },
    ])

    expect(result[0].apps).toEqual([{ id: "gmail", name: "Gmail" }, "HubSpot"])
  })

  it("still filters out an unrecognized interaction type", () => {
    const result = normalizeInteractions([{ type: "not_a_real_type", field: "x", label: "x" }])

    expect(result).toEqual([])
  })
})
