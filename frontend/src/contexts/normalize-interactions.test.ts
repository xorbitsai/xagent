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

  it("drops non-string, non-object entries from apps", () => {
    const result = normalizeInteractions([
      { type: "connect_apps", field: "connect_apps", label: "Connect", apps: ["Gmail", 42, null] },
    ])

    expect(result[0].apps).toEqual(["Gmail"])
  })

  it("keeps an { id, name } app entry instead of dropping it as non-string", () => {
    // The backend now sends an object alongside the display name when it
    // resolved a stable catalog id (two visible apps can share a name, but
    // never an id) - this sanitizer must preserve that shape, not just the
    // legacy plain-string one.
    const result = normalizeInteractions([
      {
        type: "connect_apps",
        field: "connect_apps",
        label: "Connect your apps",
        apps: [{ id: "gmail", name: "Gmail" }, "HubSpot", { id: "bad" }, {}],
      },
    ])

    expect(result[0].apps).toEqual([
      { id: "gmail", name: "Gmail" },
      "HubSpot",
      { id: "bad", name: undefined },
    ])
  })

  it("still filters out an unrecognized interaction type", () => {
    const result = normalizeInteractions([{ type: "not_a_real_type", field: "x", label: "x" }])

    expect(result).toEqual([])
  })
})
