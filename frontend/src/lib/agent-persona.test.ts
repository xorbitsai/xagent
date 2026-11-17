import { describe, expect, it } from "vitest"

import { toAgentPersona } from "./agent-persona"

describe("toAgentPersona", () => {
  it("carries the agent's name, image and id", () => {
    expect(toAgentPersona({ id: 7, name: "Maya" }, "/logo.png")).toEqual({ name: "Maya", avatar: "/logo.png", agentId: 7 })
  })

  it("accepts a numeric string id", () => {
    expect(toAgentPersona({ id: "12", name: "Maya" }, null).agentId).toBe(12)
  })

  it.each([undefined, null, "", "abc", 0, -3, 1.5, "7.5"])("gives no id for %j", (id) => {
    expect(toAgentPersona({ id, name: "Maya" }, null).agentId).toBeNull()
  })

  it("has no image when none is given", () => {
    expect(toAgentPersona({ id: 7, name: "Maya" }, undefined).avatar).toBeNull()
  })
})
