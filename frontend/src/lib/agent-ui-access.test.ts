import { describe, expect, it } from "vitest"

import {
  canDeleteAgent,
  canEditAgent,
  canPublishAgent,
  canRunAgent,
  findRunnableAgentById,
  getAgentChatHref,
  getAgentTaskHref,
} from "./agent-ui-access"

describe("agent-ui-access", () => {
  it("allows readonly published agents to run without edit actions", () => {
    const agent = {
      id: 42,
      status: "published",
      readonly: true,
      can_edit: false,
      can_publish: false,
      can_delete: false,
    }

    expect(canRunAgent(agent)).toBe(true)
    expect(canEditAgent(agent)).toBe(false)
    expect(canPublishAgent(agent)).toBe(false)
    expect(canDeleteAgent(agent)).toBe(false)
    expect(getAgentChatHref(agent)).toBe("/task?agent=42")
  })

  it("uses the shared task composer for editable owner agents too", () => {
    const agent = {
      id: 7,
      status: "published",
      readonly: false,
      can_edit: true,
    }

    expect(canRunAgent(agent)).toBe(true)
    expect(canEditAgent(agent)).toBe(true)
    expect(getAgentChatHref(agent)).toBe("/task?agent=7")
    expect(getAgentTaskHref("agent/id with spaces")).toBe("/task?agent=agent%2Fid%20with%20spaces")
  })

  it("only resolves runnable agents from query parameters", () => {
    const agents = [
      { id: 1, status: "draft" },
      { id: 2, status: "published" },
    ]

    expect(findRunnableAgentById(agents, "1")).toBeUndefined()
    expect(findRunnableAgentById(agents, "2")).toEqual(agents[1])
    expect(findRunnableAgentById(agents, null)).toBeUndefined()
  })
})
