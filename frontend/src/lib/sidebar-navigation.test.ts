import { describe, expect, it } from "vitest"

import { getNavigationGroupsForUser } from "./sidebar-navigation"

describe("sidebar navigation", () => {
  it("exposes Conversation Logs under the More resource menu", () => {
    const groups = getNavigationGroupsForUser({ is_admin: false })
    const resources = groups.find((group) => group.titleKey === "nav.sections.resources")
    const more = resources?.items.find((item) => item.href === "__resources_more__")

    expect(more?.children).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "Conversation Logs",
          nameKey: "nav.conversationLogs",
          href: "/conversation-logs",
        }),
      ])
    )

    const channels = more?.children?.find((item) => item.href === "/channels")
    const conversationLogs = more?.children?.find(
      (item) => item.href === "/conversation-logs"
    )
    expect(conversationLogs?.icon).not.toBe(channels?.icon)
  })

  it("collapses Resources by default and gives each built-in group a stable id", () => {
    const groups = getNavigationGroupsForUser({ is_admin: false })
    const agentDevelopment = groups.find((group) => group.titleKey === "nav.sections.agentDevelopment")
    const resources = groups.find((group) => group.titleKey === "nav.sections.resources")

    expect(agentDevelopment?.id).toBe("agent-development")
    expect(agentDevelopment?.defaultCollapsed).toBeFalsy()
    expect(resources?.id).toBe("resources")
    expect(resources?.defaultCollapsed).toBe(true)
  })

  it("renames the build/templates nav entries to My Team/Add teammates", () => {
    const groups = getNavigationGroupsForUser({ is_admin: false })
    const agentDevelopment = groups.find((group) => group.titleKey === "nav.sections.agentDevelopment")

    expect(agentDevelopment?.items.find((item) => item.href === "/build")).toEqual(
      expect.objectContaining({ name: "My Team", nameKey: "nav.myTeam" })
    )
    expect(agentDevelopment?.items.find((item) => item.href === "/templates")).toEqual(
      expect.objectContaining({ name: "Add teammates", nameKey: "nav.addTeammates" })
    )
  })

  it("keeps the admin-only routes free of trailing slashes so active-route matching works", () => {
    const groups = getNavigationGroupsForUser({ is_admin: true })
    const resources = groups.find((group) => group.titleKey === "nav.sections.resources")
    const more = resources?.items.find((item) => item.href === "__resources_more__")

    expect(more?.children?.find((item) => item.name === "User Management")?.href).toBe("/users")
    expect(more?.children?.find((item) => item.name === "Public MCP Apps")?.href).toBe("/admin-mcp")
  })
})
