import { describe, expect, it } from "vitest"

import { composeSidebarNavigation, getNavigationGroupsForUser, getUserMenuItemsForUser } from "./sidebar-navigation"

describe("sidebar navigation", () => {
  it("preserves Conversation Logs inside Builder with its own icon", () => {
    const groups = getNavigationGroupsForUser({ is_admin: false })
    const builder = groups.find((group) => group.id === "builder")

    expect(builder?.items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "Conversation Logs",
          nameKey: "nav.conversationLogs",
          href: "/conversation-logs",
        }),
      ])
    )

    const channels = builder?.items.find((item) => item.href === "/channels")
    const conversationLogs = builder?.items.find(
      (item) => item.href === "/conversation-logs"
    )
    expect(conversationLogs?.icon).not.toBe(channels?.icon)
  })

  it("places advanced features under one collapsed Builder group in design order", () => {
    const groups = getNavigationGroupsForUser({ is_admin: false })
    expect(groups).toHaveLength(1)
    expect(groups[0]).toMatchObject({
      id: "builder", titleKey: "nav.builder", defaultCollapsed: true,
      descriptionKey: "nav.builderDescription",
    })
    expect(groups[0].items.slice(0, 5).map((item) => item.href)).toEqual([
      "/workforces?view=teammates", "/kb", "/models", "/api-keys", "/memory",
    ])
    expect(groups[0].items[0].nameKey).toBe("nav.teammates")
  })

  it("puts common actions on the first level without duplicating them in Builder", () => {
    const topLevelRoutes = getUserMenuItemsForUser().map((item) => item.href)
    expect(topLevelRoutes).toEqual(["/tools", "/skill-hub", "/files", "/settings"])
    expect(getNavigationGroupsForUser()[0].items.some((item) => topLevelRoutes.includes(item.href))).toBe(false)
  })

  it("exposes administrative routes only to administrators", () => {
    const items = getNavigationGroupsForUser({ is_admin: true })[0].items
    expect(items.find((item) => item.name === "User Management")?.href).toBe("/users")
    expect(items.find((item) => item.name === "Public MCP Apps")?.href).toBe("/admin-mcp")
    for (const user of [undefined, null, { is_admin: false }]) {
      expect(getNavigationGroupsForUser(user)[0].items.some((item) => ["/users", "/admin-mcp"].includes(item.href))).toBe(false)
    }
  })

  it("preserves legacy groups and skips empty contributions without mutating inputs", () => {
    const base = getNavigationGroupsForUser()
    const common = getUserMenuItemsForUser()
    const snapshot = base[0].items.slice()
    const item = snapshot[0]
    const extra = [
      { id: "legacy", title: "Legacy", items: [item] },
      { id: "empty", title: "Empty", items: [] },
    ]
    const result = composeSidebarNavigation(base, common, extra)
    expect(result.groups.map((group) => group.id)).toEqual(["builder", "legacy"])
    expect(result.userItems).toEqual(common)
    expect(base[0].items).toEqual(snapshot)
    expect(result.groups[0].items).not.toBe(base[0].items)
    expect(result.groups[1].items).not.toBe(extra[0].items)
  })

  it("keeps a Builder contribution accessible if the host has no Builder group", () => {
    const contribution = { id: "ai", title: "AI", placement: "builder" as const, items: getUserMenuItemsForUser() }
    expect(composeSidebarNavigation([], [], [contribution]).groups).toEqual([contribution])
  })
})
