import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { Globe } from "lucide-react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { NavigationGroup } from "@/lib/sidebar-navigation"

import { Sidebar } from "./sidebar"

const authState = vi.hoisted(() => ({
  logout: vi.fn<() => Promise<boolean>>(),
  user: {
    id: "1",
    username: "acct_0123456789abcdef0123456789abcdef",
    email: "alice@example.com",
  },
}))
const toast = vi.hoisted(() => ({ error: vi.fn() }))
const routeState = vi.hoisted(() => ({ pathname: "/task" }))
const navState = vi.hoisted(() => ({ groups: [] as unknown[] }))

vi.mock("next/navigation", () => ({ usePathname: () => routeState.pathname, useRouter: () => ({ push: vi.fn() }) }))
vi.mock("next/image", () => ({ default: (props: React.ImgHTMLAttributes<HTMLImageElement>) => <img {...props} /> }))
vi.mock("next/link", () => ({ default: ({ children, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => <a {...props}>{children}</a> }))
vi.mock("@/contexts/auth-context", () => ({ useAuth: () => authState }))
vi.mock("@/contexts/app-context-chat", () => ({ useApp: () => ({ state: { lastTaskUpdate: 0 } }) }))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))
vi.mock("@/components/ui/sonner", () => ({ toast }))
vi.mock("@/lib/branding", () => ({ getBrandingFromEnv: () => ({ appName: "Xagent" }) }))
vi.mock("@/lib/extra-nav", () => ({ default: [] }))
vi.mock("@/lib/sidebar-navigation", () => ({
  getNavigationGroupsForUser: () => navState.groups, getUserMenuItemsForUser: () => [],
}))

describe("Sidebar logout", () => {
  beforeEach(() => {
    authState.logout.mockReset()
    toast.error.mockReset()
    authState.logout.mockResolvedValue(false)
    routeState.pathname = "/task"
    navState.groups = []
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 500 })))
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    cleanup()
  })

  it("keeps the menu open and reports a localized failure when logout cannot clear auth", async () => {
    render(<Sidebar />)
    const userMenu = screen.getByRole("button", { name: /alice@example\.com/i })
    expect(screen.queryByText("acct_0123456789abcdef0123456789abcdef")).not.toBeInTheDocument()
    fireEvent.click(userMenu)
    fireEvent.click(screen.getByRole("button", { name: "sidebar.user.logoutTitle" }))
    await waitFor(() => expect(authState.logout).toHaveBeenCalledOnce())
    expect(toast.error).toHaveBeenCalledWith("sidebar.user.logoutFailed")
    expect(screen.getByRole("button", { name: "sidebar.user.logoutTitle" })).toBeInTheDocument()
  })
})

describe("Sidebar collapsible nav groups", () => {
  const RESOURCES_GROUP: NavigationGroup = {
    title: "Resources",
    defaultCollapsed: true,
    items: [{ name: "Knowledge Base", href: "/kb", icon: Globe }],
  }

  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 500 })))
    navState.groups = [RESOURCES_GROUP]
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    cleanup()
  })

  it("collapses a defaultCollapsed group by default and expands on click", () => {
    routeState.pathname = "/task"
    render(<Sidebar />)

    const header = screen.getByRole("button", { name: "Resources" })
    expect(header).toHaveAttribute("aria-expanded", "false")
    expect(screen.queryByRole("link", { name: "Knowledge Base" })).not.toBeInTheDocument()

    fireEvent.click(header)
    expect(header).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByRole("link", { name: "Knowledge Base" })).toBeInTheDocument()
  })

  it("auto-expands a defaultCollapsed group when it owns the active route", () => {
    routeState.pathname = "/kb"
    render(<Sidebar />)

    expect(screen.getByRole("button", { name: "Resources" })).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByRole("link", { name: "Knowledge Base" })).toBeInTheDocument()
  })

  it("keeps an explicit collapse even while the group owns the active route", () => {
    routeState.pathname = "/kb"
    render(<Sidebar />)

    const header = screen.getByRole("button", { name: "Resources" })
    expect(header).toHaveAttribute("aria-expanded", "true")

    fireEvent.click(header)
    expect(header).toHaveAttribute("aria-expanded", "false")
    expect(screen.queryByRole("link", { name: "Knowledge Base" })).not.toBeInTheDocument()
  })
})
