import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { CheckCircle2, Globe, Hash, Loader2, PauseCircle, Radio, Send, XCircle, type LucideIcon } from "lucide-react"
import React from "react"
import { renderToStaticMarkup } from "react-dom/server"
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
const apiRequestMock = vi.hoisted(() => vi.fn(async () => new Response(null, { status: 401 })))

vi.mock("next/navigation", () => ({ usePathname: () => routeState.pathname, useRouter: () => ({ push: vi.fn() }) }))
vi.mock("next/image", () => ({ default: (props: React.ImgHTMLAttributes<HTMLImageElement>) => <img {...props} /> }))
vi.mock("next/link", () => ({ default: ({ children, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => <a {...props}>{children}</a> }))
vi.mock("@/contexts/auth-context", () => ({ useAuth: () => authState }))
vi.mock("@/contexts/app-context-chat", () => ({ useApp: () => ({ state: { lastTaskUpdate: 0 } }) }))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))
vi.mock("@/components/ui/sonner", () => ({ toast }))
vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))
vi.mock("@/lib/branding", () => ({ getBrandingFromEnv: () => ({ appName: "Xagent" }) }))
vi.mock("@/lib/extra-nav", () => ({ default: [] }))
vi.mock("@/lib/sidebar-navigation", () => ({
  getNavigationGroupsForUser: () => navState.groups, getUserMenuItemsForUser: () => [],
}))

describe("Sidebar task and release presentation", () => {
  beforeEach(() => {
    routeState.pathname = "/task/current"
    navState.groups = []
    apiRequestMock.mockReset()
    apiRequestMock.mockImplementation(async () => new Response(JSON.stringify({ tasks: [], pagination: { total_pages: 1 } })))
    vi.stubGlobal("fetch", vi.fn(async () => new Response(null, { status: 503 })))
  })
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    apiRequestMock.mockImplementation(async () => new Response(null, { status: 401 }))
  })

  it("renders active task status, channel badges and agent icons from the task list", async () => {
    const tasks = ["running", "completed", "failed", "paused", "waiting_for_user", "pending"].map((status, index) => ({
      task_id: index === 0 ? "current" : `task-${index}`, title: `Task ${status}`, status,
      channel_name: `Support ${index}`, channel_type: ["telegram", "slack", "feishu", "custom", "", undefined][index],
      ...(index === 0 ? { agent_id: 42, agent_logo_url: "/agent.svg" } : {}),
    }))
    apiRequestMock.mockImplementation(async () => new Response(JSON.stringify({ tasks, pagination: { total_pages: 1 } })))
    render(<Sidebar allowCollapse={false} />)
    for (const task of tasks) expect(await screen.findByRole("link", { name: new RegExp(task.title) })).toHaveAttribute("href", `/task/${task.task_id}`)
    expect(screen.getByAltText("Agent Logo")).toHaveAttribute("src", expect.stringContaining("/agent.svg"))
    expect(screen.getAllByText(/Support/)).toHaveLength(6)
    const statusIcons = [Loader2, CheckCircle2, XCircle, PauseCircle, PauseCircle, Loader2]
    const channelIcons = [Send, Hash, null, Radio]
    const iconContents = (Icon: LucideIcon) => {
      const container = document.createElement("div")
      container.innerHTML = renderToStaticMarkup(<Icon />)
      return container.querySelector("svg")!.innerHTML
    }
    for (const [index, task] of tasks.entries()) {
      const link = screen.getByRole("link", { name: new RegExp(task.title) })
      expect(link.querySelector(".opacity-0 > svg")?.innerHTML).toBe(iconContents(statusIcons[index]))
      expect(link.className.includes("bg-primary/[0.09]")).toBe(index === 0)
      const channelBadge = link.querySelector("span[aria-label]")!
      const ChannelIcon = channelIcons[index]
      if (ChannelIcon) expect(channelBadge.querySelector("svg")?.innerHTML).toBe(iconContents(ChannelIcon))
      else if (index === 2) expect(channelBadge.querySelector("img")).toHaveAttribute("src", "/icons/channels/feishu.svg")
      else expect(channelBadge.querySelector("svg, img")).toBeNull()
    }
  })

  it.each([
    { stars: 42, formatted: "42", latest: true, status: "versionLatest" },
    { stars: 1200, formatted: "1.2k", latest: false, status: "versionUpdateAvailable" },
    { stars: 2400000, formatted: "2.4M", latest: null, status: "versionStatusUnknown" },
  ])("shows release status and formats $stars repository stars", async ({ stars, formatted, latest, status }) => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify(
      url.includes("api.github.com") ? { stargazers_count: stars } : { version: "1.2.3", display_version: "v1.2.3", is_latest: latest },
    ))))
    render(<Sidebar allowCollapse={false} />)
    fireEvent.click(screen.getByRole("button", { name: /alice@example\.com/i }))
    fireEvent.click(screen.getByRole("button", { name: "sidebar.about.menu" }))
    expect(await screen.findByText(`sidebar.about.stars: ${formatted}`)).toBeInTheDocument()
    expect(screen.getByTitle(`sidebar.about.${status}`)).toBeInTheDocument()
    expect(screen.getByText(/v1\.2\.3/)).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "sidebar.about.license" })).toHaveAttribute("rel", "noopener noreferrer")
  })
})

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

describe("Sidebar profile subtitle", () => {
  beforeEach(() => {
    routeState.pathname = "/task"
    navState.groups = []
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 500 })))
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    cleanup()
  })

  it("leaves the profile unlabeled without a subtitle", () => {
    render(<Sidebar />)

    expect(screen.getByText("alice@example.com")).toBeInTheDocument()
  })

  it("shows the host-supplied subtitle under the signed-in user", () => {
    render(<Sidebar profileSubtitle="Singapore" />)

    expect(screen.getByText("alice@example.com")).toBeInTheDocument()
    expect(screen.getByText("Singapore")).toBeInTheDocument()
  })

  it("treats a blank subtitle as absent", () => {
    // /agent collapses the rail, so the profile shows only its avatar.
    routeState.pathname = "/agent"
    render(<Sidebar profileSubtitle="   " />)

    const profile = screen.getByRole("button", { name: /alice@example\.com/ })
    expect(profile.getAttribute("title")).not.toContain("\u00b7")
  })

  it("announces the subtitle from the collapsed profile rail", () => {
    // The rail has no room for the name and subtitle, so both are announced.
    routeState.pathname = "/agent"
    render(<Sidebar profileSubtitle="Singapore" />)

    const profile = screen.getByRole("button", { name: /alice@example\.com.*Singapore/ })
    expect(profile.getAttribute("title")).toMatch(/alice@example\.com.*Singapore/)
  })

  it("never reads the host deployment configuration", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 500 }))
    vi.stubGlobal("fetch", fetchMock)
    render(<Sidebar profileSubtitle="Singapore" />)

    await act(async () => {})

    // Deployment configuration belongs to the hosting app, not to core.
    const requested = fetchMock.mock.calls.map(([input]) => String(input))
    expect(requested).not.toContain("/api/deployment-config")
  })
})
