import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { Globe, Hash, Radio, Send, type LucideIcon } from "lucide-react"
import React from "react"
import { renderToStaticMarkup } from "react-dom/server"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { NavigationGroup, NavigationItem } from "@/lib/sidebar-navigation"
import { resolveTranslation, type TranslationKey } from "@/i18n/translations"

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
const navState = vi.hoisted(() => ({ groups: [] as unknown[], items: [] as NavigationItem[] }))
const extensionState = vi.hoisted(() => ({ groups: [] as NavigationGroup[] }))
const routerPush = vi.hoisted(() => vi.fn())
const appState = vi.hoisted(() => ({ lastTaskUpdate: 0 }))
const localeState = vi.hoisted(() => ({ locale: "en" as "en" | "zh" }))
const createSeededAgentTaskMock = vi.hoisted(() => vi.fn())

vi.mock("next/navigation", () => ({ usePathname: () => routeState.pathname, useRouter: () => ({ push: routerPush }) }))
vi.mock("next/image", () => ({ default: (props: React.ImgHTMLAttributes<HTMLImageElement>) => <img {...props} /> }))
vi.mock("next/link", () => ({ default: ({ children, onClick, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => <a {...props} onClick={(event) => { event.preventDefault(); onClick?.(event) }}>{children}</a> }))
vi.mock("@/contexts/auth-context", () => ({ useAuth: () => authState }))
vi.mock("@/contexts/app-context-chat", () => ({ useApp: () => ({ state: appState }) }))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ locale: localeState.locale, t: (key: TranslationKey, vars?: Record<string, string | number>) => key.startsWith("workspace.") || localeState.locale === "zh" ? resolveTranslation(localeState.locale, key, vars) : key }) }))
vi.mock("@/components/ui/sonner", () => ({ toast }))
vi.mock("@/lib/api-wrapper", () => ({ apiRequest: (input: RequestInfo | URL, init?: RequestInit) => globalThis.fetch(input, init) }))
vi.mock("@/lib/branding", () => ({ getBrandingFromEnv: () => ({ appName: "Xagent" }) }))
vi.mock("@/lib/extra-nav", () => ({ default: () => extensionState.groups }))
vi.mock("@/lib/sidebar-navigation", async () => ({
  ...await vi.importActual<typeof import("@/lib/sidebar-navigation")>("@/lib/sidebar-navigation"),
  getNavigationGroupsForUser: () => navState.groups, getUserMenuItemsForUser: () => navState.items,
}))
vi.mock("@/lib/hire-agent", () => ({ createSeededAgentTask: createSeededAgentTaskMock }))

describe("Sidebar review regressions", () => {
  beforeEach(() => {
    authState.user.id = "1"
    localeState.locale = "en"
    routeState.pathname = "/task/101"
    navState.groups = []
    routerPush.mockReset()
    toast.error.mockReset()
  })
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    authState.user.id = "1"
    localeState.locale = "en"
  })

  it.each([false, true])("discards old-identity search data (pending: %s) on account switch", async (pending) => {
    let release: (value: Response) => void = () => {}
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (authState.user.id === "2") return new Response(null, { status: 503 })
      if (url.includes("search=") && url.includes("/api/agents")) {
        if (pending) return new Promise<Response>((resolve) => { release = resolve })
        return Response.json([{ id: 8, name: "Private Alice employee", status: "published" }])
      }
      if (url.includes("/api/chat/tasks")) return Response.json({ tasks: [{ task_id: 101, title: "Private Alice conversation" }] })
      return Response.json(url.includes("/api/agents") ? [{ id: 7, name: "Alice teammate", status: "published" }] : [])
    })
    vi.stubGlobal("fetch", fetchMock)
    const view = render(<Sidebar />)
    await screen.findByText("Alice teammate")
    fireEvent.change(screen.getByRole("searchbox"), { target: { value: "private" } })
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).includes("/api/agents?search="))).toBe(true))
    if (!pending) await screen.findByText("Private Alice employee")
    authState.user.id = "2"
    view.rerender(<Sidebar />)
    expect(screen.getByRole("searchbox")).toHaveValue("")
    expect(screen.queryByText("Alice teammate")).not.toBeInTheDocument()
    expect(screen.queryByText("Private Alice employee")).not.toBeInTheDocument()
    expect(screen.queryByText("Private Alice conversation")).not.toBeInTheDocument()
    if (pending) await act(async () => { release(Response.json([{ id: 8, name: "Private Alice employee", status: "published" }])) })
    fireEvent.change(screen.getByRole("searchbox"), { target: { value: "private" } })
    await screen.findByRole("alert")
    expect(screen.queryByText("Private Alice employee")).not.toBeInTheDocument()
    expect(screen.queryByText("Private Alice conversation")).not.toBeInTheDocument()
  })

  it("renames a conversation, confirms deletion, and leaves a deleted current chat", async () => {
    const tasks = [{ task_id: 101, title: "Original conversation", created_at: "2026-10-20" }]
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (init?.method === "PUT") {
        tasks[0].title = JSON.parse(String(init.body)).title
        return Response.json({ success: true })
      }
      if (init?.method === "DELETE") return Response.json({ success: true })
      return Response.json(String(input).includes("/api/chat/tasks") ? { tasks } : [])
    })
    vi.stubGlobal("fetch", fetchMock)
    render(<Sidebar />)
    fireEvent.click(screen.getByRole("button", { name: "Show recent conversations" }))
    fireEvent.click(await screen.findByRole("button", { name: "Options for Original conversation" }))
    fireEvent.click(screen.getByRole("button", { name: "common.rename" }))
    const rename = screen.getByRole("dialog", { name: "Rename conversation" })
    fireEvent.change(within(rename).getByRole("textbox"), { target: { value: "Updated conversation" } })
    fireEvent.click(within(rename).getByRole("button", { name: "common.save" }))
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Rename conversation" })).not.toBeInTheDocument())
    expect(screen.getByRole("link", { name: /Updated conversation/ })).toHaveAttribute("href", "/task/101")
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining("/api/chat/task/101"), expect.objectContaining({ method: "PUT", body: JSON.stringify({ title: "Updated conversation" }) }))
    fireEvent.click(screen.getByRole("button", { name: "Options for Updated conversation" }))
    fireEvent.click(screen.getByRole("button", { name: "common.delete" }))
    let dialog = screen.getByRole("dialog", { name: "Delete conversation" })
    fireEvent.click(within(dialog).getByRole("button", { name: "common.cancel" }))
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(false)
    fireEvent.click(screen.getByRole("button", { name: "Options for Updated conversation" }))
    fireEvent.click(screen.getByRole("button", { name: "common.delete" }))
    dialog = screen.getByRole("dialog", { name: "Delete conversation" })
    fireEvent.click(within(dialog).getByRole("button", { name: "common.delete" }))
    await waitFor(() => expect(routerPush).toHaveBeenCalledWith("/task"))
    expect(screen.queryByRole("link", { name: /Updated conversation/ })).not.toBeInTheDocument()
  })

  it.each(["rename", "delete"])("keeps conversation data and the dialog when %s fails", async (kind) => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (init?.method) return new Response(null, { status: 500 })
      return Response.json(String(input).includes("/api/chat/tasks") ? { tasks: [{ task_id: 101, title: "Keep this conversation" }] } : [])
    }))
    render(<Sidebar />)
    fireEvent.click(screen.getByRole("button", { name: "Show recent conversations" }))
    fireEvent.click(await screen.findByRole("button", { name: "Options for Keep this conversation" }))
    fireEvent.click(screen.getByRole("button", { name: `common.${kind}` }))
    const dialog = screen.getByRole("dialog", { name: kind === "rename" ? "Rename conversation" : "Delete conversation" })
    if (kind === "rename") fireEvent.change(within(dialog).getByRole("textbox"), { target: { value: "New title" } })
    fireEvent.click(within(dialog).getByRole("button", { name: kind === "rename" ? "common.save" : "common.delete" }))
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(kind === "rename" ? "Could not rename conversation" : "common.deleteFailed"))
    expect(screen.getByRole("dialog", { name: kind === "rename" ? "Rename conversation" : "Delete conversation" })).toBeInTheDocument()
    expect(screen.getByText("Keep this conversation")).toBeInTheDocument()
    expect(routerPush).not.toHaveBeenCalled()
  })

  it.each([false, true])("distinguishes failed routine reads from an empty list (partial: %s)", async (partial) => {
    let retry = false
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith("/triggers")) {
        if (url.includes("/7/") || !partial && !retry) return new Response(null, { status: 503 })
        return Response.json([{ id: 81, type: "scheduled", name: "Morning report", enabled: true }])
      }
      if (url.endsWith("/api/agents")) return Response.json([
        { id: 7, name: "Ben", status: "published" },
        { id: 8, name: "Maya", status: "published" },
        { id: 9, name: "Shared", status: "published", readonly: true },
      ])
      return Response.json(url.includes("/api/chat/tasks") ? { tasks: [] } : [])
    })
    vi.stubGlobal("fetch", fetchMock)
    render(<Sidebar />)
    await screen.findByText("Ben")
    fireEvent.click(screen.getByRole("button", { name: "Recurring jobs" }))
    expect(await screen.findByRole("alert")).toHaveTextContent("existing schedules have not been stopped")
    expect(screen.queryByText("Nothing runs on a schedule yet.")).not.toBeInTheDocument()
    if (partial) expect(screen.getByRole("switch", { name: "Toggle Morning report" })).toBeChecked()
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("/9/triggers"))).toBe(false)
    retry = true
    fireEvent.click(screen.getByRole("button", { name: "common.retry" }))
    expect(await screen.findByRole("switch", { name: "Toggle Morning report" })).toBeChecked()
  })

  it("renders Chinese workspace, picker and routine labels", async () => {
    localeState.locale = "zh"
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => Response.json(String(input).includes("/api/chat/tasks") ? { tasks: [] } : [])))
    render(<Sidebar />)
    expect(screen.getByRole("searchbox")).toHaveAttribute("placeholder", "搜索员工和消息")
    expect(screen.getByRole("heading", { name: "我的团队" })).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "开始新工作" }))
    fireEvent.click(screen.getByRole("button", { name: "添加员工" }))
    expect(screen.getByRole("dialog", { name: "添加员工" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "创建" })).toBeInTheDocument()
    expect(screen.getByText("创建自己的员工")).toBeInTheDocument()
  })
})

describe("Sidebar conversation presentation", () => {
  beforeEach(() => {
    routeState.pathname = "/task/current"
    navState.groups = []
    vi.stubGlobal("fetch", vi.fn(async () => new Response(null, { status: 503 })))
  })
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it("renders active conversations, channel badges and agent portraits in Recent", async () => {
    const tasks = ["running", "completed", "failed", "paused", "waiting_for_user", "pending"].map((status, index) => ({
      task_id: index === 0 ? "current" : `task-${index}`, title: `Task ${status}`, status,
      channel_name: `Support ${index}`, channel_type: ["telegram", "slack", "feishu", "custom", "", undefined][index],
      ...(index === 0 ? { agent_id: 42, agent_logo_url: "/agent.svg" } : {}),
    }))
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => Response.json(
      String(input).includes("/api/chat/tasks") ? { tasks, pagination: { has_next: false } } : [],
    )))
    render(<Sidebar allowCollapse={false} />)
    fireEvent.click(screen.getByRole("button", { name: "Show recent conversations" }))
    for (const task of tasks) expect(await screen.findByRole("link", { name: new RegExp(task.title) })).toHaveAttribute("href", `/task/${task.task_id}`)
    expect(screen.getByRole("link", { name: /Task running/ }).querySelector("img")).toHaveAttribute("src", expect.stringContaining("/agent.svg"))
    expect(screen.getAllByText(/Support/)).toHaveLength(6)
    const channelIcons = [Send, Hash, null, Radio]
    const iconContents = (Icon: LucideIcon) => {
      const container = document.createElement("div")
      container.innerHTML = renderToStaticMarkup(<Icon />)
      return container.querySelector("svg")!.innerHTML
    }
    for (const [index, task] of tasks.entries()) {
      const link = screen.getByRole("link", { name: new RegExp(task.title) })
      expect(link.className.includes("bg-white")).toBe(index === 0)
      const channelBadge = link.querySelector("span[title]")!
      const ChannelIcon = channelIcons[index]
      if (ChannelIcon) expect(channelBadge.querySelector("svg")?.innerHTML).toBe(iconContents(ChannelIcon))
      else if (index === 2) expect(channelBadge.querySelector("img")).toHaveAttribute("src", "/icons/channels/feishu.svg")
      else expect(channelBadge.querySelector("svg, img")).toBeNull()
    }
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
    fireEvent.click(screen.getByRole("button", { name: /alice@example\.com/i }))

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
    fireEvent.click(screen.getByRole("button", { name: /alice@example\.com/i }))

    expect(screen.getByRole("button", { name: "Resources" })).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByRole("link", { name: "Knowledge Base" })).toBeInTheDocument()
  })

  it("keeps an explicit collapse even while the group owns the active route", () => {
    routeState.pathname = "/kb"
    render(<Sidebar />)
    fireEvent.click(screen.getByRole("button", { name: /alice@example\.com/i }))

    const header = screen.getByRole("button", { name: "Resources" })
    expect(header).toHaveAttribute("aria-expanded", "true")

    fireEvent.click(header)
    expect(header).toHaveAttribute("aria-expanded", "false")
    expect(screen.queryByRole("link", { name: "Knowledge Base" })).not.toBeInTheDocument()
  })
})

describe("Sidebar workspace menu", () => {
  beforeEach(async () => {
    const navigation = await vi.importActual<typeof import("@/lib/sidebar-navigation")>("@/lib/sidebar-navigation")
    navState.groups = navigation.getNavigationGroupsForUser()
    navState.items = navigation.getUserMenuItemsForUser()
    routeState.pathname = "/task"
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 500 })))
  })
  afterEach(() => {
    navState.items = []
    navState.groups = []
    extensionState.groups = []
    vi.unstubAllGlobals()
    cleanup()
  })

  it("shows common links above a single collapsible Builder and closes after navigation", () => {
    render(<Sidebar />)
    const trigger = screen.getByRole("button", { name: /alice@example\.com/i })
    expect(screen.queryByTestId("workspace-menu")).not.toBeInTheDocument()
    fireEvent.click(trigger)
    const menu = within(screen.getByTestId("workspace-menu"))
    expect(menu.getAllByRole("link").map((link) => link.getAttribute("href"))).toEqual([
      "/tools", "/skill-hub", "/files", "/settings",
    ])
    const builder = menu.getByRole("button", { name: "nav.builder" })
    expect(builder).toHaveAttribute("aria-expanded", "false")
    fireEvent.click(builder)
    expect(menu.getByRole("link", { name: "nav.teammates" })).toHaveAttribute("href", "/workforces?view=teammates")
    expect(menu.queryByText("nav.sections.resources")).not.toBeInTheDocument()
    expect(menu.queryByText("nav.sections.agentDevelopment")).not.toBeInTheDocument()
    fireEvent.click(menu.getByRole("link", { name: "nav.models" }))
    expect(trigger).toHaveAttribute("aria-expanded", "false")
    expect(screen.queryByTestId("workspace-menu")).not.toBeInTheDocument()
  })

  it("opens Builder and highlights its active child, including nested routes", () => {
    routeState.pathname = "/kb/42"
    render(<Sidebar />)
    fireEvent.click(screen.getByRole("button", { name: /alice@example\.com/i }))
    expect(screen.getByRole("button", { name: "nav.builder" })).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByRole("link", { name: "nav.knowledge" })).toHaveAttribute("aria-current", "page")
    fireEvent.keyDown(document, { key: "Escape" })
    expect(screen.queryByTestId("workspace-menu")).not.toBeInTheDocument()
  })

  it("preserves nested extension menus instead of flattening them", () => {
    navState.groups.push({
      title: "Extension", defaultCollapsed: true,
      items: [{ name: "Reports", href: "__reports__", icon: Globe, children: [
        { name: "Usage", href: "/reports/usage", icon: Globe },
      ] }],
    })
    routeState.pathname = "/reports/usage"
    render(<Sidebar />)
    fireEvent.click(screen.getByRole("button", { name: /alice@example\.com/i }))
    const reports = screen.getByRole("button", { name: "Reports" })
    expect(reports).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByRole("link", { name: "Usage" })).toHaveAttribute("aria-current", "page")
    expect(screen.queryByRole("link", { name: "Reports" })).not.toBeInTheDocument()
    fireEvent.click(reports)
    expect(screen.queryByRole("link", { name: "Usage" })).not.toBeInTheDocument()
  })

  it("reuses the core renderer for a top-level Team and Builder contributions", () => {
    extensionState.groups = [
      { id: "teams", title: "Team", placement: "workspace", icon: Globe, items: [
        { name: "People", href: "/teams", icon: Globe },
        { name: "Billing", href: "/billing", icon: Globe },
      ] },
      { id: "create-with-ai", title: "Create with AI", placement: "builder", items: [
        { name: "AI Slides", href: "/create-with-ai/slides", icon: Globe },
      ] },
    ]
    routeState.pathname = "/billing"
    render(<Sidebar />)
    fireEvent.click(screen.getByRole("button", { name: /alice@example\.com/i }))
    const menu = within(screen.getByTestId("workspace-menu"))
    expect(menu.getByRole("button", { name: "Team" })).toHaveAttribute("aria-expanded", "true")
    expect(menu.getByRole("link", { name: "Billing" })).toHaveAttribute("aria-current", "page")
    const builder = menu.getByRole("button", { name: "nav.builder" })
    expect(builder).toHaveAttribute("aria-expanded", "false")
    expect(menu.queryByRole("button", { name: "Create with AI" })).not.toBeInTheDocument()
    fireEvent.click(builder)
    expect(menu.getByRole("link", { name: "AI Slides" })).toHaveAttribute("href", "/create-with-ai/slides")
    expect(menu.getAllByRole("link", { name: "Billing" })).toHaveLength(1)
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

describe("Sidebar teammate conversations", () => {
  beforeEach(() => {
    appState.lastTaskUpdate = 0
    routerPush.mockReset()
    createSeededAgentTaskMock.mockReset()
    createSeededAgentTaskMock.mockResolvedValue({ taskId: 303, agentId: 9 })
    routeState.pathname = "/task"
    navState.groups = []
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes("/api/agents")) {
        return Response.json([
          {
            id: 7,
            name: "Maya",
            description: "Turn one brief into ready-to-publish campaigns",
            template_id: "social-media",
            status: "published",
          },
          {
            id: 8,
            name: "Custom Bot",
            description: "A custom teammate",
            template_id: null,
            status: "published",
          },
        ])
      }
      if (url.includes("/api/chat/tasks")) {
        return Response.json({
          tasks: [
            {
              task_id: 101,
              title: "Older chat",
              status: "completed",
              agent_id: 7,
              created_at: "2026-10-18T09:00:00Z",
              updated_at: "2026-10-18T09:30:00Z",
            },
            {
              task_id: 102,
              title: "Latest chat",
              status: "completed",
              agent_id: 7,
              created_at: "2026-10-17T09:00:00Z",
              updated_at: "2026-10-19T09:30:00Z",
            },
          ],
        })
      }
      if (url.includes("/api/templates/")) {
        return Response.json([
          {
            id: "social-media",
            name: "Social Media Content Manager",
            description: "Social publishing",
            category: "marketing",
            persona: {
              name: "Maya",
              role: "Social Media Content Manager",
              avatar: "/marketplace/avatars/maya.png",
              intro: "Hi, I'm Maya.",
              kickoff_questions: [],
            },
          },
        ])
      }
      return new Response(null, { status: 404 })
    }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    cleanup()
  })

  it("shows only the teammate name and category in the title row", async () => {
    render(<Sidebar />)

    expect(await screen.findByText("templates.categoryTitles.marketing")).toBeInTheDocument()
    expect(screen.queryByText("Turn one brief into ready-to-publish campaigns")).not.toBeInTheDocument()
    expect(screen.getByText("templates.categoryTitles.general")).toBeInTheDocument()
    expect(screen.queryByText("Add teammates")).not.toBeInTheDocument()
  })

  it("uses the employee's template avatar when no custom logo was uploaded", async () => {
    render(<Sidebar />)

    const maya = await screen.findByText("Maya")
    const row = maya.closest("a")
    expect(row?.querySelector("img")).toHaveAttribute("src", "/marketplace/avatars/maya.png")
  })

  it("searches the server after a debounce and loads older matching conversations", async () => {
    const baseFetch = globalThis.fetch
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), "http://localhost")
      if (url.searchParams.has("search")) {
        if (url.pathname === "/api/agents") return Response.json([{ id: 8, name: "Custom Bot", status: "published" }])
        const page = Number(url.searchParams.get("page"))
        return Response.json({
          tasks: [{ task_id: 800 + page, title: `Archived conversation ${page}`, created_at: "2025-01-01", status: "completed" }],
          pagination: { has_next: page === 1 },
        })
      }
      return baseFetch(input, init)
    })
    vi.stubGlobal("fetch", fetchMock)
    render(<Sidebar />)
    await screen.findByText("Maya")
    const input = screen.getByRole("searchbox")
    fireEvent.change(input, { target: { value: "refund" } })
    fireEvent.change(input, { target: { value: "refund & tax" } })
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("search="))).toBe(false)
    await screen.findByText("Archived conversation 1")
    expect(screen.queryByText("Maya")).not.toBeInTheDocument()
    expect(screen.getByText("Custom Bot")).toBeInTheDocument()
    const searchCalls = fetchMock.mock.calls.filter(([url]) => String(url).includes("search="))
    expect(searchCalls).toHaveLength(2)
    expect(searchCalls.every(([url]) => new URL(String(url), "http://localhost").searchParams.get("search") === "refund & tax")).toBe(true)
    fireEvent.click(screen.getByRole("button", { name: "sidebar.search.loadMore" }))
    await screen.findByText("Archived conversation 2")
    expect(screen.getByText("Archived conversation 1").closest("a")).toHaveAttribute("href", "/task/801")
    fireEvent.click(screen.getByRole("button", { name: "Clear search" }))
    expect(screen.getByText("Maya")).toBeInTheDocument()
    expect(screen.queryByText("Archived conversation 1")).not.toBeInTheDocument()
  })

  it("ignores a late response after the keyword changes or is cleared", async () => {
    const baseFetch = globalThis.fetch
    let resolveOld: (response: Response) => void = () => {}
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), "http://localhost")
      if (url.searchParams.get("search") === "old" && url.pathname === "/api/agents") {
        return new Promise<Response>((resolve) => { resolveOld = resolve })
      }
      if (url.searchParams.has("search")) {
        return Response.json(url.pathname === "/api/agents" ? [] : { tasks: [] })
      }
      return baseFetch(input, init)
    })
    vi.stubGlobal("fetch", fetchMock)
    render(<Sidebar />)
    await screen.findByText("Maya")
    fireEvent.change(screen.getByRole("searchbox"), { target: { value: "old" } })
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).includes("search=old"))).toBe(true))
    fireEvent.change(screen.getByRole("searchbox"), { target: { value: "new" } })
    await screen.findByText("sidebar.search.empty")
    await act(async () => { resolveOld(Response.json([{ id: 99, name: "Stale result", status: "published" }])) })
    expect(screen.queryByText("Stale result")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Clear search" }))
    expect(screen.getByText("Maya")).toBeInTheDocument()
  })

  it("reports search failures instead of presenting an empty success result", async () => {
    const baseFetch = globalThis.fetch
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) =>
      String(input).includes("search=") ? new Response(null, { status: 500 }) : baseFetch(input, init)
    ))
    render(<Sidebar />)
    await screen.findByText("Maya")
    fireEvent.change(screen.getByRole("searchbox"), { target: { value: "refund" } })
    expect(await screen.findByRole("alert")).toHaveTextContent("sidebar.search.failed")
    expect(screen.queryByText("sidebar.search.empty")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Clear search" }))
    expect(screen.getByText("Maya")).toBeInTheDocument()
  })

  it("links a teammate to their most recently active conversation", async () => {
    render(<Sidebar />)

    const latestChat = await screen.findByText("Latest chat")
    expect(latestChat.closest("a")).toHaveAttribute("href", "/task/102")
    expect(screen.queryByText("Older chat")).not.toBeInTheDocument()

    const customBot = screen.getByText("Custom Bot")
    expect(customBot.closest("a")).toHaveAttribute("href", "/task?agent=8")
    expect(screen.getByText("Just joined — ready for your first job")).toBeInTheDocument()
  })

  it("ignores an older workspace response after a hire refresh", async () => {
    const baseFetch = globalThis.fetch
    let finishOld!: (response: Response) => void
    let agentCalls = 0
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/agents")) {
        agentCalls += 1
        if (agentCalls === 1) return new Promise<Response>((resolve) => { finishOld = resolve })
        return Promise.resolve(Response.json([{ id: 10, name: "Kevin", status: "published" }]))
      }
      return baseFetch(input, init)
    }))
    const { rerender } = render(<Sidebar />)
    await waitFor(() => expect(agentCalls).toBe(1))
    appState.lastTaskUpdate = 1
    rerender(<Sidebar />)
    expect(await screen.findByText("Kevin")).toBeInTheDocument()
    await act(async () => { finishOld(Response.json([{ id: 7, name: "Outdated teammate", status: "published" }])) })
    expect(screen.getByText("Kevin")).toBeInTheDocument()
    expect(screen.queryByText("Outdated teammate")).not.toBeInTheDocument()
  })

  it("finds a conversation outside the loaded page instead of seeding another intro", async () => {
    const baseFetch = globalThis.fetch
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).includes("agent_id=8")) return Promise.resolve(Response.json({
        tasks: [{ task_id: 88, agent_id: 8, title: "Archived conversation" }],
      }))
      return baseFetch(input, init)
    }))
    render(<Sidebar />)
    fireEvent.click((await screen.findByText("Custom Bot")).closest("a")!)
    await waitFor(() => expect(routerPush).toHaveBeenCalledWith("/task/88"))
    expect(createSeededAgentTaskMock).not.toHaveBeenCalled()
  })

  it("lets Recent reach conversations beyond the first page", async () => {
    const baseFetch = globalThis.fetch
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), "http://localhost")
      if (url.pathname === "/api/chat/tasks") {
        const page = Number(url.searchParams.get("page"))
        return Promise.resolve(Response.json({
          tasks: [{ task_id: page, title: `Conversation page ${page}`, created_at: "2026-10-01" }],
          pagination: { has_next: page === 1 },
        }))
      }
      return baseFetch(input, init)
    }))
    render(<Sidebar />)
    await screen.findByText("Maya")
    fireEvent.click(screen.getByRole("button", { name: "Show recent conversations" }))
    expect(await screen.findByText("Conversation page 1")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "sidebar.search.loadMore" }))
    expect(await screen.findByText("Conversation page 2")).toBeInTheDocument()
    expect(screen.getByText("Conversation page 1")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "sidebar.search.loadMore" })).not.toBeInTheDocument()
  })

  it("does not create a new conversation when the history lookup fails", async () => {
    const baseFetch = globalThis.fetch
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).includes("agent_id=7")) return Promise.resolve(new Response(null, { status: 503 }))
      return baseFetch(input, init)
    }))
    render(<Sidebar />)
    fireEvent.click((await screen.findByText("Maya")).closest("a")!)
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith("chatPage.page.startConversationFailed"))
    expect(createSeededAgentTaskMock).not.toHaveBeenCalled()
    expect(routerPush).not.toHaveBeenCalled()
  })

  it("opens an unmessaged template teammate with the same seeded first conversation as hiring", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes("/api/agents")) {
        return Response.json([{ id: 9, name: "Ben", template_id: "support-email", status: "published" }])
      }
      if (url.includes("/api/chat/tasks")) return Response.json({ tasks: [] })
      if (url.includes("/api/templates/")) {
        return Response.json([{
          id: "support-email",
          name: "Support Email Specialist",
          description: "Answers support email",
          category: "support",
          connections: [{ name: "Gmail" }],
          persona: {
            name: "Ben",
            role: "Support Email Specialist",
            intro: "Hi there — I'm Ben, your Support Email Specialist.",
            kickoff_questions: ["Which inbox am I working?"],
          },
        }])
      }
      return new Response(null, { status: 404 })
    }))

    render(<Sidebar />)
    const ben = await screen.findByText("Ben")
    fireEvent.click(ben.closest("a")!)

    await waitFor(() => expect(createSeededAgentTaskMock).toHaveBeenCalledWith({
      agentId: 9,
      persona: expect.objectContaining({ name: "Ben", role: "Support Email Specialist" }),
      connections: [{ name: "Gmail" }],
      strings: {
        beforeWeStart: "templates.marketplace.beforeWeStart",
        closingNote: "templates.marketplace.hireClosingNote",
        connectAppsLabel: "chatPage.clarification.connectApps.title",
      },
    }))
    expect(routerPush).toHaveBeenCalledWith("/task/303")
    expect((await screen.findByText("Ben — Support Email Specialist")).closest("a")).toHaveAttribute("href", "/task/303")
  })
})
