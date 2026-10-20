"use client"

import Image from "next/image"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  Bot,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronsUpDown,
  Clock3,
  FileText,
  Hash,
  Loader2,
  LogOut,
  Plus,
  Radio,
  RotateCcw,
  Search,
  Send,
  Settings2,
  Sparkles,
  Wrench,
  X,
} from "lucide-react"

import { apiRequest } from "@/lib/api-wrapper"
import {
  type AgentTrigger,
  listOwnerTriggers,
  updateOwnerTrigger,
} from "@/lib/agent-triggers-api"
import { getBrandingFromEnv } from "@/lib/branding"
import { getCompactChannelName, getChannelTooltip } from "@/lib/channel-display"
import extraNav from "@/lib/extra-nav"
import { categoryLabel } from "@/lib/template-categories"
import { createSeededAgentTask } from "@/lib/hire-agent"
import { getAgentTaskHref } from "@/lib/agent-ui-access"
import {
  composeSidebarNavigation,
  getNavigationGroupsForUser,
  getUserMenuItemsForUser,
  type NavigationGroup,
  type NavigationItem,
} from "@/lib/sidebar-navigation"
import { userDisplayLabel } from "@/lib/user-display"
import { cn, getApiUrl, resolveAgentLogoUrl } from "@/lib/utils"
import { useAuth } from "@/contexts/auth-context"
import { useApp } from "@/contexts/app-context-chat"
import { useI18n, type Translate } from "@/contexts/i18n-context"
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Switch } from "@/components/ui/switch"
import { PersonaAvatar } from "@/components/templates/persona-avatar"
import { toast } from "@/components/ui/sonner"
import type { Template } from "@/types/template"

export {
  getNavigationGroupsForUser,
  getUserMenuItemsForUser,
}
export type {
  NavigationGroup,
  NavigationItem,
}

interface Task {
  task_id: string | number
  title: string
  status: "completed" | "running" | "failed" | "pending" | "paused" | "waiting_for_user"
  created_at: string | number
  updated_at?: string | number
  description?: string
  agent_id?: number
  agent_name?: string
  agent_logo_url?: string
  channel_name?: string
  channel_type?: string
}

interface AgentSummary {
  id: number
  name: string
  description?: string | null
  logo_url?: string | null
  template_id?: string | null
  status?: string
  readonly?: boolean
  can_edit?: boolean
}

interface TemplateSummary {
  id: string
  name: string
  description: string
  category?: string
  featured?: boolean
  persona?: { name?: string; avatar?: string | null; role?: string }
  hired?: boolean
  hired_agent_id?: number | null
  type?: string
}

interface SidebarProps {
  isCollapsible?: boolean
  className?: string
  allowCollapse?: boolean
  profileSubtitle?: string
}

const TASKS_PER_PAGE = 40

const CHANNEL_ICON_PATHS: Record<string, string> = {
  feishu: "/icons/channels/feishu.svg",
}

function ChannelTypeIcon({ channelType }: { channelType?: string }) {
  const normalizedType = channelType?.trim().toLowerCase()
  if (!normalizedType) return null
  if (normalizedType === "telegram") {
    return <Send className="h-3 w-3 shrink-0 text-[#229ED9]" aria-hidden="true" />
  }
  if (normalizedType === "slack") {
    return <Hash className="h-3 w-3 shrink-0 text-[#4A154B]" aria-hidden="true" />
  }
  const iconPath = CHANNEL_ICON_PATHS[normalizedType]
  if (iconPath) {
    return <Image src={iconPath} alt="" width={12} height={12} className="h-3 w-3 shrink-0" unoptimized aria-hidden="true" />
  }
  return <Radio className="h-3 w-3 shrink-0" aria-hidden="true" />
}

function compactTime(value: string | number): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ""
  const now = new Date()
  if (date.toDateString() === now.toDateString()) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  }
  const yesterday = new Date(now)
  yesterday.setDate(now.getDate() - 1)
  if (date.toDateString() === yesterday.toDateString()) return "Yesterday"
  return date.toLocaleDateString([], { month: "short", day: "numeric" })
}

function taskPresence(status?: Task["status"]) {
  if (status === "running" || status === "pending") return { dot: "bg-[#6477f4]", ring: "ring-[#aeb8ff]", label: "Working" }
  if (status === "paused" || status === "waiting_for_user") return { dot: "bg-[#efaa36]", ring: "ring-[#f6cb82]", label: "Waiting" }
  if (status === "completed") return { dot: "bg-[#2eb67d]", ring: "ring-[#9cdec4]", label: "Done" }
  if (status === "failed") return { dot: "bg-[#dc5a5a]", ring: "ring-[#efadad]", label: "Needs attention" }
  return { dot: "bg-[#a8a8b2]", ring: "ring-transparent", label: "Available" }
}

function SectionMenu({
  group,
  pathname,
  t,
  onNavigate,
}: {
  group: NavigationGroup
  pathname: string
  t: Translate
  onNavigate?: () => void
}) {
  const ownsActiveRoute = group.items.some((item) => isNavigationItemActive(item, pathname))
  const [collapsedOverride, setCollapsedOverride] = useState<boolean | null>(null)
  const collapsed = collapsedOverride ?? Boolean(group.defaultCollapsed && !ownsActiveRoute)
  const title = group.titleKey ? t(group.titleKey) : group.title
  const contentId = React.useId()

  return (
    <div className="px-2 py-1">
      <button
        type="button"
        aria-label={title}
        aria-expanded={!collapsed}
        aria-controls={contentId}
        onClick={() => setCollapsedOverride(!collapsed)}
        className={cn("flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-[12px] hover:bg-accent", !collapsed && "bg-muted/70")}
      >
        {group.icon && <group.icon className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />}
        <span className="min-w-0 flex-1">
          <span className="block font-medium">{title}</span>
          {group.descriptionKey && <span className="mt-0.5 block text-[10px] leading-relaxed text-muted-foreground">{t(group.descriptionKey)}</span>}
        </span>
        {collapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
      </button>
      {!collapsed && (
        <div id={contentId} className="ml-3 mt-1 space-y-0.5 border-l border-border/60 pl-2">
          {group.items.map((item) => <WorkspaceMenuItem key={item.href} item={item} pathname={pathname} t={t} onNavigate={onNavigate} />)}
        </div>
      )}
    </div>
  )
}

function isNavigationItemActive(item: NavigationItem, pathname: string): boolean {
  if (item.children?.length) return item.children.some((child) => isNavigationItemActive(child, pathname))
  const itemPath = item.href.split("?")[0]
  return pathname === itemPath || pathname.startsWith(`${itemPath}/`)
}

function WorkspaceMenuItem({ item, pathname, t, onNavigate }: {
  item: NavigationItem
  pathname: string
  t: Translate
  onNavigate?: () => void
}) {
  if (item.children?.length) {
    return <SectionMenu group={{ title: item.name, titleKey: item.nameKey, descriptionKey: item.descriptionKey, icon: item.icon, items: item.children, defaultCollapsed: true }} pathname={pathname} t={t} onNavigate={onNavigate} />
  }
  const active = isNavigationItemActive(item, pathname)
  return (
    <Link href={item.href} onClick={onNavigate} aria-current={active ? "page" : undefined}
      className={cn("flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[12px] transition-colors", active ? "bg-[#eef1fe] font-semibold text-[#2536e0]" : "text-foreground hover:bg-accent")}>
      <item.icon className={cn("h-3.5 w-3.5 shrink-0", !active && "text-muted-foreground")} />
      <span className="min-w-0">
        <span className="block">{item.nameKey ? t(item.nameKey) : item.name}</span>
        {item.descriptionKey && <span className="mt-0.5 block text-[10px] leading-relaxed text-muted-foreground">{t(item.descriptionKey)}</span>}
      </span>
    </Link>
  )
}

function TeammatePickerDialog({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const router = useRouter()
  const { locale } = useI18n()
  const [templates, setTemplates] = useState<TemplateSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [category, setCategory] = useState("All")

  useEffect(() => {
    if (!open) return
    let cancelled = false
    setLoading(true)
    apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`)
      .then(async (response) => response.ok ? response.json() : [])
      .then((data: unknown) => {
        if (!cancelled) setTemplates(Array.isArray(data) ? data.filter((item): item is TemplateSummary => Boolean(item && typeof item === "object" && "id" in item && "name" in item)) : [])
      })
      .catch(() => { if (!cancelled) setTemplates([]) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [locale, open])

  const agentTemplates = templates.filter((template) => (template.type || "agent") === "agent")
  const categories = ["All", ...Array.from(new Set(agentTemplates.map((template) => template.category).filter(Boolean) as string[]))]
  const visibleTemplates = agentTemplates.filter((template) => category === "All" || template.category === category).slice(0, 7)

  const go = (href: string) => {
    onOpenChange(false)
    router.push(href)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[min(520px,calc(100vw-24px))] max-w-none gap-0 overflow-hidden rounded-[20px] border-border bg-background p-0 shadow-2xl">
        <div className="border-b border-border px-5 pb-4 pt-5">
          <div className="text-[9px] font-bold uppercase tracking-[0.18em] text-muted-foreground">XAGENT</div>
          <DialogTitle className="mt-1 text-xl font-semibold tracking-tight">Add a teammate</DialogTitle>
          <p className="mt-1 text-sm text-muted-foreground">Pick the job you want done and the teammate arrives configured.</p>
        </div>
        <div className="max-h-[62vh] overflow-y-auto p-4">
          <div className="flex items-center gap-3 rounded-2xl border border-[#d9defb] bg-[#f7f8ff] p-3">
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-white text-[#2536e0] shadow-sm"><Sparkles className="h-4 w-4" /></span>
            <div className="min-w-0 flex-1">
              <b className="block text-sm font-semibold">Build your own teammate</b>
              <span className="block text-xs text-muted-foreground">Describe the job, or configure one field by field.</span>
            </div>
            <button onClick={() => go("/build?create=true")} className="inline-flex h-9 items-center gap-1.5 rounded-full bg-[#2536e0] px-4 text-xs font-semibold text-white hover:bg-[#1e2fdb]">
              <Sparkles className="h-3.5 w-3.5" /> Build
            </button>
          </div>

          <div className="my-4 flex gap-1 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
            {categories.slice(0, 7).map((item) => (
              <button key={item} onClick={() => setCategory(item)} className={cn("rounded-full px-3 py-1.5 text-xs font-medium", category === item ? "bg-[#eef1fe] text-[#2536e0]" : "text-muted-foreground hover:bg-accent")}>{item}</button>
            ))}
          </div>

          {loading ? (
            <div className="grid place-items-center py-12"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>
          ) : (
            <div className="space-y-1">
              {visibleTemplates.map((template) => {
                const target = template.hired && template.hired_agent_id ? getAgentTaskHref(template.hired_agent_id) : `/templates/${encodeURIComponent(template.id)}`
                return (
                  <button key={template.id} onClick={() => go(target)} className="group flex w-full items-center gap-3 rounded-xl px-2.5 py-2.5 text-left hover:bg-accent">
                    <PersonaAvatar persona={{ name: template.persona?.name || template.name, avatar: template.persona?.avatar }} sizeClassName="h-9 w-9" textClassName="text-xs" decorative />
                    <span className="min-w-0 flex-1">
                      <b className="flex items-center gap-2 text-[13px] font-semibold"><span className="truncate">{template.persona?.name || template.name}</span>{template.featured && <span className="rounded-full bg-[#eef1fe] px-2 py-0.5 text-[8px] font-bold uppercase tracking-wide text-[#2536e0]">Featured</span>}</b>
                      <span className="mt-0.5 block truncate text-xs text-muted-foreground">{template.persona?.role || template.description}</span>
                    </span>
                    <ChevronRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
                  </button>
                )
              })}
            </div>
          )}
        </div>
        <div className="border-t border-border p-3">
          <button onClick={() => go("/templates")} className="flex h-9 w-full items-center justify-center gap-2 rounded-full border border-border text-xs font-semibold hover:bg-accent">See all teammates <ChevronRight className="h-3.5 w-3.5" /></button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function RoutinesDialog({
  open,
  onOpenChange,
  agents,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  agents: AgentSummary[]
}) {
  const [items, setItems] = useState<Array<{ agent: AgentSummary; trigger: AgentTrigger }>>([])
  const [loading, setLoading] = useState(false)
  const [busyId, setBusyId] = useState<number | null>(null)

  const load = useCallback(async () => {
    if (!open) return
    setLoading(true)
    const results = await Promise.allSettled(
      agents.filter((agent) => agent.status === "published").map(async (agent) => ({
        agent,
        triggers: await listOwnerTriggers({ kind: "agent", id: agent.id }),
      })),
    )
    const next = results.flatMap((result) => result.status === "fulfilled"
      ? result.value.triggers.filter((trigger) => trigger.type === "scheduled").map((trigger) => ({ agent: result.value.agent, trigger }))
      : [])
    setItems(next)
    setLoading(false)
  }, [agents, open])

  useEffect(() => { void load() }, [load])

  const toggle = async (entry: { agent: AgentSummary; trigger: AgentTrigger }) => {
    setBusyId(entry.trigger.id)
    try {
      const updated = await updateOwnerTrigger({ kind: "agent", id: entry.agent.id }, entry.trigger.id, { enabled: !entry.trigger.enabled })
      setItems((current) => current.map((item) => item.trigger.id === updated.id && item.agent.id === entry.agent.id ? { ...item, trigger: updated } : item))
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not update recurring job")
    } finally {
      setBusyId(null)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[min(500px,calc(100vw-24px))] max-w-none gap-0 overflow-hidden rounded-[20px] p-0">
        <div className="border-b border-border px-5 pb-4 pt-5">
          <div className="text-[9px] font-bold uppercase tracking-[0.18em] text-muted-foreground">XAGENT</div>
          <DialogTitle className="mt-1 text-xl font-semibold">Recurring jobs</DialogTitle>
          <p className="mt-1 text-sm text-muted-foreground">Standing work, and the switch that stops it.</p>
        </div>
        <div className="max-h-[62vh] overflow-y-auto p-4">
          {loading ? (
            <div className="grid place-items-center py-12"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>
          ) : items.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-border px-6 py-10 text-center">
              <Clock3 className="mx-auto h-6 w-6 text-muted-foreground" />
              <p className="mt-3 text-sm font-medium">Nothing runs on a schedule yet.</p>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">Ask a teammate to do something every morning and it will appear here.</p>
            </div>
          ) : (
            <div className="space-y-2">
              {items.map((entry) => (
                <div key={`${entry.agent.id}:${entry.trigger.id}`} className="flex items-center gap-3 rounded-xl border border-border p-3">
                  <PersonaAvatar persona={{ name: entry.agent.name, avatar: resolveAgentLogoUrl(entry.agent.logo_url ?? null, getApiUrl()) }} sizeClassName="h-9 w-9" textClassName="text-xs" decorative />
                  <div className="min-w-0 flex-1">
                    <b className="block truncate text-sm font-semibold">{entry.trigger.name}</b>
                    <span className="block truncate text-xs text-muted-foreground">{entry.agent.name}{entry.trigger.next_run_at ? ` · ${new Date(entry.trigger.next_run_at).toLocaleString()}` : ""}</span>
                  </div>
                  {busyId === entry.trigger.id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Switch checked={entry.trigger.enabled} onCheckedChange={() => void toggle(entry)} aria-label={`Toggle ${entry.trigger.name}`} />}
                </div>
              ))}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

export function Sidebar({ className, allowCollapse = true, profileSubtitle }: SidebarProps) {
  const pathname = usePathname()
  const router = useRouter()
  const { user, logout } = useAuth()
  const { state } = useApp()
  const { locale, t } = useI18n()
  const branding = getBrandingFromEnv()
  const userLabel = userDisplayLabel(user, t("sidebar.user.defaultName"))
  const subtitle = profileSubtitle?.trim() || null
  const profileLabel = subtitle ? `${userLabel} · ${subtitle}` : userLabel
  const { groups: navigationGroups, userItems: userMenuItems } = useMemo(() => {
    const extra = typeof extraNav === "function" ? extraNav(user) : extraNav
    return composeSidebarNavigation(getNavigationGroupsForUser(user), getUserMenuItemsForUser(user), extra)
  }, [user])

  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [view, setView] = useState<"team" | "recent">("team")
  const [search, setSearch] = useState("")
  const searchQuery = search.trim()
  const [searchPage, setSearchPage] = useState(1)
  const [searchAttempt, setSearchAttempt] = useState(0)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState(false)
  const [searchResults, setSearchResults] = useState<{
    query: string; agents: AgentSummary[]; tasks: Task[]; hasNext: boolean
  }>({ query: "", agents: [], tasks: [], hasNext: false })
  const [startOpen, setStartOpen] = useState(false)
  const [userMenuOpen, setUserMenuOpen] = useState(false)
  const [pickerOpen, setPickerOpen] = useState(false)
  const [routinesOpen, setRoutinesOpen] = useState(false)
  const [agents, setAgents] = useState<AgentSummary[]>([])
  const [tasks, setTasks] = useState<Task[]>([])
  const [taskPage, setTaskPage] = useState(1)
  const [tasksHasNext, setTasksHasNext] = useState(false)
  const [loadingMoreTasks, setLoadingMoreTasks] = useState(false)
  const loadingMoreTasksRef = useRef(false)
  const [templatesById, setTemplatesById] = useState<Record<string, Template>>({})
  const [openingAgentId, setOpeningAgentId] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const userMenuRef = useRef<HTMLDivElement>(null)
  const openingAgentRef = useRef<number | null>(null)

  const workspaceVersion = useRef(0)
  useEffect(() => {
    const version = ++workspaceVersion.current
    const current = () => version === workspaceVersion.current
    setLoading(true)
    setTaskPage(1)
    setTasksHasNext(false)
    loadingMoreTasksRef.current = false
    setLoadingMoreTasks(false)
    void (async () => {
      const [agentResult, taskResult, templateResult] = await Promise.allSettled([
        apiRequest(`${getApiUrl()}/api/agents`),
        apiRequest(`${getApiUrl()}/api/chat/tasks?page=1&per_page=${TASKS_PER_PAGE}`),
        apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`),
      ])
      if (agentResult.status === "fulfilled" && agentResult.value.ok) {
        const data = await agentResult.value.json().catch(() => [])
        if (current()) setAgents(Array.isArray(data) ? data : [])
      }
      if (taskResult.status === "fulfilled" && taskResult.value.ok) {
        const data = await taskResult.value.json().catch(() => ({ tasks: [] }))
        if (current()) {
          setTasks(Array.isArray(data) ? data : Array.isArray(data?.tasks) ? data.tasks : [])
          setTasksHasNext(Boolean(data?.pagination?.has_next))
        }
      }
      if (templateResult.status === "fulfilled" && templateResult.value.ok) {
        const data: unknown = await templateResult.value.json().catch(() => [])
        const nextTemplates: Record<string, Template> = {}
        for (const template of Array.isArray(data) ? data : []) {
          if (template && typeof template === "object" && "id" in template && typeof template.id === "string") {
            nextTemplates[template.id] = template as Template
          }
        }
        if (current()) setTemplatesById(nextTemplates)
      }
      if (current()) setLoading(false)
    })()
    return () => { workspaceVersion.current += 1 }
  }, [locale, user?.id, state.lastTaskUpdate])

  const loadMoreTasks = async () => {
    if (loadingMoreTasksRef.current || !tasksHasNext) return
    const version = workspaceVersion.current
    loadingMoreTasksRef.current = true
    setLoadingMoreTasks(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/chat/tasks?page=${taskPage + 1}&per_page=${TASKS_PER_PAGE}`)
      if (!response.ok) throw new Error("Could not load conversations")
      const data = await response.json()
      if (!Array.isArray(data?.tasks)) throw new Error("Invalid conversations")
      if (version !== workspaceVersion.current) return
      setTasks((previous) => Array.from(new Map(
        [...previous, ...data.tasks].map((task: Task) => [String(task.task_id), task]),
      ).values()))
      setTaskPage((page) => page + 1)
      setTasksHasNext(Boolean(data.pagination?.has_next))
    } catch {
      if (version === workspaceVersion.current) toast.error(t("sidebar.search.failed"))
    } finally {
      if (version === workspaceVersion.current) {
        loadingMoreTasksRef.current = false
        setLoadingMoreTasks(false)
      }
    }
  }

  useEffect(() => {
    if (!searchQuery) {
      setSearching(false)
      setSearchError(false)
      setSearchResults({ query: "", agents: [], tasks: [], hasNext: false })
      return
    }
    let current = true
    const controller = new AbortController()
    setSearching(true)
    setSearchError(false)
    const timer = setTimeout(async () => {
      try {
        const agentParams = new URLSearchParams({ search: searchQuery })
        const taskParams = new URLSearchParams({
          search: searchQuery, page: String(searchPage), per_page: String(TASKS_PER_PAGE),
        })
        const responses = await Promise.all([
          apiRequest(`${getApiUrl()}/api/agents?${agentParams}`, { signal: controller.signal }),
          apiRequest(`${getApiUrl()}/api/chat/tasks?${taskParams}`, { signal: controller.signal }),
        ])
        if (responses.some((response) => !response.ok)) throw new Error("Search failed")
        const [matchedAgents, matchedTasks] = await Promise.all(responses.map((response) => response.json()))
        if (!Array.isArray(matchedAgents) || !Array.isArray(matchedTasks?.tasks)) throw new Error("Invalid search results")
        if (!current) return
        setSearchResults((previous) => ({
          query: searchQuery,
          agents: matchedAgents,
          tasks: searchPage > 1 && previous.query === searchQuery
            ? Array.from(new Map([...previous.tasks, ...matchedTasks.tasks].map((task: Task) => [String(task.task_id), task])).values())
            : matchedTasks.tasks,
          hasNext: Boolean(matchedTasks.pagination?.has_next),
        }))
      } catch {
        if (current) setSearchError(true)
      } finally {
        if (current) setSearching(false)
      }
    }, searchPage === 1 ? 250 : 0)
    return () => {
      current = false
      clearTimeout(timer)
      controller.abort()
    }
  }, [searchQuery, searchPage, searchAttempt, state.lastTaskUpdate])

  const changeSearch = (value: string) => {
    setSearch(value)
    setSearchPage(1)
  }

  useEffect(() => {
    const close = (event: MouseEvent | TouchEvent) => {
      if (userMenuRef.current && !userMenuRef.current.contains(event.target as Node)) setUserMenuOpen(false)
    }
    document.addEventListener("mousedown", close)
    document.addEventListener("touchstart", close)
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setUserMenuOpen(false)
    }
    document.addEventListener("keydown", closeOnEscape)
    return () => {
      document.removeEventListener("mousedown", close)
      document.removeEventListener("touchstart", close)
      document.removeEventListener("keydown", closeOnEscape)
    }
  }, [])

  const activeTaskId = pathname.match(/^\/task\/([^/]+)\/?$/)?.[1] ?? null
  const isAgentBuilderPage = pathname === "/agent" || pathname.startsWith("/agent/new")
  const shouldCollapse = allowCollapse && (isAgentBuilderPage || !sidebarOpen)

  const latestTaskByAgent = useMemo(() => {
    const map = new Map<number, Task>()
    for (const task of tasks) {
      if (!task.agent_id) continue
      const existing = map.get(task.agent_id)
      const activityAt = new Date(task.updated_at ?? task.created_at).getTime()
      const existingActivityAt = existing ? new Date(existing.updated_at ?? existing.created_at).getTime() : Number.NEGATIVE_INFINITY
      if (!existing || activityAt > existingActivityAt) map.set(task.agent_id, task)
    }
    return map
  }, [tasks])

  const agentsById = useMemo(
    () => new Map(agents.map((agent) => [agent.id, agent])),
    [agents],
  )

  const getAgentAvatar = useCallback((agent: AgentSummary): string | null => {
    const uploadedLogo = resolveAgentLogoUrl(agent.logo_url ?? null, getApiUrl())
    if (uploadedLogo) return uploadedLogo
    const template = agent.template_id ? templatesById[agent.template_id] : undefined
    return template?.persona?.avatar ?? null
  }, [templatesById])

  const getAgentCategory = useCallback((agent: AgentSummary) => {
    const template = agent.template_id ? templatesById[agent.template_id] : undefined
    return categoryLabel(t, template?.category ?? "general")
  }, [t, templatesById])

  const openFirstConversation = useCallback(async (agent: AgentSummary) => {
    const template = agent.template_id ? templatesById[agent.template_id] : undefined
    if (openingAgentRef.current !== null) return

    openingAgentRef.current = agent.id
    setOpeningAgentId(agent.id)
    try {
      // The sidebar only loads a page of recent tasks. A missing task there
      // does not mean this teammate has never chatted. Resolve server-side
      // before creating an intro, including when an older chat was resumed.
      const response = await apiRequest(`${getApiUrl()}/api/chat/tasks?agent_id=${agent.id}&page=1&per_page=1`)
      if (!response.ok) throw new Error("Could not load teammate conversations")
      const data = await response.json()
      if (!Array.isArray(data?.tasks)) throw new Error("Invalid teammate conversations")
      const latest = data.tasks[0] as Task | undefined
      if (latest) {
        router.push(`/task/${latest.task_id}`)
        return
      }
      if (!template?.persona) {
        router.push(getAgentTaskHref(agent.id))
        return
      }
      const result = await createSeededAgentTask({
        agentId: agent.id,
        persona: template.persona,
        connections: template.connections,
        strings: {
          beforeWeStart: t("templates.marketplace.beforeWeStart"),
          closingNote: t("templates.marketplace.hireClosingNote"),
          connectAppsLabel: t("chatPage.clarification.connectApps.title"),
        },
      })
      const fullTitle = template.persona.role
        ? `${template.persona.name} — ${template.persona.role}`
        : template.persona.name
      const title = fullTitle.length > 50 ? `${fullTitle.slice(0, 50)}...` : fullTitle
      const createdAt = new Date().toISOString()
      setTasks((current) => [{
        task_id: result.taskId,
        title,
        status: "pending",
        created_at: createdAt,
        updated_at: createdAt,
        agent_id: agent.id,
        agent_name: agent.name,
        agent_logo_url: agent.logo_url ?? undefined,
      }, ...current])
      router.push(`/task/${result.taskId}`)
    } catch {
      toast.error(t("chatPage.page.startConversationFailed"))
    } finally {
      openingAgentRef.current = null
      setOpeningAgentId(null)
    }
  }, [router, t, templatesById])

  const hasSearchResults = searchResults.query === searchQuery
  const visibleAgents = (searchQuery ? hasSearchResults ? searchResults.agents : [] : agents)
    .filter((agent) => agent.status === "published")
  const visibleTasks = searchQuery ? hasSearchResults ? searchResults.tasks : [] : tasks

  if (shouldCollapse) {
    return (
      <aside className={cn("relative flex h-full w-[58px] shrink-0 flex-col items-center border-r border-border bg-[#f8f8fa] py-3", className)}>
        <Link href="/task" className="grid h-9 w-9 place-items-center rounded-xl hover:bg-white">
          <Image src={branding.logoPath} alt={branding.logoAlt} width={28} height={28} className="h-7 w-7 rounded-lg object-contain" />
        </Link>
        <button onClick={() => setSidebarOpen(true)} className="mt-3 grid h-9 w-9 place-items-center rounded-xl text-muted-foreground hover:bg-white hover:text-foreground" aria-label="Open navigation"><ChevronRight className="h-4 w-4" /></button>
        <div className="mt-auto">
          <button onClick={() => setSidebarOpen(true)} aria-label={profileLabel} title={profileLabel} className="grid h-9 w-9 place-items-center rounded-full bg-[#2536e0] text-xs font-semibold uppercase text-white">{userLabel.charAt(0)}</button>
        </div>
      </aside>
    )
  }

  return (
    <>
      <aside className={cn("flex h-full w-[270px] shrink-0 flex-col border-r border-[#ececf1] bg-[#f8f8fa] text-[#1a1a1f]", className)}>
        <div className="flex h-14 items-center gap-1 px-3">
          <Link href="/task" className="flex min-w-0 flex-1 items-center gap-2 rounded-xl px-2 py-1.5 hover:bg-white">
            <Image src={branding.logoPath} alt={branding.logoAlt} width={24} height={24} className="h-6 w-6 rounded-md object-contain" />
            <span className="truncate text-sm font-bold tracking-tight">{branding.appName}</span>
          </Link>
          <Popover open={startOpen} onOpenChange={setStartOpen}>
            <PopoverTrigger asChild>
              <button aria-label="Start something new" className="grid h-8 w-8 place-items-center rounded-lg text-muted-foreground hover:bg-white hover:text-[#2536e0]"><Plus className="h-4 w-4" /></button>
            </PopoverTrigger>
            <PopoverContent align="end" sideOffset={8} className="w-56 rounded-2xl p-1.5 shadow-xl">
              <button onClick={() => { setStartOpen(false); router.push("/task") }} className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left hover:bg-accent"><span className="grid h-8 w-8 place-items-center rounded-lg bg-[#eef1fe] text-[#2536e0]"><Sparkles className="h-4 w-4" /></span><b className="text-sm font-semibold">New task</b><ChevronRight className="ml-auto h-4 w-4 text-muted-foreground" /></button>
              <div className="my-1 h-px bg-border" />
              <button onClick={() => { setStartOpen(false); setPickerOpen(true) }} className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left hover:bg-accent"><span className="grid h-8 w-8 place-items-center rounded-lg bg-[#f4efff] text-[#7657d8]"><Bot className="h-4 w-4" /></span><b className="text-sm font-semibold">Add a teammate</b><ChevronRight className="ml-auto h-4 w-4 text-muted-foreground" /></button>
            </PopoverContent>
          </Popover>
          {allowCollapse && <button onClick={() => setSidebarOpen(false)} aria-label="Close navigation" className="grid h-8 w-8 place-items-center rounded-lg text-muted-foreground hover:bg-white hover:text-foreground"><ChevronLeft className="h-4 w-4" /></button>}
        </div>

        <div className="px-3 pb-2">
          <label className="relative block">
            <span className="sr-only">Search your team and conversations</span>
            <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <input value={search} onChange={(event) => changeSearch(event.target.value)} type="search" placeholder="Search teammates and messages" className="h-9 w-full rounded-xl border border-[#e7e7ec] bg-white pl-9 pr-8 text-xs outline-none transition-shadow placeholder:text-[#9a9aa5] focus:border-[#b9c1f7] focus:ring-4 focus:ring-[#2536e0]/[0.07] [&::-webkit-search-cancel-button]:appearance-none" />
            {search && <button onClick={() => changeSearch("")} aria-label="Clear search" className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1 text-muted-foreground hover:bg-accent"><X className="h-3 w-3" /></button>}
          </label>
        </div>

        <div className="flex min-h-0 flex-1 flex-col px-2.5">
          <div className="flex items-center px-2 pb-2 pt-2">
            <h2 className="flex-1 text-[10.5px] font-bold uppercase tracking-[0.08em] text-[#6b6b76]">{view === "team" ? "Your team" : "Recent"}</h2>
            <button onClick={() => router.push("/workforces?view=teammates")} aria-label="Manage my team" title="Manage my team" className="grid h-7 w-7 place-items-center rounded-lg text-muted-foreground hover:bg-white hover:text-[#2536e0]"><Settings2 className="h-3.5 w-3.5" /></button>
            <button onClick={() => setView(view === "team" ? "recent" : "team")} aria-label={view === "team" ? "Show recent conversations" : "Show my team"} className="ml-1 inline-flex h-7 items-center gap-1.5 rounded-lg px-2 text-[11px] font-medium text-muted-foreground hover:bg-white hover:text-foreground">{view === "team" ? <Clock3 className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}{view === "team" ? "Recent" : "Team"}</button>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto pb-3 [scrollbar-width:thin]">
            {(searchQuery ? searching && searchPage === 1 : loading) ? <div role="status" aria-label={t("common.loading")} className="grid place-items-center py-10"><Loader2 className="h-4 w-4 animate-spin text-muted-foreground" /></div> : <>
            {searchQuery && searchError && <div role="alert" className="px-3 py-4 text-center text-xs text-muted-foreground"><p>{t("sidebar.search.failed")}</p><button onClick={() => setSearchAttempt((attempt) => attempt + 1)} className="mt-2 text-primary">{t("common.retry")}</button></div>}
            {(view === "team" || searchQuery) && (
              <div className="space-y-0.5">
                {searchQuery && visibleAgents.length > 0 && <p className="px-3 py-2 text-xs text-muted-foreground">{t("teamPage.tabs.teammates")}</p>}
                {visibleAgents.map((agent) => {
                  const task = latestTaskByAgent.get(agent.id)
                  const presence = taskPresence(task?.status)
                  const active = String(task?.task_id) === activeTaskId
                  const destination = task ? `/task/${task.task_id}` : getAgentTaskHref(agent.id)
                  const isOpening = openingAgentId === agent.id
                  return (
                    <Link
                      key={agent.id}
                      href={destination}
                      aria-busy={isOpening || undefined}
                      onClick={(event) => {
                        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) return
                        event.preventDefault()
                        void openFirstConversation(agent)
                      }}
                      className={cn("group flex items-start gap-2.5 rounded-xl px-2.5 py-2 transition-colors", active ? "bg-white shadow-[0_1px_2px_rgba(24,28,60,.06)]" : "hover:bg-[#f0f0f3]", isOpening && "pointer-events-none opacity-70")}
                    >
                      <span className={cn("relative mt-0.5 rounded-full ring-1 ring-offset-2 ring-offset-[#f8f8fa]", presence.ring)}>
                        <PersonaAvatar persona={{ name: agent.name, avatar: getAgentAvatar(agent) }} sizeClassName="h-8 w-8" textClassName="text-[11px]" decorative />
                        <i className={cn("absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2 border-[#f8f8fa]", presence.dot)} />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="flex items-baseline gap-1.5"><b className="max-w-[48%] shrink-0 truncate text-[13px] font-semibold">{agent.name}</b><small className="min-w-0 truncate text-[10.5px] text-muted-foreground">{getAgentCategory(agent)}</small>{isOpening ? <Loader2 className="ml-auto h-3 w-3 shrink-0 animate-spin text-muted-foreground" /> : task && <time className="ml-auto shrink-0 text-[9.5px] text-muted-foreground">{compactTime(task.updated_at ?? task.created_at)}</time>}</span>
                        <span className="mt-0.5 block truncate text-[11.5px] leading-4 text-muted-foreground">{task?.title || "Just joined — ready for your first job"}</span>
                      </span>
                    </Link>
                  )
                })}
              </div>
            )}
            {(view === "recent" || searchQuery) && (
              <div className="space-y-0.5">
                {searchQuery && visibleTasks.length > 0 && <p className="px-3 py-2 text-xs text-muted-foreground">{t("sidebar.search.conversations")}</p>}
                {visibleTasks.map((task) => {
                  const active = String(task.task_id) === activeTaskId
                  const compactChannelName = task.channel_name ? getCompactChannelName(task.channel_name, task.channel_type) : null
                  const taskAgent = task.agent_id ? agentsById.get(task.agent_id) : undefined
                  const taskAvatar = task.agent_logo_url
                    ? resolveAgentLogoUrl(task.agent_logo_url, getApiUrl())
                    : taskAgent
                      ? getAgentAvatar(taskAgent)
                      : branding.logoPath
                  return (
                    <Link key={task.task_id} href={`/task/${task.task_id}`} className={cn("flex items-start gap-2.5 rounded-xl px-2.5 py-2 hover:bg-[#f0f0f3]", active && "bg-white shadow-[0_1px_2px_rgba(24,28,60,.06)]")}>
                      <PersonaAvatar persona={{ name: task.agent_name || branding.appName, avatar: taskAvatar }} sizeClassName="h-7 w-7" textClassName="text-[10px]" decorative />
                      <span className="min-w-0 flex-1"><span className="flex items-center gap-2"><b className="truncate text-xs font-semibold">{task.title || "Untitled task"}</b><time className="ml-auto shrink-0 text-[9.5px] text-muted-foreground">{compactTime(task.created_at)}</time></span><span className="mt-0.5 flex items-center gap-1.5 truncate text-[10.5px] text-muted-foreground">{task.agent_name || branding.appName}{compactChannelName && <span title={getChannelTooltip(task.channel_name!, task.channel_type)} className="inline-flex items-center gap-1 rounded border border-border px-1"><ChannelTypeIcon channelType={task.channel_type} />{compactChannelName}</span>}</span></span>
                    </Link>
                  )
                })}
                {!searchQuery && visibleTasks.length === 0 && <p className="px-3 py-8 text-center text-xs text-muted-foreground">No conversations yet.</p>}
                {!searchQuery && tasksHasNext && <button disabled={loadingMoreTasks} onClick={() => void loadMoreTasks()} className="w-full px-3 py-3 text-xs text-primary disabled:opacity-50">{loadingMoreTasks ? t("common.loading") : t("sidebar.search.loadMore")}</button>}
                {searchQuery && hasSearchResults && searchResults.hasNext && !searchError && <button disabled={searching} onClick={() => setSearchPage((page) => page + 1)} className="w-full px-3 py-3 text-xs text-primary disabled:opacity-50">{searching ? t("common.loading") : t("sidebar.search.loadMore")}</button>}
              </div>
            )}
            {searchQuery && !searchError && !searching && hasSearchResults && visibleAgents.length === 0 && visibleTasks.length === 0 && <p role="status" className="px-3 py-8 text-center text-xs text-muted-foreground">{t("sidebar.search.empty")}</p>}
            </>}
          </div>
        </div>

        <div className="border-t border-[#ececf1] px-2.5 py-2">
          <Link href="/files" className={cn("flex items-center gap-2.5 rounded-xl px-2.5 py-2 text-[12.5px] font-medium text-muted-foreground hover:bg-white hover:text-foreground", pathname.startsWith("/files") && "bg-white text-[#2536e0]")}><FileText className="h-4 w-4" /> Files</Link>
          <Link href="/tools" className={cn("flex items-center gap-2.5 rounded-xl px-2.5 py-2 text-[12.5px] font-medium text-muted-foreground hover:bg-white hover:text-foreground", pathname.startsWith("/tools") && "bg-white text-[#2536e0]")}><Wrench className="h-4 w-4" /> Tools and apps</Link>
          <button onClick={() => setRoutinesOpen(true)} className="flex w-full items-center gap-2.5 rounded-xl px-2.5 py-2 text-left text-[12.5px] font-medium text-muted-foreground hover:bg-white hover:text-foreground"><RotateCcw className="h-4 w-4" /> Recurring jobs</button>
        </div>

        <div ref={userMenuRef} className="relative border-t border-[#ececf1] p-2.5">
          {userMenuOpen && <div data-testid="workspace-menu" className="absolute bottom-[calc(100%+8px)] left-2 right-2 z-50 max-h-[70vh] overflow-y-auto rounded-2xl border border-border bg-popover py-2 shadow-2xl">
            <div className="flex items-center gap-3 px-4 py-3"><span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-[#2536e0] text-xs font-semibold uppercase text-white">{userLabel.charAt(0)}</span><span className="min-w-0"><b className="block truncate text-sm font-semibold">{userLabel}</b>{subtitle && <small className="block truncate text-xs text-muted-foreground">{subtitle}</small>}</span></div>
            <div className="mx-3 my-1 h-px bg-border" />
            <div className="space-y-0.5 px-2">
              {userMenuItems.map((item) => <WorkspaceMenuItem key={item.href} item={item} pathname={pathname} t={t} onNavigate={() => setUserMenuOpen(false)} />)}
            </div>
            {navigationGroups.map((group) => <SectionMenu key={`${group.id ?? group.title}:${pathname}`} group={group} pathname={pathname} t={t} onNavigate={() => setUserMenuOpen(false)} />)}
            <div className="mx-3 my-1 h-px bg-border" />
            <button onClick={() => void (async () => { if (await logout()) setUserMenuOpen(false); else toast.error(t("sidebar.user.logoutFailed")) })()} className="mx-2 flex w-[calc(100%-16px)] items-center gap-2 rounded-lg px-2 py-2 text-left text-[13px] text-red-600 hover:bg-red-50"><LogOut className="h-4 w-4" />{t("sidebar.user.logoutTitle")}</button>
          </div>}
          <button onClick={() => setUserMenuOpen((open) => !open)} aria-expanded={userMenuOpen} aria-label={profileLabel} className="flex w-full items-center gap-2.5 rounded-xl px-2 py-2 text-left hover:bg-white">
            <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-[#2536e0] text-xs font-semibold uppercase text-white">{userLabel.charAt(0)}</span>
            <span className="min-w-0 flex-1"><b className="block truncate text-[13px] font-semibold">{userLabel}</b>{subtitle && <small className="block truncate text-[11px] text-muted-foreground">{subtitle}</small>}</span>
            <ChevronsUpDown className="h-3.5 w-3.5 text-muted-foreground" />
          </button>
        </div>
      </aside>

      <TeammatePickerDialog open={pickerOpen} onOpenChange={setPickerOpen} />
      <RoutinesDialog open={routinesOpen} onOpenChange={setRoutinesOpen} agents={agents} />
    </>
  )
}
