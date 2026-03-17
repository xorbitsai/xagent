"use client"

import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { cn } from "@/lib/utils"
import { useState, useEffect, useCallback, useRef } from "react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useAuth } from "@/contexts/auth-context"
import { useApp } from "@/contexts/app-context-chat"
import { getBrandingFromEnv } from "@/lib/branding"
import {
  Activity,
  FileText,
  User,
  LogOut,
  Menu,
  X,
  ChevronDown,
  ChevronRight,
  Sparkles,
  Zap,
  Settings,
  Wrench,
  Users,
  Brain,
  Server,
  HardDrive,
  Layers,
  MessageSquare,
  Loader2,
  Trash2,
  CheckCircle2,
  XCircle,
  PauseCircle,
  Info,
  Tag,
  Github,
  Star,
} from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog"

import { useI18n } from "@/contexts/i18n-context"

interface Task {
  task_id: string
  title: string
  status: "completed" | "running" | "failed" | "pending" | "paused"
  created_at: string | number
  description?: string
  agent_id?: number
  agent_logo_url?: string
}

interface VersionInfo {
  version: string
  display_version?: string
  commit?: string
  build_time?: string
  latest_version?: string | null
  is_latest?: boolean | null
}

function formatStars(stars: number): string {
  if (stars >= 1000000) return `${(stars / 1000000).toFixed(1)}M`
  if (stars >= 1000) return `${(stars / 1000).toFixed(1)}k`
  return String(stars)
}

interface NavigationItem {
  name: string
  href: string
  icon: any
  color?: string
  children?: NavigationItem[]
  showTasks?: boolean
  nameKey?: string
}

interface NavigationGroup {
  title: string
  titleKey?: string
  items: NavigationItem[]
}

const navigationGroups: NavigationGroup[] = [
  {
    title: "Agent Development",
    titleKey: "nav.sections.agentDevelopment",
    items: [
      {
        name: "Task",
        nameKey: "nav.task",
        href: "/task",
        icon: Sparkles,
        color: "text-blue-500"
      },
      {
        name: "BUILD",
        nameKey: "nav.build",
        href: "/build",
        icon: Zap,
        color: "text-yellow-400"
      },
      {
        name: "Templates",
        nameKey: "nav.templates",
        href: "/templates",
        icon: Layers,
        color: "text-purple-400"
      },
    ]
  },
  {
    title: "Resources",
    titleKey: "nav.sections.resources",
    items: [
      {
        name: "Knowledge Base",
        nameKey: "nav.knowledgeBase",
        href: "/kb",
        icon: Brain,
        color: "text-gray-500"
      },
      {
        name: "Models",
        nameKey: "nav.models",
        href: "/models",
        icon: Server,
        color: "text-gray-500"
      },
      {
        name: "Memory",
        nameKey: "nav.memory",
        href: "/memory",
        icon: HardDrive,
        color: "text-gray-500"
      }
    ]
  }
]

const baseUserMenuItems: NavigationItem[] = [
  {
    name: "Tools",
    nameKey: "nav.tools",
    href: "/tools",
    icon: Wrench,
    color: "text-blue-400"
  },
  {
    name: "Files",
    nameKey: "nav.files",
    href: "/files",
    icon: FileText,
    color: "text-blue-400"
  },
  {
    name: "Monitoring",
    nameKey: "nav.monitoring",
    href: "/monitoring",
    icon: Activity,
    color: "text-blue-400"
  },
  {
    name: "Settings",
    nameKey: "nav.settings",
    href: "/settings",
    icon: Settings,
    color: "text-blue-400"
  }
]

const getUserMenuItemsForUser = (user: any): NavigationItem[] => {
  const menuItems = [...baseUserMenuItems]

  if (user?.is_admin) {
    menuItems.splice(-1, 0, {
      name: "User Management",
      nameKey: "nav.userManagement",
      href: "/users/",
      icon: Users,
      color: "text-blue-400"
    })
  }

  return menuItems
}

interface SidebarProps {
  isCollapsible?: boolean
  className?: string
}

export function Sidebar({ className }: SidebarProps) {
  const pathname = usePathname()
  const router = useRouter()
  const { user, logout } = useAuth()
  const branding = getBrandingFromEnv()
  const { t } = useI18n()
  const { state } = useApp()
  const githubUrl = process.env.NEXT_PUBLIC_GITHUB_URL || "https://github.com/xorbitsai/xagent"
  const normalizedGithubUrl = githubUrl.replace(/\.git$/, "").replace(/\/$/, "")
  const githubRepoDisplay = normalizedGithubUrl.replace(/^https?:\/\/github\.com\//i, "")
  const licenseUrl = `${normalizedGithubUrl}/blob/main/LICENSE`
  const [githubStars, setGithubStars] = useState<number | null>(null)

  const deleteTask = async (taskId: string, e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()

    if (!confirm(t('common.deleteConfirm'))) return

    try {
      const response = await apiRequest(`${getApiUrl()}/api/chat/task/${taskId}`, {
        method: 'DELETE',
        headers: {}
      })

      if (response.ok) {
        setTasks(prev => prev.filter(task => task.task_id !== taskId))

        if (Number(getCurrentTaskId()) === Number(taskId)) {
          router.push('/task')
        }
      }
    } catch (error) {
      console.error('Failed to delete task:', error)
    }
  }

  const [isExpanded, setIsExpanded] = useState(false)
  const [expandedMenus, setExpandedMenus] = useState<string[]>(["/agent"]) // Use href as a stable key
  const [showUserMenu, setShowUserMenu] = useState(false)
  const [isAboutOpen, setIsAboutOpen] = useState(false)
  const sidebarRef = useRef<HTMLDivElement | null>(null)
  const userMenuRef = useRef<HTMLDivElement | null>(null)

  // Handle click outside for user menu
  useEffect(() => {
    const handleClickOutsideUserMenu = (event: MouseEvent | TouchEvent) => {
      if (userMenuRef.current && !userMenuRef.current.contains(event.target as Node)) {
        setShowUserMenu(false)
      }
    }

    if (showUserMenu) {
      document.addEventListener('mousedown', handleClickOutsideUserMenu)
      document.addEventListener('touchstart', handleClickOutsideUserMenu)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutsideUserMenu)
      document.removeEventListener('touchstart', handleClickOutsideUserMenu)
    }
  }, [showUserMenu])

  // Get currently selected task ID (parsed from path, supports /task/[id] format)
  const getCurrentTaskId = useCallback(() => {
    // Match /task/[number] pattern
    const match = pathname.match(/^\/task\/(\d+)\/?$/);
    if (match) {
      return match[1];
    }
    return null;
  }, [pathname])

  const [tasks, setTasks] = useState<Task[]>([])
  const [versionInfo, setVersionInfo] = useState<VersionInfo | null>(null)
  const [isLoadingTasks, setIsLoadingTasks] = useState(false)
  const [isHistoryExpanded, setIsHistoryExpanded] = useState(true)
  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const navRef = useRef<HTMLElement | null>(null)
  const displayVersion = versionInfo?.display_version || "unknown"

  useEffect(() => {
    let isCancelled = false

    const loadVersion = async () => {
      try {
        const response = await fetch(`${getApiUrl()}/api/system/version`, {
          method: "GET",
          cache: "no-store",
        })

        if (!response.ok) {
          throw new Error(`Failed to load version: ${response.status}`)
        }

        const data = (await response.json()) as VersionInfo
        if (!isCancelled) {
          setVersionInfo({
            version: data.version || "unknown",
            display_version: data.display_version || "unknown",
            commit: data.commit || "",
            build_time: data.build_time || "",
            latest_version: data.latest_version ?? null,
            is_latest: data.is_latest ?? null,
          })
        }
      } catch {
        if (!isCancelled) {
          setVersionInfo({
            version: "unknown",
            display_version: "unknown",
            commit: "",
            build_time: "",
            latest_version: null,
            is_latest: null,
          })
        }
      }
    }

    void loadVersion()

    return () => {
      isCancelled = true
    }
  }, [])

  useEffect(() => {
    if (!isAboutOpen) return

    const match = githubRepoDisplay.match(/^([^/]+)\/([^/]+)$/)
    if (!match) {
      setGithubStars(null)
      return
    }

    const controller = new AbortController()
    const [, owner, repo] = match

    const loadStars = async () => {
      try {
        const response = await fetch(`https://api.github.com/repos/${owner}/${repo}`, {
          method: "GET",
          headers: { Accept: "application/vnd.github+json" },
          signal: controller.signal,
        })
        if (!response.ok) {
          setGithubStars(null)
          return
        }
        const data = (await response.json()) as { stargazers_count?: number }
        setGithubStars(typeof data.stargazers_count === "number" ? data.stargazers_count : null)
      } catch {
        if (!controller.signal.aborted) {
          setGithubStars(null)
        }
      }
    }

    void loadStars()

    return () => {
      controller.abort()
    }
  }, [githubRepoDisplay, isAboutOpen])

  // Load task list
  const loadTasks = useCallback(async (pageNum = 1, isAppending = false) => {
    if (isAppending) {
      setIsLoadingMore(true)
    } else {
      setIsLoadingTasks(true)
    }

    try {
      const response = await apiRequest(`${getApiUrl()}/api/chat/tasks?exclude_agent_type=text2sql&page=${pageNum}&per_page=10`)
      if (response.ok) {
        const data = await response.json()
        // Handle new API response format {tasks: [...], pagination: {...}}
        const newTasks = data.tasks || (Array.isArray(data) ? data : [])

        if (isAppending) {
          setTasks(prev => [...prev, ...newTasks])
        } else {
          setTasks(newTasks)
        }

        // Update pagination status
        const totalPages = data.pagination?.total_pages || 1
        setHasMore(pageNum < totalPages)
        setPage(pageNum)
      }
    } catch (error) {
      console.error('Failed to load tasks:', error)
    } finally {
      setIsLoadingTasks(false)
      setIsLoadingMore(false)
    }
  }, [])

  // Monitor task list changes, if content is not enough to fill the container and there is more data, automatically load the next page
  useEffect(() => {
    if (!navRef.current) return

    const { scrollHeight, clientHeight } = navRef.current
    // If content height is less than or equal to container height (plus a buffer), and there is more data, and not loading
    if (scrollHeight <= clientHeight + 20 && hasMore && !isLoadingMore && !isLoadingTasks) {
       // Use setTimeout to avoid continuous state updates in one render cycle
       const timer = setTimeout(() => {
         loadTasks(page + 1, true)
       }, 100)
       return () => clearTimeout(timer)
    }
  }, [tasks, hasMore, isLoadingMore, isLoadingTasks, page, loadTasks])

  useEffect(() => {
    if (isHistoryExpanded) {
      loadTasks(1, false)
    }
  }, [isHistoryExpanded, loadTasks, state.lastTaskUpdate])

  const handleScroll = (e: React.UIEvent<HTMLElement>) => {
    const { scrollTop, scrollHeight, clientHeight } = e.currentTarget
    if (scrollHeight - scrollTop <= clientHeight + 20 && hasMore && !isLoadingMore && !isLoadingTasks) {
       loadTasks(page + 1, true)
    }
  }

  // Sidebar is hidden by default on Agent pages, but kept visible on Vibe and Build pages, and shown on other pages
  // For agent pages, sidebar is only shown when isExpanded is true
  // Build page no longer automatically hides
  // /agent/[id] page does not auto-collapse (for agent chat)
  const isAgentChatPage = pathname.match(/^\/agent\/\d+$/)
  const shouldShowSidebar = !((pathname.startsWith('/agent') && !pathname.startsWith('/agent/vibe') && !isAgentChatPage)) || isExpanded
  const isAgentPage = (pathname.startsWith('/agent') && !pathname.startsWith('/agent/vibe') && !isAgentChatPage)

  // When in collapsible state and expanded, click outside sidebar to automatically collapse
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent | TouchEvent) => {
      if (!sidebarRef.current) return
      // Only process when in collapsible page and currently expanded
      if (isAgentPage && shouldShowSidebar && isExpanded) {
        if (!sidebarRef.current.contains(event.target as Node)) {
          setIsExpanded(false)
        }
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    document.addEventListener('touchstart', handleClickOutside)

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('touchstart', handleClickOutside)
    }
  }, [isAgentPage, shouldShowSidebar, isExpanded])

  const toggleMenu = (menuName: string) => {
    setExpandedMenus(prev =>
      prev.includes(menuName)
        ? prev.filter(name => name !== menuName)
        : [...prev, menuName]
    )
  }

  const isMenuExpanded = (menuName: string) => {
    return expandedMenus.includes(menuName)
  }

  if (isAgentPage && !shouldShowSidebar) {
    return (
      <div className="flex items-center justify-center w-12 bg-card border-r border-border">
        <button
          onClick={() => setIsExpanded(true)}
          className="p-2 text-muted-foreground hover:text-foreground hover:bg-accent rounded-md transition-colors"
        >
          <Menu className="h-5 w-5" />
        </button>
      </div>
    )
  }

  return (
    <div ref={sidebarRef} className={cn(
      "flex flex-col bg-card border-r border-border transition-all duration-300",
      isAgentPage ? "h-full" : "h-full",
      shouldShowSidebar ? "w-64" : "w-0",
      className
    )}>
      {/* Logo */}
      <div className="flex h-16 items-center justify-between border-b border-border px-4">
        <Link href="/task" className="flex items-center gap-2">
          <img
            src={branding.logoPath}
            alt={branding.logoAlt}
            className="h-12 w-12"
          />
          <h1 className="text-2xl font-bold gradient-text">{branding.appName}</h1>
        </Link>
        {isAgentPage && (
          <button
            onClick={() => setIsExpanded(false)}
            className="p-1 text-muted-foreground hover:text-foreground hover:bg-accent rounded-md transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        )}
      </div>

      {/* Navigation */}
      <nav
        ref={navRef}
        className="flex-1 min-h-0 overflow-y-auto px-3 pb-4 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:'none'] [scrollbar-width:'none']"
        onScroll={handleScroll}
      >
        {/* Sticky Navigation Groups */}
        <div className="sticky top-0 z-10 bg-card -mx-3 px-3 pb-2 pt-2">
          {/* Groups */}
          {navigationGroups.map((group, groupIndex) => (
            <div key={group.title} className={cn("mb-6", groupIndex === 0 && "mt-0")}>
              <div className="px-4 py-2 text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">
                {group.titleKey ? t(group.titleKey) : group.title}
              </div>
              <div className="space-y-1">
                {group.items.map((item) => {
                  const isActive = pathname === item.href ||
                    (item.href !== "/" && pathname.startsWith(item.href))
                  const hasChildren = item.children && item.children.length > 0
                  const isExpanded = isMenuExpanded(item.href)

                  const activeStyle = `
                    bg-gradient-to-r
                    from-[hsl(var(--sidebar-active-bg-from))]
                    to-[hsl(var(--sidebar-active-bg-to))]
                    text-[hsl(var(--sidebar-active-text))]
                    font-semibold
                    relative
                    before:absolute before:left-0 before:top-1/2 before:-translate-y-1/2
                    before:h-[100%] before:w-1
                    before:bg-[hsl(var(--sidebar-active-border))]
                    before:rounded-r-full
                  `;
                  const inactiveStyle = "text-muted-foreground hover:bg-accent/50 hover:text-foreground"

                  if (hasChildren) {
                    return (
                      <div key={item.name} className="mb-1">
                        <button
                          onClick={() => toggleMenu(item.href)}
                          className={cn(
                            "group flex items-center justify-between w-full px-4 py-2 text-sm transition-colors relative",
                            isActive ? activeStyle : inactiveStyle
                          )}
                        >
                          <div className="flex items-center gap-3">
                            <item.icon className={cn("h-5 w-5", isActive ? "text-[hsl(var(--sidebar-active-text))]" : "text-gray-500")} />
                            {item.nameKey ? t(item.nameKey) : item.name}
                          </div>
                          {isExpanded ? (
                            <ChevronDown className="h-3 w-3 opacity-50" />
                          ) : (
                            <ChevronRight className="h-3 w-3 opacity-50" />
                          )}
                        </button>
                        {isExpanded && item.children && (
                          <div className="ml-4 mt-1 space-y-1 border-l border-border/40 pl-2">
                            {item.children.map((child) => {
                              const isChildActive = pathname === child.href
                              return (
                                <div key={child.href}>
                                  <Link
                                    href={child.href}
                                    className={cn(
                                      "group flex items-center px-4 py-2 text-sm font-medium rounded-r-full transition-colors",
                                      isChildActive
                                        ? "bg-[hsl(var(--sidebar-active-bg-from))] text-[hsl(var(--sidebar-active-text))] relative before:absolute before:left-0 before:top-1/2 before:-translate-y-1/2 before:h-4 before:w-1 before:bg-[hsl(var(--sidebar-active-border))] before:rounded-r-full"
                                        : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                                    )}
                                  >
                                    <child.icon className={cn("h-4 w-4 mr-3", isChildActive ? "text-[hsl(var(--sidebar-active-text))]" : child.color || "text-muted-foreground")} />
                                    {child.nameKey ? t(child.nameKey) : child.name}
                                  </Link>
                                </div>
                              )
                            })}
                          </div>
                        )}
                      </div>
                    )
                  }

                  return (
                    <Link
                      key={item.name}
                      href={item.href}
                      className={cn(
                        "group flex items-center px-4 py-3 text-sm font-medium transition-colors mb-1",
                        isActive ? activeStyle : inactiveStyle
                      )}
                    >
                      <item.icon className={cn("h-5 w-5 mr-3", isActive ? "text-[hsl(var(--sidebar-active-text))]" : "text-gray-500")} />
                      {item.nameKey ? t(item.nameKey) : item.name}
                    </Link>
                  )
                })}
              </div>
            </div>
          ))}
        </div>

        {/* History Section */}
        <div className="pt-4 border-t border-border mt-2">
          <div
            className="px-4 py-2 text-xs font-bold text-slate-400 uppercase tracking-wider mb-1 flex items-center justify-between cursor-pointer hover:text-slate-300 transition-colors"
            onClick={() => setIsHistoryExpanded(!isHistoryExpanded)}
          >
            <span>{t('nav.history')}</span>
            {isHistoryExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          </div>

          {isHistoryExpanded && (
            <div className="space-y-1">
              {isLoadingTasks ? (
                  <div className="flex items-center justify-center py-4">
                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                  </div>
              ) : tasks.length > 0 ? (
                <>
                  {tasks.map(task => {
                    const currentTaskId = getCurrentTaskId();
                    return (
                      <Link
                        key={task.task_id}
                        href={`/task/${task.task_id}`}
                        title={task.title}
                        className={cn(
                          "group flex items-center px-4 py-2 text-sm font-medium transition-colors mb-1 truncate relative pr-8",
                          String(currentTaskId) === String(task.task_id)
                            ? "bg-accent/80 text-accent-foreground font-medium"
                            : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                        )}
                      >
                        <div className="relative h-4 w-4 mr-3 flex-shrink-0">
                          {task.agent_id && task.agent_logo_url ? (
                             <img
                               src={`${getApiUrl()}${task.agent_logo_url}`}
                               alt="Agent Logo"
                               className="h-4 w-4 absolute inset-0 transition-opacity duration-200 group-hover:opacity-0 rounded-full object-cover"
                             />
                          ) : (
                            <MessageSquare className={cn(
                              "h-4 w-4 absolute inset-0 transition-opacity duration-200 group-hover:opacity-0",
                              String(currentTaskId) === String(task.task_id) ? "text-accent-foreground" : "text-gray-500"
                            )} />
                          )}
                          <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center">
                            {task.status === 'running' && <Loader2 className="h-4 w-4 animate-spin text-blue-500" />}
                            {task.status === 'completed' && <CheckCircle2 className="h-4 w-4 text-green-500" />}
                            {task.status === 'failed' && <XCircle className="h-4 w-4 text-red-500" />}
                            {task.status === 'paused' && <PauseCircle className="h-4 w-4 text-yellow-500" />}
                            {task.status === 'pending' && <Loader2 className="h-4 w-4 animate-spin text-gray-400" />}
                          </div>
                        </div>
                        <span className="truncate flex-1 text-left">{task.title || "Untitled Task"}</span>
                        <button
                          onClick={(e) => deleteTask(task.task_id, e)}
                          className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity p-1 text-muted-foreground hover:text-red-500 rounded-md hover:bg-slate-200 dark:hover:bg-slate-700 z-10"
                          title={t('common.delete')}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </Link>
                  )})}
                  {isLoadingMore && (
                      <div className="flex items-center justify-center py-2">
                        <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                      </div>
                  )}
                </>
              ) : (
                <div className="px-4 py-2 text-sm text-muted-foreground">
                  {t('common.noData')}
                </div>
              )}
            </div>
          )}
        </div>
      </nav>


      {/* User Profile */}
      <div className="border-t border-border p-4 relative" ref={userMenuRef}>
        {showUserMenu && (
          <div className="absolute bottom-full left-4 right-4 mb-2 bg-popover border border-border rounded-lg shadow-lg overflow-hidden animate-in fade-in zoom-in-95 duration-200 z-50">
             <div className="py-1">
                {getUserMenuItemsForUser(user).map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="flex items-center px-4 py-2 text-sm text-foreground hover:bg-accent transition-colors"
                    onClick={() => setShowUserMenu(false)}
                  >
                    <item.icon className="h-4 w-4 mr-3 text-muted-foreground" />
                    {item.nameKey ? t(item.nameKey) : item.name}
                  </Link>
                ))}
                <button
                  onClick={() => {
                    setShowUserMenu(false)
                    setIsAboutOpen(true)
                  }}
                  className="flex w-full items-center px-4 py-2 text-sm text-foreground hover:bg-accent transition-colors text-left"
                >
                  <Info className="h-4 w-4 mr-3 text-muted-foreground" />
                  {t("sidebar.about.menu")}
                </button>
                <div className="h-px bg-border my-1 mx-2" />
                <button
                  onClick={() => {
                    logout()
                    setShowUserMenu(false)
                  }}
                  className="flex w-full items-center px-4 py-2 text-sm hover:bg-red-50 dark:hover:bg-red-900/10 transition-colors text-left"
                >
                  <LogOut className="h-4 w-4 mr-3" />
                  {t('sidebar.user.logoutTitle')}
                </button>
             </div>
          </div>
        )}
        <button
          onClick={() => setShowUserMenu(!showUserMenu)}
          className="flex w-full items-center gap-2 hover:bg-accent/50 p-2 -ml-2 rounded-lg transition-colors text-left"
        >
          <div className="h-8 w-8 shrink-0 rounded-full bg-accent flex items-center justify-center">
            <User className="h-4 w-4 text-accent-foreground" />
          </div>
          <div className="ml-1 min-w-0 flex-1">
            <p className="truncate whitespace-nowrap text-base font-medium text-foreground">{user?.username || t('sidebar.user.defaultName')}</p>
            <div className="mt-0">
              <span
                className="inline-flex shrink-0 max-w-[8.5rem] items-center gap-1.5 overflow-hidden whitespace-nowrap rounded-md border border-border px-2 py-0.5 text-[11px] text-muted-foreground text-ellipsis"
                title={
                  versionInfo?.is_latest === true
                    ? t("sidebar.about.versionLatest")
                    : versionInfo?.is_latest === false
                      ? t("sidebar.about.versionUpdateAvailable")
                      : t("sidebar.about.versionStatusUnknown")
                }
              >
                <span
                  className={cn(
                    "h-1.5 w-1.5 rounded-full",
                    versionInfo?.is_latest === true
                      ? "bg-green-500"
                      : versionInfo?.is_latest === false
                        ? "bg-yellow-400"
                        : "bg-gray-400"
                  )}
                />
                {displayVersion}
              </span>
            </div>
          </div>
          <ChevronDown className={cn("h-4 w-4 text-muted-foreground transition-transform", showUserMenu && "rotate-180")} />
        </button>
      </div>

      <Dialog open={isAboutOpen} onOpenChange={setIsAboutOpen}>
        <DialogContent className="w-[min(760px,calc(100%-2rem))] max-w-none p-0 overflow-hidden">
          <DialogTitle className="sr-only">{t("sidebar.about.title")}</DialogTitle>
          <div className="grid grid-cols-10 min-h-[240px]">
            <div className="col-span-3 border-r border-border flex flex-col items-center justify-center px-6 py-8 text-center">
              <img
                src={branding.logoPath}
                alt={branding.logoAlt}
                className="h-14 w-14"
              />
              <div className="mt-3 text-base font-medium text-foreground">{branding.appName}</div>
            </div>
            <div className="col-span-7 px-8 py-8 flex flex-col justify-center gap-4">
              <div className="flex min-h-7 items-center gap-3 text-sm text-foreground">
                <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-accent text-accent-foreground">
                  <Tag className="h-4 w-4" />
                </span>
                <span className="inline-flex max-w-full items-center gap-1.5 whitespace-nowrap leading-7">
                  <span>{t("sidebar.about.version")}: {displayVersion}</span>
                  <span
                    className={cn(
                      "inline-block h-2 w-2 rounded-full",
                      versionInfo?.is_latest === true
                        ? "bg-green-500"
                        : versionInfo?.is_latest === false
                          ? "bg-yellow-400"
                          : "bg-gray-400"
                    )}
                    title={
                      versionInfo?.is_latest === true
                        ? t("sidebar.about.versionLatest")
                        : versionInfo?.is_latest === false
                          ? t("sidebar.about.versionUpdateAvailable")
                          : t("sidebar.about.versionStatusUnknown")
                    }
                  />
                </span>
              </div>
              <div className="flex min-h-7 items-center gap-3 text-sm text-foreground">
                <span className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-accent text-accent-foreground">
                  <Github className="h-4 w-4" />
                </span>
                <span className="leading-7">
                  {t("sidebar.about.github")}:{" "}
                  <a
                    href={githubUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-500 hover:underline break-all"
                  >
                    {githubRepoDisplay}
                  </a>
                </span>
              </div>
              <div className="flex min-h-7 items-center gap-3 text-sm text-foreground">
                <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-accent text-accent-foreground">
                  <Star className="h-4 w-4" />
                </span>
                <span className="leading-7">{t("sidebar.about.stars")}: {githubStars === null ? "--" : formatStars(githubStars)}</span>
              </div>
              <div className="flex min-h-7 items-center gap-3 text-sm text-foreground">
                <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-accent text-accent-foreground">
                  <FileText className="h-4 w-4" />
                </span>
                <a
                  href={licenseUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="leading-7 text-blue-500 hover:underline"
                >
                  {t("sidebar.about.license")}
                </a>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
