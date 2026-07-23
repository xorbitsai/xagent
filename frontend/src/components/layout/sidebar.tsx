"use client"

import Image from "next/image"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { cn } from "@/lib/utils"
import { SearchInput } from "@/components/ui/search-input"
import { useState, useEffect, useCallback, useMemo, useRef } from "react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useAuth } from "@/contexts/auth-context"
import { useApp } from "@/contexts/app-context-chat"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { getBrandingFromEnv } from "@/lib/branding"
import { getChannelTooltip, getCompactChannelName } from "@/lib/channel-display"
import { toast } from "@/components/ui/sonner"
import extraNav from "@/lib/extra-nav"
import {
  getNavigationGroupsForUser,
  getUserMenuItemsForUser,
  type NavigationItem,
  type NavigationGroup,
} from "@/lib/sidebar-navigation"
import {
  FileText,
  LogOut,
  X,
  ChevronDown,
  ChevronRight,
  ChevronLeft,
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
  MoreHorizontal,
  Edit2,
  Search,
  Radio,
  Send,
  ChevronsUpDown,
} from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog"
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover"

import { useI18n } from "@/contexts/i18n-context"

export {
  getNavigationGroupsForUser,
  getUserMenuItemsForUser,
}
export type {
  NavigationGroup,
  NavigationItem,
}

interface Task {
  task_id: string
  title: string
  status: "completed" | "running" | "failed" | "pending" | "paused" | "waiting_for_user"
  created_at: string | number
  description?: string
  agent_id?: number
  agent_logo_url?: string
  channel_id?: number
  channel_name?: string
  channel_type?: string
}

interface VersionInfo {
  version: string
  display_version?: string
  commit?: string
  build_time?: string
  latest_version?: string | null
  is_latest?: boolean | null
}

const TASKS_PER_PAGE = 10

const CHANNEL_ICON_PATHS: Record<string, string> = {
  feishu: "/icons/channels/feishu.svg",
}

function ChannelTypeIcon({ channelType }: { channelType?: string }) {
  const normalizedType = channelType?.trim().toLowerCase()
  if (!normalizedType) return null

  if (normalizedType === "telegram") {
    return <Send className="h-3 w-3 flex-shrink-0 text-[#229ED9]" aria-hidden="true" />
  }

  const iconPath = CHANNEL_ICON_PATHS[normalizedType]
  if (iconPath) {
    return (
      <Image
        src={iconPath}
        alt=""
        width={12}
        height={12}
        className="h-3 w-3 flex-shrink-0"
        unoptimized
        aria-hidden="true"
      />
    )
  }
  return <Radio className="h-3 w-3 flex-shrink-0" aria-hidden="true" />
}

function formatStars(stars: number): string {
  if (stars >= 1000000) return `${(stars / 1000000).toFixed(1)}M`
  if (stars >= 1000) return `${(stars / 1000).toFixed(1)}k`
  return String(stars)
}

interface SidebarProps {
  isCollapsible?: boolean
  className?: string
  allowCollapse?: boolean
}

export function Sidebar({ className, allowCollapse = true }: SidebarProps) {
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
  const navigationGroups = useMemo(() => {
    const extra = typeof extraNav === "function" ? extraNav(user) : extraNav
    return [...getNavigationGroupsForUser(user), ...extra]
  }, [user])
  const [githubStars, setGithubStars] = useState<number | null>(null)

  const [taskToDelete, setTaskToDelete] = useState<string | null>(null)
  const [isDeletingTask, setIsDeletingTask] = useState(false)

  const confirmDeleteTask = async () => {
    if (!taskToDelete) return
    const taskId = taskToDelete
    setIsDeletingTask(true)

    try {
      const response = await apiRequest(`${getApiUrl()}/api/chat/task/${taskId}`, {
        method: 'DELETE',
        headers: {}
      })

      if (response.ok) {
        setTasks(prev => prev.filter(task => String(task.task_id) !== String(taskId)))

        // Clean up refs and state
        taskStatusRef.current.delete(String(taskId))
        setUnreadTasks(prev => {
          if (!prev.has(String(taskId))) return prev
          const next = new Set(prev)
          next.delete(String(taskId))
          return next
        })

        if (String(getCurrentTaskId()) === String(taskId)) {
          router.push('/task')
        }
        setTaskToDelete(null)
      } else {
        toast.error(t('common.deleteFailed'))
      }
    } catch (error) {
      console.error('Failed to delete task:', error)
      toast.error(t('common.deleteFailed'))
    } finally {
      setIsDeletingTask(false)
    }
  }

  const deleteTask = (taskId: string, e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setTaskToDelete(taskId)
  }

  const startRenaming = (task: Task) => {
    setEditingTaskId(String(task.task_id))
    setEditingTitle(task.title || "Untitled Task")
  }

  const submitRename = async (taskId: string) => {
    const trimmedTitle = editingTitle.trim()
    const task = tasks.find(t => String(t.task_id) === String(taskId))

    // Do not call API if title is empty or unchanged
    if (!trimmedTitle || (task && task.title === trimmedTitle)) {
      setEditingTaskId(null)
      return
    }

    try {
      const response = await apiRequest(`${getApiUrl()}/api/chat/task/${taskId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ title: trimmedTitle })
      })

      if (response.ok) {
        setTasks(prev => prev.map(t =>
          String(t.task_id) === String(taskId)
            ? { ...t, title: trimmedTitle }
            : t
        ))
      }
    } catch (error) {
      console.error('Failed to rename task:', error)
    } finally {
      setEditingTaskId(null)
    }
  }

  const cancelRename = () => {
    setEditingTaskId(null)
    setEditingTitle("")
  }

  const [isExpanded, setIsExpanded] = useState(false)
  const [expandedMenus, setExpandedMenus] = useState<string[]>(["/agent"]) // Use href as a stable key
  const [showUserMenu, setShowUserMenu] = useState(false)
  const [isAboutOpen, setIsAboutOpen] = useState(false)
  const sidebarRef = useRef<HTMLDivElement | null>(null)
  const contentScrollRef = useRef<HTMLDivElement | null>(null)
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
    // Match /task/[id] pattern
    const match = pathname.match(/^\/task\/([^/]+)\/?$/);
    if (match) {
      return match[1];
    }
    return null;
  }, [pathname])

  const [tasks, setTasks] = useState<Task[]>([])
  const [unreadTasks, setUnreadTasks] = useState<Set<string>>(new Set())
  const taskStatusRef = useRef<Map<string, string>>(new Map())
  const [versionInfo, setVersionInfo] = useState<VersionInfo | null>(null)
  const [isLoadingTasks, setIsLoadingTasks] = useState(false)
  const [isHistoryExpanded, setIsHistoryExpanded] = useState(true)
  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const pathnameRef = useRef(pathname)
  const pageRef = useRef(page)
  const displayVersion = versionInfo?.display_version || "unknown"

  // Search state
  const [searchQuery, setSearchQuery] = useState("")
  const searchRef = useRef("")
  const [isSearchFocused, setIsSearchFocused] = useState(false)
  const [isSearchVisible, setIsSearchVisible] = useState(false)

  // Rename state
  const [editingTaskId, setEditingTaskId] = useState<string | null>(null)
  const [editingTitle, setEditingTitle] = useState("")

  // Loading state ref for polling interval
  const loadingRef = useRef({ isLoadingTasks, isLoadingMore })
  loadingRef.current = { isLoadingTasks, isLoadingMore }

  useEffect(() => {
    pathnameRef.current = pathname
  }, [pathname])

  useEffect(() => {
    pageRef.current = page
  }, [page])

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
  const loadTasks = useCallback(async (pageNum = 1, isAppending = false, isPolling = false) => {
    if (isAppending) {
      setIsLoadingMore(true)
    } else if (!isPolling) {
      setIsLoadingTasks(true)
    }

    try {
      const searchParam = searchRef.current ? `&search=${encodeURIComponent(searchRef.current)}` : ''
      const response = await apiRequest(`${getApiUrl()}/api/chat/tasks?page=${pageNum}&per_page=${TASKS_PER_PAGE}${searchParam}`)
      if (response.ok) {
        const data = await response.json()
        // Handle new API response format {tasks: [...], pagination: {...}}
        const newTasks = data.tasks || (Array.isArray(data) ? data : [])

        // Update task status ref and check for unread completed tasks
        const currentUnreadUpdates = new Set<string>()
        const match = pathnameRef.current.match(/^\/task\/([^/]+)\/?$/)
        const currentTaskId = match ? match[1] : null

        newTasks.forEach((task: Task) => {
          const stringTaskId = String(task.task_id)
          const prevStatus = taskStatusRef.current.get(stringTaskId)
          // If task completed and wasn't completed before (and we have a previous record)
          if (task.status === 'completed' && prevStatus && prevStatus !== 'completed') {
            // Only mark as unread if we are not currently on this task page
            if (String(currentTaskId) !== stringTaskId) {
              currentUnreadUpdates.add(stringTaskId)
            }
          }
          taskStatusRef.current.set(stringTaskId, task.status)
        })

        if (currentUnreadUpdates.size > 0) {
          setUnreadTasks(prev => {
            const next = new Set(prev)
            currentUnreadUpdates.forEach(id => next.add(id))
            return next
          })
        }

        const totalPages = data.pagination?.total_pages || 1
        const loadedPage = isPolling ? Math.min(pageRef.current, totalPages) : pageNum

        if (isPolling) {
          setTasks(prev => {
            const newTaskIds = new Set(newTasks.map((t: Task) => String(t.task_id)))
            const remainingTasks = prev
              .slice(Math.min(TASKS_PER_PAGE, prev.length))
              .filter(t => !newTaskIds.has(String(t.task_id)))

            // Polling only refreshes page 1, so replace that slice and trim retained pages
            // to the current loaded page when the server reports fewer total pages.
            return [...newTasks, ...remainingTasks].slice(0, loadedPage * TASKS_PER_PAGE)
          })
        } else if (isAppending) {
          setTasks(prev => [...prev, ...newTasks])
        } else {
          setTasks(newTasks)
        }

        // Update pagination status
        if (isPolling) {
          // Polling always refreshes page 1, so keep the user's loaded page state intact.
          setHasMore(loadedPage < totalPages)

          if (loadedPage !== pageRef.current) {
            setPage(loadedPage)
          }
        } else {
          setHasMore(pageNum < totalPages)
          setPage(pageNum)
        }
      }
    } catch (error) {
      console.error('Failed to load tasks:', error)
    } finally {
      setIsLoadingTasks(false)
      setIsLoadingMore(false)
    }
  }, [])

  // Poll for task updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Only poll if window is visible and not already loading
      if (document.visibilityState === 'visible' && !loadingRef.current.isLoadingTasks && !loadingRef.current.isLoadingMore) {
        loadTasks(1, false, true)
      }
    }, 30000) // Poll every 30 seconds

    return () => clearInterval(interval)
  }, [loadTasks])

  // Clear unread status when entering a task page
  useEffect(() => {
    const currentTaskId = getCurrentTaskId()
    if (currentTaskId) {
      setUnreadTasks(prev => {
        if (!prev.has(String(currentTaskId))) return prev

        const next = new Set(prev)
        next.delete(String(currentTaskId))
        return next
      })
    }
  }, [pathname, getCurrentTaskId])

  // Monitor task list changes, if content is not enough to fill the container and there is more data, automatically load the next page
  useEffect(() => {
    if (!contentScrollRef.current || !isHistoryExpanded) return

    const { scrollHeight, clientHeight } = contentScrollRef.current
    const isVisible = contentScrollRef.current.getClientRects().length > 0
    if (!isVisible || clientHeight <= 0) return

    // If content height is less than or equal to container height (plus a buffer), and there is more data, and not loading
    if (scrollHeight <= clientHeight + 20 && hasMore && !isLoadingMore && !isLoadingTasks) {
      // Use setTimeout to avoid continuous state updates in one render cycle
      const timer = setTimeout(() => {
        loadTasks(page + 1, true)
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [tasks, hasMore, isLoadingMore, isLoadingTasks, page, loadTasks, isHistoryExpanded])

  useEffect(() => {
    if (isHistoryExpanded) {
      loadTasks(1, false)
    }
  }, [isHistoryExpanded, loadTasks, state.lastTaskUpdate])

  // Debounce search query
  useEffect(() => {
    const timer = setTimeout(() => {
      if (searchRef.current !== searchQuery) {
        searchRef.current = searchQuery

        // Auto-expand when searching
        if (searchQuery && !isHistoryExpanded) {
          setIsHistoryExpanded(true)
        } else if (isHistoryExpanded) {
          loadTasks(1, false)
        }
      }
    }, 500)
    return () => clearTimeout(timer)
  }, [searchQuery, loadTasks, isHistoryExpanded])

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    if (!isHistoryExpanded) return

    const { scrollTop, scrollHeight, clientHeight } = e.currentTarget
    if (clientHeight <= 0) return

    if (scrollHeight - scrollTop <= clientHeight + 20 && hasMore && !isLoadingMore && !isLoadingTasks) {
      loadTasks(page + 1, true)
    }
  }

  // Sidebar is hidden by default on Agent pages, but kept visible on Vibe and Build pages, and shown on other pages
  // For agent pages, sidebar is only shown when isExpanded is true
  // Build page no longer automatically hides
  // /agent/[id] page does not auto-collapse (for agent chat)
  const isAgentChatPage = pathname.match(/^\/agent\/\d+$/)
  const isAgentPage = (pathname.startsWith('/agent') && !pathname.startsWith('/agent/vibe') && !isAgentChatPage)
  const shouldShowSidebar = !isAgentPage || isExpanded || !allowCollapse

  // Allow user to manually collapse the sidebar
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)

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

  const isPathActive = (href: string) => {
    if (href === "/") {
      return pathname === "/"
    }
    return pathname === href || pathname.startsWith(`${href}/`)
  }

  const isItemActive = (item: NavigationItem) => {
    if (item.children?.length) {
      return item.children.some((child: NavigationItem) => isPathActive(child.href))
    }
    return isPathActive(item.href)
  }

  if (allowCollapse && ((isAgentPage && !shouldShowSidebar) || !isSidebarOpen)) {
    return (
      <div className="flex flex-col items-center justify-start py-3 w-[54px] bg-secondary border-r border-border shrink-0 h-full relative">
        <Link href="/task" className="flex items-center justify-center mb-6">
          <img
            src={branding.logoPath}
            alt={branding.logoAlt}
            className="h-7 w-7 rounded-lg"
          />
        </Link>
        <button
          onClick={() => isAgentPage ? setIsExpanded(true) : setIsSidebarOpen(true)}
          className="absolute -right-3 top-5 bg-card border border-border rounded-full p-0.5 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors z-50 shadow-sm"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
        <div className="flex flex-col gap-1 w-full px-1.5">
          {navigationGroups.map((group) => (
            group.items.map((item: NavigationItem) => {
              const isActive = isItemActive(item)
              const hasChildren = item.children && item.children.length > 0

              if (hasChildren) {
                return (
                  <button
                    key={item.name}
                    type="button"
                    title={item.nameKey ? t(item.nameKey) : item.name}
                    onClick={() => isAgentPage ? setIsExpanded(true) : setIsSidebarOpen(true)}
                    className={cn(
                      "flex w-full items-center justify-center p-2 rounded-[7px] transition-colors",
                      isActive ? "bg-primary/[0.09] text-[hsl(var(--sidebar-active-text))]" : "text-muted-foreground hover:bg-accent hover:text-foreground"
                    )}
                  >
                    <item.icon className="h-4 w-4" />
                  </button>
                )
              }

              return (
                <Link
                  key={item.name}
                  href={item.href}
                  title={item.nameKey ? t(item.nameKey) : item.name}
                  className={cn(
                    "flex items-center justify-center p-2 rounded-[7px] transition-colors",
                    isActive ? "bg-primary/[0.09] text-[hsl(var(--sidebar-active-text))]" : "text-muted-foreground hover:bg-accent hover:text-foreground"
                  )}
                >
                  <item.icon className="h-4 w-4" />
                </Link>
              )
            })
          ))}
        </div>
        <div className="mt-auto w-full px-1.5 border-t border-border pt-3">
          <button
            onClick={() => isAgentPage ? setIsExpanded(true) : setIsSidebarOpen(true)}
            className="flex items-center justify-center w-full p-2 hover:bg-accent rounded-[7px] transition-colors"
            title={user?.username || t('sidebar.user.defaultName')}
          >
            <div className="h-7 w-7 rounded-full bg-primary flex items-center justify-center text-xs font-semibold text-white uppercase shrink-0">
              {(user?.username || t('sidebar.user.defaultName')).charAt(0)}
            </div>
          </button>
        </div>
      </div>
    )
  }

  return (
    <div ref={sidebarRef} className={cn(
      "flex flex-col bg-secondary border-r border-border transition-all duration-300 shrink-0",
      isAgentPage ? "h-full" : "h-full",
      shouldShowSidebar ? "w-60" : "w-0",
      className
    )}>
      {/* Logo */}
      <div className="flex h-[90px] items-center justify-between px-4 relative">
        <Link href="/" className="flex items-center gap-2.5">
          <img
            src={branding.logoPath}
            alt={branding.logoAlt}
            className="h-12 w-12 rounded-lg"
          />
          <h1 className="text-[32px] font-bold tracking-tight" style={{ color: "#2745A6" }}>{branding.appName}</h1>
        </Link>
        {allowCollapse && (
          <button
            onClick={() => isAgentPage ? setIsExpanded(false) : setIsSidebarOpen(false)}
            className="absolute -right-3 top-4 bg-card border border-border rounded-full p-0.5 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors z-50 shadow-sm"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Navigation */}
      <div
        ref={contentScrollRef}
        onScroll={handleScroll}
        className="flex-1 flex flex-col min-h-0 overflow-y-auto px-[10px]"
      >
        {/* Sticky Navigation Groups */}
        <nav
          className="z-10 bg-transparent py-1"
        >
          {/* Groups */}
          {navigationGroups.map((group, groupIndex) => (
            <div key={group.title} className={cn("mb-4", groupIndex === 0 && "mt-0")}>
              <div className="px-2 pb-1.5 text-[10.5px] font-semibold text-muted-foreground uppercase tracking-[0.08em]">
                {group.titleKey ? t(group.titleKey) : group.title}
              </div>
              <div className="space-y-0.5">
                {group.items.map((item: NavigationItem) => {
                  const isActive = isItemActive(item)
                  const hasChildren = item.children && item.children.length > 0
                  const isExpanded = isMenuExpanded(item.href)

                  const activeStyle = "bg-primary/[0.09] text-[hsl(var(--sidebar-active-text))] font-semibold rounded-[7px]"
                  const inactiveStyle = "text-muted-foreground hover:bg-accent hover:text-foreground rounded-[7px]"

                  if (hasChildren) {
                    return (
                      <Popover key={item.name} open={isExpanded} onOpenChange={() => toggleMenu(item.href)}>
                        <PopoverTrigger asChild>
                          <button
                            className={cn(
                              "group flex items-center justify-between px-2.5 py-2 text-[13.5px] transition-colors relative w-full",
                              isActive ? activeStyle : inactiveStyle
                            )}
                          >
                            <div className="flex items-center gap-2.5">
                              <item.icon className={cn("h-4 w-4", isActive ? "text-[hsl(var(--sidebar-active-text))]" : "text-muted-foreground")} />
                              {item.nameKey ? t(item.nameKey) : item.name}
                            </div>
                            <ChevronRight className="h-3 w-3 opacity-50" />
                          </button>
                        </PopoverTrigger>
                        {item.children && (
                          <PopoverContent
                            side="right"
                            align="start"
                            sideOffset={8}
                            className="w-56 rounded-xl border border-border bg-popover p-2 shadow-lg"
                          >
                            <div className="space-y-0.5">
                              {item.children.map((child: NavigationItem) => {
                                const isChildActive = pathname === child.href
                                return (
                                  <Link
                                    key={child.href}
                                    href={child.href}
                                    onClick={() => toggleMenu(item.href)}
                                    className={cn(
                                      "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13.5px] font-medium transition-colors",
                                      isChildActive
                                        ? "bg-primary/[0.09] text-[hsl(var(--sidebar-active-text))]"
                                        : "text-foreground hover:bg-accent"
                                    )}
                                  >
                                    <child.icon className="h-4 w-4 text-muted-foreground" />
                                    {child.nameKey ? t(child.nameKey) : child.name}
                                  </Link>
                                )
                              })}
                            </div>
                          </PopoverContent>
                        )}
                      </Popover>
                    )
                  }

                  return (
                    <Link
                      key={item.name}
                      href={item.href}
                      className={cn(
                        "group flex items-center px-2.5 py-2 text-[13.5px] font-medium transition-colors",
                        isActive ? activeStyle : inactiveStyle
                      )}
                    >
                      <item.icon className={cn("h-4 w-4 mr-2.5", isActive ? "text-[hsl(var(--sidebar-active-text))]" : "text-muted-foreground")} />
                      {item.nameKey ? t(item.nameKey) : item.name}
                    </Link>
                  )
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* History Section */}
        <div className="flex flex-col overflow-hidden shrink-0 border-t border-border pt-1 pb-4">
          <div
            className="px-2 pt-4 pb-1.5 text-[10.5px] font-semibold text-muted-foreground uppercase tracking-[0.08em] flex items-center justify-between h-8 shrink-0"
          >
            {(isSearchVisible || isSearchFocused || searchQuery) ? (
              <div className="flex-1 relative mr-2 h-full flex items-center">
                <SearchInput
                  placeholder={t('nav.search')}
                  value={searchQuery}
                  onChange={setSearchQuery}
                  onFocus={() => setIsSearchFocused(true)}
                  onBlur={() => {
                    setIsSearchFocused(false)
                    setIsSearchVisible(false)
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Escape') {
                      setSearchQuery('')
                      setIsSearchVisible(false)
                      setIsSearchFocused(false)
                      e.currentTarget.blur()
                    }
                  }}
                  containerClassName="w-full h-7"
                  className="h-7 text-[12px] text-foreground bg-transparent border-border focus:border-primary [&::-webkit-search-cancel-button]:hidden"
                  autoFocus
                />
                <button
                  className={cn(
                    "absolute right-1.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 p-1 rounded-full transition-colors",
                    !searchQuery && "opacity-0 pointer-events-none"
                  )}
                  onMouseDown={(e) => {
                    // Prevent the default behavior to avoid triggering the input’s blur event
                    e.preventDefault()
                  }}
                  onClick={(e) => {
                    e.stopPropagation()
                    setSearchQuery('')
                    setIsSearchVisible(false)
                    setIsSearchFocused(false)
                  }}
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ) : (
              <div className="flex-1 flex items-center min-w-0 mr-2">
                <span className="truncate">{t('nav.history')}</span>
                <div
                  className="cursor-pointer p-1 hover:bg-accent rounded transition-colors text-muted-foreground hover:text-foreground flex-shrink-0"
                  onClick={() => setIsSearchVisible(true)}
                >
                  <Search className="h-3 w-3" />
                </div>
              </div>
            )}
            <div
              className="cursor-pointer p-1 -mr-1 hover:bg-accent rounded transition-colors ml-1"
              onClick={() => setIsHistoryExpanded(!isHistoryExpanded)}
            >
              {isHistoryExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            </div>
          </div>

          {isHistoryExpanded && (
            <div
              className="space-y-1"
            >
              {isLoadingTasks ? (
                <div className="flex items-center justify-center py-4">
                  <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                </div>
              ) : tasks.length > 0 ? (
                <>
                  {tasks.map(task => {
                    const currentTaskId = getCurrentTaskId();
                    const compactChannelName = task.channel_name
                      ? getCompactChannelName(task.channel_name, task.channel_type)
                      : null
                    return (
                      <Link
                        key={task.task_id}
                        href={`/task/${task.task_id}`}
                        title={task.title}
                        className={cn(
                          "group flex items-center px-2.5 py-[5px] text-[12.5px] font-medium transition-colors truncate relative pr-8 rounded-[7px]",
                          String(currentTaskId) === String(task.task_id)
                            ? "bg-primary/[0.09] text-[hsl(var(--sidebar-active-text))]"
                            : "text-muted-foreground hover:bg-accent hover:text-foreground"
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
                            {(task.status === 'paused' || task.status === 'waiting_for_user') && <PauseCircle className="h-4 w-4 text-yellow-500" />}
                            {task.status === 'pending' && <Loader2 className="h-4 w-4 animate-spin text-gray-400" />}
                          </div>
                        </div>
                        {editingTaskId === String(task.task_id) ? (
                          <div className="flex-1 mr-2" onClick={(e) => { e.preventDefault(); e.stopPropagation(); }}>
                            <input
                              type="text"
                              value={editingTitle}
                              onChange={(e) => setEditingTitle(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') submitRename(String(task.task_id));
                                if (e.key === 'Escape') cancelRename();
                              }}
                              onBlur={() => submitRename(String(task.task_id))}
                              autoFocus
                              className="w-full bg-transparent border-b border-primary outline-none text-sm text-foreground"
                            />
                          </div>
                        ) : (
                          <span className="truncate flex-1 text-left flex items-center gap-2">
                            <span className="truncate">{task.title || "Untitled Task"}</span>
                            {task.channel_name && compactChannelName && (
                              <span
                                className="inline-flex max-w-[88px] flex-shrink-0 items-center gap-1 rounded border border-border/50 bg-accent/50 px-1.5 text-[10px] text-muted-foreground"
                                title={getChannelTooltip(task.channel_name, task.channel_type)}
                                aria-label={getChannelTooltip(task.channel_name, task.channel_type)}
                              >
                                <ChannelTypeIcon channelType={task.channel_type} />
                                <span className="truncate">{compactChannelName}</span>
                              </span>
                            )}
                          </span>
                        )}
                        {unreadTasks.has(String(task.task_id)) && (
                          <div className="absolute right-4 top-1/2 -translate-y-1/2 h-2 w-2 rounded-full bg-primary group-hover:opacity-0 transition-opacity" />
                        )}
                        <div className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity" onClick={(e) => { e.preventDefault(); e.stopPropagation(); }}>
                          <Popover>
                            <PopoverTrigger asChild>
                              <MoreHorizontal className="text-muted-foreground/60 h-4 w-4 hover:text-foreground" />
                            </PopoverTrigger>
                            <PopoverContent align="end" className="w-32 p-1" onClick={(e) => { e.preventDefault(); e.stopPropagation(); }}>
                              <div className="flex flex-col">
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    startRenaming(task);
                                  }}
                                  className="flex w-full items-center px-2 py-1.5 text-sm hover:bg-accent rounded-sm transition-colors text-left"
                                >
                                  <Edit2 className="h-3.5 w-3.5 mr-2" />
                                  {t('common.rename')}
                                </button>
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    deleteTask(String(task.task_id), e);
                                  }}
                                  className="flex w-full items-center px-2 py-1.5 text-sm text-red-500 hover:bg-red-50 dark:hover:bg-red-900/10 rounded-sm transition-colors text-left"
                                >
                                  <Trash2 className="h-3.5 w-3.5 mr-2" />
                                  {t('common.delete')}
                                </button>
                              </div>
                            </PopoverContent>
                          </Popover>
                        </div>
                      </Link>
                    )
                  })}
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
      </div>

      {/* User Profile */}
      <div className="px-[10px] py-3 border-t border-border relative mt-auto shrink-0" ref={userMenuRef}>
        {showUserMenu && (
          <div className="absolute bottom-full left-4 right-4 mb-2 bg-popover border border-border rounded-lg shadow-lg overflow-hidden animate-in fade-in zoom-in-95 duration-200 z-50">
            <div className="py-1">
              {getUserMenuItemsForUser(user).map((item: NavigationItem) => (
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
          className="flex w-full items-center gap-2.5 hover:bg-accent px-2 py-2 rounded-[7px] transition-colors text-left"
        >
          <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center text-xs font-semibold text-white uppercase shrink-0">
            {(user?.username || t('sidebar.user.defaultName')).charAt(0)}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-[13px] font-semibold text-foreground truncate">{user?.username || t('sidebar.user.defaultName')}</p>
          </div>
          <ChevronsUpDown className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
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

      <ConfirmDialog
        isOpen={!!taskToDelete}
        onOpenChange={(open) => !open && setTaskToDelete(null)}
        onConfirm={confirmDeleteTask}
        isLoading={isDeletingTask}
      />
    </div >
  )
}
