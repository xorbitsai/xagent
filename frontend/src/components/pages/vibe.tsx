"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { getApiUrl, getAuthHeaders } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useAuth } from "@/contexts/auth-context"
import { useI18n } from "@/contexts/i18n-context"
import {
  MessageSquare,
  Search,
  Plus,
  Clock,
  CheckCircle,
  XCircle,
  PlayCircle,
  Trash2,
  Zap,
  FileText,
  BarChart3,
  Image,
  Bot,
  Presentation,
  FileSpreadsheet,
  Palette,
  Workflow,
  Database,
  TrendingUp,
  Users,
  Building2,
  ShoppingCart,
  Target,
  Calendar,
  Mail,
  Phone,
  Globe,
  Map,
  CreditCard,
  AlertTriangle,
  CheckSquare,
  Download,
  Upload,
  Settings,
  MoreHorizontal
} from "lucide-react"
import Link from "next/link"

interface Task {
  task_id: string
  title: string
  status: "completed" | "running" | "failed" | "pending"
  created_at: string | number
  description?: string
  vibe_mode?: "task" | "process"
}

export function VibePage() {
  const { token } = useAuth()
  const { t } = useI18n()
  const [tasks, setTasks] = useState<Task[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [isLoading, setIsLoading] = useState(true)
  const [currentPage, setCurrentPage] = useState(1)
  const [totalTasks, setTotalTasks] = useState(0)
  const [totalPages, setTotalPages] = useState(0)
  const tasksPerPage = 10

  useEffect(() => {
    loadTasks()
  }, [currentPage])

  useEffect(() => {
    // Reset to first page when search query changes
    setCurrentPage(1)
  }, [searchQuery])

  const loadTasks = async () => {
    try {
      const params = new URLSearchParams({
        page: currentPage.toString(),
        per_page: tasksPerPage.toString()
      })

      if (searchQuery) {
        params.append('search', searchQuery)
      }

      // Show only STANDARD agent tasks in vibe page
      params.append('agent_type', 'standard')

      const response = await apiRequest(`${getApiUrl()}/api/chat/tasks?${params}`, {
        headers: {}
      })

      if (response.ok) {
        const data = await response.json()

        if (data.tasks && Array.isArray(data.tasks)) {
          setTasks(data.tasks)
          setTotalTasks(data.pagination?.total_count || 0)
          setTotalPages(data.pagination?.total_pages || 1)
        } else {
          // Handle legacy API response format
          const tasksArray = Array.isArray(data) ? data : []
          setTasks(tasksArray)
          setTotalTasks(tasksArray.length)
          setTotalPages(Math.ceil(tasksArray.length / tasksPerPage) || 1)
        }
      }
    } catch (error) {
      console.error('Failed to load tasks:', error)
    } finally {
      setIsLoading(false)
    }
  }

  // totalPages is now set from API response

  const handlePageChange = (page: number) => {
    setCurrentPage(page)
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-400" />
      case "running":
        return <PlayCircle className="h-4 w-4 text-blue-400" />
      case "failed":
        return <XCircle className="h-4 w-4 text-red-400" />
      default:
        return <Clock className="h-4 w-4 text-muted-foreground" />
    }
  }

  const getStatusBadgeText = (status: string, t: ReturnType<typeof useI18n>["t"]) => {
    const variants = {
      completed: "bg-green-500/20 text-green-400 border border-green-500/30",
      running: "bg-blue-500/20 text-blue-400 border border-blue-500/30",
      failed: "bg-red-500/20 text-red-400 border border-red-500/30",
      pending: "bg-muted/50 text-muted-foreground border border-muted"
    }

    return (
      <Badge variant="outline" className={variants[status as keyof typeof variants]}>
        {t(`vibe.list.status.${status}`)}
      </Badge>
    )
  }

  const deleteTask = async (taskId: string) => {
    if (!confirm(t('vibe.actions.delete_confirm'))) return

    try {
      const response = await apiRequest(`${getApiUrl()}/api/chat/task/${taskId}`, {
        method: 'DELETE',
        headers: {}
      })

      if (response.ok) {
        setTasks(prev => prev.filter(task => task.task_id !== taskId))
      }
    } catch (error) {
      console.error('Failed to delete task:', error)
    }
  }

  return (
    <div className="h-full overflow-auto bg-background">
      <div className="p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">{t('vibe.title')}</h1>
          <p className="text-muted-foreground">{t('vibe.description')}</p>
        </div>

        {/* Quick Examples */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-foreground">{t('vibe.actions.you_can')}</h2>
            <Link href="/debug">
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                {t('vibe.actions.new_task')}
              </Button>
            </Link>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <Card className="p-5 hover:shadow-lg transition-all duration-300 cursor-pointer bg-gradient-to-br from-card/80 via-card/60 to-card/40 backdrop-blur-sm border-border/30 hover:border-primary/30 hover:shadow-primary/10 group relative overflow-hidden">
              {/* Background decoration */}
              <div className="absolute inset-0 bg-gradient-to-br from-transparent via-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              {/* Light border */}
              <div className="absolute inset-0 rounded-lg border border-transparent group-hover:border-primary/20 transition-all duration-300"></div>

              <div className="flex items-center mb-3 relative z-10">
                <div className="p-2 rounded-lg bg-blue-500/10 mr-3 group-hover:scale-110 group-hover:rotate-6 transition-transform duration-300">
                  <Presentation className="h-5 w-5 text-blue-400" />
                </div>
                <h3 className="font-semibold text-foreground group-hover:text-blue-400 transition-colors">{t('vibe.quick_examples.make_ppt.title')}</h3>
              </div>
              <p className="text-sm text-muted-foreground relative z-10 group-hover:text-foreground/80 transition-colors">{t('vibe.quick_examples.make_ppt.description')}</p>
            </Card>

            <Card className="p-5 hover:shadow-lg transition-all duration-300 cursor-pointer bg-gradient-to-br from-card/80 via-card/60 to-card/40 backdrop-blur-sm border-border/30 hover:border-primary/30 hover:shadow-primary/10 group relative overflow-hidden">
              {/* Background decoration */}
              <div className="absolute inset-0 bg-gradient-to-br from-transparent via-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              {/* Light border */}
              <div className="absolute inset-0 rounded-lg border border-transparent group-hover:border-primary/20 transition-all duration-300"></div>

              <div className="flex items-center mb-3 relative z-10">
                <div className="p-2 rounded-lg bg-emerald-500/10 mr-3 group-hover:scale-110 group-hover:rotate-6 transition-transform duration-300">
                  <BarChart3 className="h-5 w-5 text-emerald-400" />
                </div>
                <h3 className="font-semibold text-foreground group-hover:text-emerald-400 transition-colors">{t('vibe.quick_examples.data_analysis.title')}</h3>
              </div>
              <p className="text-sm text-muted-foreground relative z-10 group-hover:text-foreground/80 transition-colors">{t('vibe.quick_examples.data_analysis.description')}</p>
            </Card>

            <Card className="p-5 hover:shadow-lg transition-all duration-300 cursor-pointer bg-gradient-to-br from-card/80 via-card/60 to-card/40 backdrop-blur-sm border-border/30 hover:border-primary/30 hover:shadow-primary/10 group relative overflow-hidden">
              {/* Background decoration */}
              <div className="absolute inset-0 bg-gradient-to-br from-transparent via-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              {/* Light border */}
              <div className="absolute inset-0 rounded-lg border border-transparent group-hover:border-primary/20 transition-all duration-300"></div>

              <div className="flex items-center mb-3 relative z-10">
                <div className="p-2 rounded-lg bg-purple-500/10 mr-3 group-hover:scale-110 group-hover:rotate-6 transition-transform duration-300">
                  <Palette className="h-5 w-5 text-purple-400" />
                </div>
                <h3 className="font-semibold text-foreground group-hover:text-purple-400 transition-colors">{t('vibe.quick_examples.design_poster.title')}</h3>
              </div>
              <p className="text-sm text-muted-foreground relative z-10 group-hover:text-foreground/80 transition-colors">{t('vibe.quick_examples.design_poster.description')}</p>
            </Card>

            <Card className="p-5 hover:shadow-lg transition-all duration-300 cursor-pointer bg-gradient-to-br from-card/80 via-card/60 to-card/40 backdrop-blur-sm border-border/30 hover:border-primary/30 hover:shadow-primary/10 group relative overflow-hidden">
              {/* Background decoration */}
              <div className="absolute inset-0 bg-gradient-to-br from-transparent via-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              {/* Light border */}
              <div className="absolute inset-0 rounded-lg border border-transparent group-hover:border-primary/20 transition-all duration-300"></div>

              <div className="flex items-center mb-3 relative z-10">
                <div className="p-2 rounded-lg bg-orange-500/10 mr-3 group-hover:scale-110 group-hover:rotate-6 transition-transform duration-300">
                  <Workflow className="h-5 w-5 text-orange-400" />
                </div>
                <h3 className="font-semibold text-foreground group-hover:text-orange-400 transition-colors">{t('vibe.quick_examples.automation.title')}</h3>
              </div>
              <p className="text-sm text-muted-foreground relative z-10 group-hover:text-foreground/80 transition-colors">{t('vibe.quick_examples.automation.description')}</p>
            </Card>

            <Card className="p-5 hover:shadow-lg transition-all duration-300 cursor-pointer bg-gradient-to-br from-card/80 via-card/60 to-card/40 backdrop-blur-sm border-border/30 hover:border-primary/30 hover:shadow-primary/10 group relative overflow-hidden">
              {/* Background decoration */}
              <div className="absolute inset-0 bg-gradient-to-br from-transparent via-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              {/* Light border */}
              <div className="absolute inset-0 rounded-lg border border-transparent group-hover:border-primary/20 transition-all duration-300"></div>

              <div className="flex items-center mb-3 relative z-10">
                <div className="p-2 rounded-lg bg-gray-500/10 mr-3 group-hover:scale-110 group-hover:rotate-6 transition-transform duration-300">
                  <MoreHorizontal className="h-5 w-5 text-gray-400" />
                </div>
                <h3 className="font-semibold text-foreground group-hover:text-gray-400 transition-colors">{t('vibe.quick_examples.more.title')}</h3>
              </div>
              <p className="text-sm text-muted-foreground relative z-10 group-hover:text-foreground/80 transition-colors">{t('vibe.quick_examples.more.description')}</p>
            </Card>
          </div>
        </div>

        {/* Tasks List */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-foreground">{t('vibe.list.title')}</h2>
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder={t('vibe.list.search_placeholder')}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9 w-64 bg-card border-border text-foreground placeholder:text-muted-foreground"
                />
              </div>
            </div>
          </div>

          <Card className="bg-card border-border">
            <div className="divide-y divide-border">
              {isLoading ? (
                <div className="p-6 text-center text-muted-foreground">
                  {t('common.loading')}
                </div>
              ) : tasks.length === 0 ? (
                <div className="p-6 text-center text-muted-foreground">
                  {t('vibe.list.empty')}
                </div>
              ) : (
                tasks.map((task) => (
                  <div key={task.task_id} className="p-4 hover:bg-accent transition-colors group">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3 flex-1">
                        {getStatusIcon(task.status)}
                        <div className="flex-1 min-w-0">
                          <Link
                            href={`/debug?id=${task.task_id}`}
                            className="text-sm font-medium text-foreground hover:text-primary truncate block transition-colors group-hover:translate-x-1"
                          >
                            {task.title}
                          </Link>
                          <p className="text-xs text-muted-foreground">
                            {new Date(task.created_at).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        {/* Vibe Mode Badge */}
                        {task.vibe_mode === 'process' ? (
                          <Badge variant="outline" className="text-xs bg-purple-500/10 text-purple-400 border-purple-500/30">
                            <Workflow className="h-3 w-3 mr-1" />
                            {t('vibe.list.mode.process')}
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/30">
                            <Target className="h-3 w-3 mr-1" />
                            {t('vibe.list.mode.task')}
                          </Badge>
                        )}
                        {getStatusBadgeText(task.status, t)}
                        <button
                          onClick={() => deleteTask(task.task_id)}
                          className="text-muted-foreground hover:text-destructive transition-colors p-1 rounded hover:bg-destructive/10"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="p-4 border-t border-border">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-muted-foreground">
                    {t('vibe.list.pagination.summary', {
                      from: ((currentPage - 1) * tasksPerPage) + 1,
                      to: Math.min(currentPage * tasksPerPage, totalTasks),
                      total: totalTasks,
                    })}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange(currentPage - 1)}
                      disabled={currentPage === 1}
                      className="h-8 px-3"
                    >
                      {t('vibe.list.pagination.prev')}
                    </Button>
                    <div className="flex items-center gap-1">
                      {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                        let pageNum
                        if (totalPages <= 5) {
                          pageNum = i + 1
                        } else if (currentPage <= 3) {
                          pageNum = i + 1
                        } else if (currentPage >= totalPages - 2) {
                          pageNum = totalPages - 4 + i
                        } else {
                          pageNum = currentPage - 2 + i
                        }

                        return (
                          <Button
                            key={pageNum}
                            variant={currentPage === pageNum ? "default" : "outline"}
                            size="sm"
                            onClick={() => handlePageChange(pageNum)}
                            className="h-8 w-8 p-0"
                          >
                            {pageNum}
                          </Button>
                        )
                      })}
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange(currentPage + 1)}
                      disabled={currentPage === totalPages}
                      className="h-8 px-3"
                    >
                      {t('vibe.list.pagination.next')}
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}
