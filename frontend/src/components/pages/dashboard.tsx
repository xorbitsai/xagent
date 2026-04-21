"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { getApiUrl, getAuthHeaders } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useAuth } from "@/contexts/auth-context"
import { getBrandingFromEnv } from "@/lib/branding"
import { useI18n } from "@/contexts/i18n-context"
import {
  MessageSquare,
  FileText,
  Search,
  Plus,
  Activity,
  BarChart3,
  Clock,
  CheckCircle,
  XCircle,
  PlayCircle,
  Trash2,
  Users,
  Zap,
  TrendingUp,
  Server,
  Settings,
  Cpu,
  ArrowRight,
  Wand2,
  Rocket,
  Target,
  Sparkles,
  Brain,
  Workflow
} from "lucide-react"
import Link from "next/link"

interface Task {
  task_id: string
  title: string
  status: "completed" | "running" | "failed" | "pending"
  created_at: string | number
  description?: string
  execution_mode?: "flash" | "balanced" | "think"
}

interface FileItem {
  filename: string
  file_size: number
  uploaded_at: string
}

export function DashboardPage() {
  const { token } = useAuth()
  const branding = getBrandingFromEnv()
  const { t } = useI18n()
  const [tasks, setTasks] = useState<Task[]>([])
  const [files, setFiles] = useState<FileItem[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    try {
      // Load recent tasks
      const tasksResponse = await apiRequest(`${getApiUrl()}/api/chat/tasks?per_page=10&agent_type=standard`, {
        headers: {}
      })

      if (tasksResponse.ok) {
        const tasksData = await tasksResponse.json()
        // Handle both new format (with pagination) and old format (direct array)
        if (tasksData.tasks && Array.isArray(tasksData.tasks)) {
          setTasks(tasksData.tasks)
        } else if (Array.isArray(tasksData)) {
          setTasks(tasksData.slice(0, 10))
        } else {
          setTasks([])
        }
      }

      // Load recent files
      const filesResponse = await apiRequest(`${getApiUrl()}/api/files/list`, {
        headers: {}
      })
      if (filesResponse.ok) {
        const filesData = await filesResponse.json()
        if (filesData && filesData.files) {
          setFiles(filesData.files.slice(0, 8))
        }
      }
    } catch (error) {
      console.error('Failed to load dashboard data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const filteredTasks = tasks.filter(task =>
    task.title.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "running":
        return <PlayCircle className="h-4 w-4 text-blue-500" />
      case "failed":
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusBadge = (status: string) => {
    const variants = {
      completed: "bg-green-500/10 text-green-500",
      running: "bg-blue-500/10 text-blue-500",
      failed: "bg-red-500/10 text-red-500",
      pending: "bg-gray-500/10 text-gray-500"
    }

    return (
      <Badge className={variants[status as keyof typeof variants]}>
        {t(`dashboard.tasks.status.${status}`)}
      </Badge>
    )
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const deleteTask = async (taskId: string) => {
    if (!confirm(t('dashboard.tasks.delete_confirm'))) return

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

  // Statistics data status
  const [dashboardStats, setDashboardStats] = useState({
    totalTasks: 0,
    activeAgents: 0,
    deployedApps: 0,
    todayCalls: 0
  })
  const [isLoadingStats, setIsLoadingStats] = useState(true)

  // Fetch dashboard statistics
  useEffect(() => {
    const loadDashboardStats = async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/monitor/dashboard-stats`, {
        headers: {}
      })
        if (response.ok) {
          const stats = await response.json()
          setDashboardStats(stats)
        }
      } catch (error) {
        console.error('Failed to load dashboard stats:', error)
      } finally {
        setIsLoadingStats(false)
      }
    }

    loadDashboardStats()
  }, [])

  // Statistics data
  const stats = [
    {
      title: t("dashboard.stats.total_tasks"),
      value: isLoadingStats ? "..." : dashboardStats.totalTasks,
      icon: MessageSquare,
      color: "text-blue-500",
      bgColor: "bg-blue-500/10"
    },
    {
      title: t("dashboard.stats.active_agents"),
      value: isLoadingStats ? "..." : dashboardStats.activeAgents,
      icon: Users,
      color: "text-green-500",
      bgColor: "bg-green-500/10"
    },
    {
      title: t("dashboard.stats.deployed_apps"),
      value: isLoadingStats ? "..." : dashboardStats.deployedApps,
      icon: Server,
      color: "text-purple-500",
      bgColor: "bg-purple-500/10"
    },
    {
      title: t("dashboard.stats.today_calls"),
      value: isLoadingStats ? "..." : dashboardStats.todayCalls,
      icon: Zap,
      color: "text-orange-500",
      bgColor: "bg-orange-500/10"
    }
  ]

  // Main features
  const mainFeatures = [
    {
      title: t("dashboard.features.vibe.title"),
      description: t("dashboard.features.vibe.description"),
      icon: Wand2,
      href: "/agent",
      color: "bg-purple-500/10 text-purple-500"
    },
    {
      title: t("dashboard.features.build.title"),
      description: t("dashboard.features.build.description"),
      icon: Zap,
      href: "/build",
      color: "bg-yellow-500/10 text-yellow-500"
    },
    {
      title: t("dashboard.features.deploy.title"),
      description: t("dashboard.features.deploy.description"),
      icon: Rocket,
      href: "/deploy",
      color: "bg-green-500/10 text-green-500"
    },
    {
      title: t("dashboard.features.models.title"),
      description: t("dashboard.features.models.description"),
      icon: Cpu,
      href: "/models",
      color: "bg-indigo-500/10 text-indigo-500"
    },
    {
      title: t("dashboard.features.monitoring.title"),
      description: t("dashboard.features.monitoring.description"),
      icon: Activity,
      href: "/monitoring",
      color: "bg-orange-500/10 text-orange-500"
    },
    {
      title: t("dashboard.features.files.title"),
      description: t("dashboard.features.files.description"),
      icon: FileText,
      href: "/files",
      color: "bg-blue-500/10 text-blue-500"
    }
  ]

  return (
    <div className="h-full overflow-auto bg-[#0E1117]">
      <div className="p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-[#E6EDF3] mb-2">{branding.appName}</h1>
          <p className="text-lg text-[#8B949E]">{process.env.NEXT_PUBLIC_APP_SUBTITLE ? branding.subtitle : t('branding.subtitle')}</p>
        </div>

        {/* Statistics cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {stats.map((stat, index) => {
            // Define different gradient backgrounds for each statistic card
            const getStatGradient = () => {
              if (stat.color.includes("blue")) return "from-blue-500/10 via-blue-600/5 to-transparent";
              if (stat.color.includes("green")) return "from-green-500/10 via-green-600/5 to-transparent";
              if (stat.color.includes("purple")) return "from-purple-500/10 via-purple-600/5 to-transparent";
              if (stat.color.includes("orange")) return "from-orange-500/10 via-orange-600/5 to-transparent";
              return "from-primary/10 via-primary/5 to-transparent";
            };

            return (
              <Card key={index} className={`p-6 bg-gradient-to-br ${getStatGradient()} backdrop-blur-sm border-border/30 hover:border-border/80 transition-all duration-300 hover:shadow-xl hover:shadow-primary/5 hover:-translate-y-1 group relative overflow-hidden`}>
                {/* Background decoration */}
                <div className="absolute inset-0 bg-gradient-to-br from-transparent via-primary/3 to-transparent opacity-50"></div>
                {/* Light effect border */}
                <div className="absolute inset-0 rounded-lg border border-transparent group-hover:border-primary/20 transition-all duration-300"></div>

                <div className="flex items-center justify-between relative z-10">
                  <div>
                    <p className="text-sm text-muted-foreground mb-1 group-hover:text-foreground/80 transition-colors">{stat.title}</p>
                    <p className="text-2xl font-bold text-foreground group-hover:text-primary transition-colors">{stat.value}</p>
                  </div>
                  <div className={`p-3 rounded-lg ${stat.bgColor} bg-opacity-40 backdrop-blur-sm group-hover:scale-110 group-hover:rotate-3 transition-transform duration-300 shadow-lg ring-1 ring-white/10`}>
                    <stat.icon className={`h-6 w-6 ${stat.color}`} />
                  </div>
                </div>
              </Card>
            );
          })}
        </div>

        {/* Main features */}
        <div className="mb-8">
          <h2 className="text-2xl font-semibold mb-6 text-foreground">{t('dashboard.features.title')}</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {mainFeatures.map((feature, index) => {
              // Determine hover text color based on color
              const getHoverColor = () => {
                if (feature.color.includes("purple")) return "group-hover:text-purple-400";
                if (feature.color.includes("yellow")) return "group-hover:text-yellow-400";
                if (feature.color.includes("indigo")) return "group-hover:text-indigo-400";
                if (feature.color.includes("green")) return "group-hover:text-green-400";
                if (feature.color.includes("orange")) return "group-hover:text-orange-400";
                if (feature.color.includes("gray")) return "group-hover:text-gray-400";
                return "group-hover:text-blue-400";
              };

              // Define different gradient backgrounds for each feature card
              const getFeatureGradient = () => {
                if (feature.color.includes("purple")) return "from-purple-500/15 via-purple-600/8 to-indigo-500/5";
                if (feature.color.includes("yellow")) return "from-yellow-500/15 via-amber-600/8 to-orange-500/5";
                if (feature.color.includes("indigo")) return "from-indigo-500/15 via-blue-600/8 to-cyan-500/5";
                if (feature.color.includes("green")) return "from-green-500/15 via-emerald-600/8 to-teal-500/5";
                if (feature.color.includes("orange")) return "from-orange-500/15 via-red-600/8 to-rose-500/5";
                return "from-blue-500/15 via-cyan-600/8 to-sky-500/5";
              };

              return (
                <Link key={index} href={feature.href}>
                  <Card className={`p-6 hover:shadow-2xl transition-all duration-300 cursor-pointer bg-gradient-to-br ${getFeatureGradient()} backdrop-blur-sm border-border/30 hover:border-primary/30 hover:shadow-primary/10 group relative overflow-hidden`}>
                    {/* Dynamic background decoration */}
                    <div className="absolute inset-0 bg-gradient-to-br from-transparent via-white/2 to-transparent opacity-30"></div>
                    {/* Geometric decoration */}
                    <div className="absolute top-2 right-2 w-16 h-16 rounded-full bg-gradient-to-br from-white/5 to-transparent opacity-20 group-hover:opacity-40 transition-opacity duration-500"></div>
                    <div className="absolute bottom-2 left-2 w-12 h-12 rounded-lg bg-gradient-to-tr from-white/3 to-transparent opacity-15 group-hover:opacity-30 transition-opacity duration-500"></div>
                    {/* Light effect border */}
                    <div className="absolute inset-0 rounded-lg border border-transparent group-hover:border-primary/20 transition-all duration-300"></div>

                    <div className="flex items-center justify-between mb-4 relative z-10">
                      <div className="flex items-center">
                        <div className={`p-3 rounded-lg ${feature.color} bg-opacity-50 backdrop-blur-sm mr-4 group-hover:scale-125 group-hover:rotate-6 transition-transform duration-300 shadow-lg group-hover:shadow-xl ring-1 ring-white/20`}>
                          <feature.icon className="h-6 w-6" />
                        </div>
                        <h3 className={`text-lg font-semibold text-foreground ${getHoverColor()} transition-colors duration-300 group-hover:translate-x-1`}>{feature.title}</h3>
                      </div>
                      <ArrowRight className={`h-5 w-5 text-muted-foreground opacity-0 group-hover:opacity-100 ${getHoverColor()} transition-all duration-300 group-hover:translate-x-2`} />
                    </div>
                    <p className="text-sm text-muted-foreground relative z-10 group-hover:text-foreground/80 transition-colors duration-300">{feature.description}</p>
                  </Card>
                </Link>
              );
            })}
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Recent Tasks */}
          <div className="lg:col-span-2">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-foreground">{t('dashboard.tasks.title')}</h2>
              <div className="flex items-center gap-2">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder={t('dashboard.tasks.search_placeholder')}
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9 w-64 bg-card border-border text-foreground placeholder:text-muted-foreground"
                  />
                </div>
                <Link href="/agent">
                  <Button size="sm">
                    <Plus className="h-4 w-4 mr-2" />
                    {t('dashboard.tasks.new_task')}
                  </Button>
                </Link>
              </div>
            </div>

            <Card className="bg-card border-border">
              <div className="divide-y divide-border">
                {isLoading ? (
                  <div className="p-6 text-center text-muted-foreground">
                    {t('common.loading')}
                  </div>
                ) : filteredTasks.length === 0 ? (
                  <div className="p-6 text-center text-muted-foreground">
                    {t('dashboard.tasks.empty')}
                  </div>
                ) : (
                  filteredTasks.map((task) => (
                    <div key={task.task_id} className="p-4 hover:bg-accent transition-colors">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3 flex-1">
                          {getStatusIcon(task.status)}
                          <div className="flex-1 min-w-0">
                            <Link
                              href={`/agent?id=${task.task_id}`}
                              className="text-sm font-medium text-foreground hover:text-primary truncate block"
                            >
                              {task.title}
                            </Link>
                            <p className="text-xs text-muted-foreground">
                              {new Date(task.created_at).toLocaleString()}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          {/* Execution Mode Badge */}
                          {task.execution_mode === 'flash' ? (
                            <Badge variant="outline" className="text-xs bg-yellow-500/10 text-yellow-400 border-yellow-500/30">
                              <Zap className="h-3 w-3 mr-1" />
                              {t('vibe.list.mode.flash')}
                            </Badge>
                          ) : task.execution_mode === 'balanced' ? (
                            <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/30">
                              <Sparkles className="h-3 w-3 mr-1" />
                              {t('vibe.list.mode.balanced')}
                            </Badge>
                          ) : task.execution_mode === 'think' ? (
                            <Badge variant="outline" className="text-xs bg-purple-500/10 text-purple-400 border-purple-500/30">
                              <Brain className="h-3 w-3 mr-1" />
                              {t('vibe.list.mode.think')}
                            </Badge>
                          ) : (
                            <Badge variant="outline" className="text-xs bg-gray-500/10 text-gray-400 border-gray-500/30">
                              <Target className="h-3 w-3 mr-1" />
                              {t('vibe.list.mode.unknown')}
                            </Badge>
                          )}
                          {getStatusBadge(task.status)}
                          <button
                            onClick={() => deleteTask(task.task_id)}
                            className="text-muted-foreground hover:text-destructive"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </Card>
          </div>

          {/* Files */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-foreground">{t('dashboard.files.title')}</h2>
              <Link href="/files">
                <Button variant="outline" size="sm" className="border-border text-foreground hover:bg-accent">
                  {t('dashboard.files.view_all')}
                </Button>
              </Link>
            </div>

            <Card className="bg-card border-border">
              <div className="divide-y divide-border">
                {files.length === 0 ? (
                  <div className="p-6 text-center text-muted-foreground">
                    {t('dashboard.files.empty')}
                  </div>
                ) : (
                  files.map((file, index) => (
                    <div key={index} className="p-4 hover:bg-accent transition-colors">
                      <div className="flex items-center space-x-3">
                        <FileText className="h-4 w-4 text-muted-foreground" />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-foreground truncate">
                            {file.filename}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {formatFileSize(file.file_size)} • {new Date(file.uploaded_at).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
