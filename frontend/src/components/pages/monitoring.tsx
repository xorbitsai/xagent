"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import {
  Activity,
  BarChart3,
  Zap,
  Clock,
  TrendingUp,
  TrendingDown,
  Users,
  Server,
  Cpu
} from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"

interface Stats {
  totalCalls: number
  successRate: number
  avgResponseTime: number
  activeModels: number
  totalTokens: number
  todayCalls: number
}

interface PopularTool {
  name: string
  description: string
  usage_count: number
  avg_duration: number
}

interface ModelStat {
  name: string
  status: string
  usage_rate: number
  success_rate: number
  total_tasks: number
  successful_tasks: number
  failed_tasks: number
}

export function MonitoringPage() {
  const [stats, setStats] = useState<Stats>({
    totalCalls: 0,
    successRate: 0,
    avgResponseTime: 0,
    activeModels: 0,
    totalTokens: 0,
    todayCalls: 0
  })
  const [popularTools, setPopularTools] = useState<PopularTool[]>([])
  const [modelStats, setModelStats] = useState<ModelStat[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const { t } = useI18n()

  useEffect(() => {
    loadMonitoringData()
  }, [])

  const loadMonitoringData = async () => {
    setIsLoading(true)
    try {
      // Load all data in parallel
      const [statsResponse, popularToolsResponse, modelStatsResponse] = await Promise.all([
        apiRequest(`${getApiUrl()}/api/monitor/stats`),
        apiRequest(`${getApiUrl()}/api/monitor/popular-tools`),
        apiRequest(`${getApiUrl()}/api/monitor/model-stats`)
      ])

      if (statsResponse.ok) {
        const statsData = await statsResponse.json()
        setStats(statsData)
      }

      if (popularToolsResponse.ok) {
        const toolsData = await popularToolsResponse.json()
        setPopularTools(toolsData)
      }

      if (modelStatsResponse.ok) {
        const modelsData = await modelStatsResponse.json()
        setModelStats(modelsData)
      }
    } catch (error) {
      console.error('Failed to load monitoring data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="h-full overflow-auto bg-[#0E1117]">
      <div className="p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-[22px] font-bold leading-tight">{t('monitoring.title')}</h1>
          <p className="text-[13px] text-muted-foreground mt-0.5">{t('monitoring.description')}</p>
        </div>

        {/* Stats cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mb-8">
          <Card className="p-6 bg-[#161B22] border-[rgba(255,255,255,0.08)]">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-[#8B949E] mb-1">{t('monitoring.cards.totalCalls')}</p>
                <p className="text-2xl font-bold text-[#E6EDF3]">
                  {isLoading ? "..." : stats.totalCalls.toLocaleString()}
                </p>
              </div>
              <div className="p-3 rounded-lg bg-blue-500/10">
                <Zap className="h-6 w-6 text-blue-500" />
              </div>
            </div>
          </Card>

          <Card className="p-6 bg-[#161B22] border-[rgba(255,255,255,0.08)]">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-[#8B949E] mb-1">{t('monitoring.cards.successRate')}</p>
                <p className="text-2xl font-bold text-[#E6EDF3]">
                  {isLoading ? "..." : stats.successRate}%
                </p>
              </div>
              <div className="p-3 rounded-lg bg-green-500/10">
                <TrendingUp className="h-6 w-6 text-green-500" />
              </div>
            </div>
          </Card>

          <Card className="p-6 bg-[#161B22] border-[rgba(255,255,255,0.08)]">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-[#8B949E] mb-1">{t('monitoring.cards.avgResponseTime')}</p>
                <p className="text-2xl font-bold text-[#E6EDF3]">
                  {isLoading ? "..." : stats.avgResponseTime}s
                </p>
              </div>
              <div className="p-3 rounded-lg bg-purple-500/10">
                <Clock className="h-6 w-6 text-purple-500" />
              </div>
            </div>
          </Card>

          <Card className="p-6 bg-[#161B22] border-[rgba(255,255,255,0.08)]">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-[#8B949E] mb-1">{t('monitoring.cards.todayCalls')}</p>
                <p className="text-2xl font-bold text-[#E6EDF3]">
                  {isLoading ? "..." : stats.todayCalls.toLocaleString()}
                </p>
              </div>
              <div className="p-3 rounded-lg bg-orange-500/10">
                <Activity className="h-6 w-6 text-orange-500" />
              </div>
            </div>
          </Card>
        </div>

        {/* Stats content */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <Card className="bg-[#161B22] border-[rgba(255,255,255,0.08)]">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-[#E6EDF3] mb-4">{t('monitoring.models.title')}</h3>
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {isLoading ? (
                  <div className="text-center text-[#8B949E]">{t('common.loading')}</div>
                ) : modelStats.length === 0 ? (
                  <div className="text-center text-[#8B949E]">{t('monitoring.models.empty')}</div>
                ) : (
                  modelStats.map((model, index) => (
                    <div key={index} className="flex items-center justify-between p-4 bg-[#0E1117] rounded-lg">
                      <div className="flex items-center space-x-3">
                        <Cpu className={`h-5 w-5 ${
                          model.status === 'running' ? 'text-green-500' :
                          model.status === 'idle' ? 'text-blue-500' : 'text-gray-500'
                        }`} />
                        <div>
                          <p className="font-medium text-[#E6EDF3]">{model.name}</p>
                          <p className="text-sm text-[#8B949E]">
                            {t('monitoring.models.summary', { success: model.success_rate, total: model.total_tasks })}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-lg font-semibold text-[#E6EDF3]">{model.usage_rate}%</p>
                        <p className="text-sm text-[#8B949E]">{t('monitoring.models.usageRate')}</p>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </Card>

          {/* Tool Usage */}
          <Card className="bg-[#161B22] border-[rgba(255,255,255,0.08)]">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-[#E6EDF3] mb-4">{t('monitoring.tools.title')}</h3>
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {isLoading ? (
                  <div className="text-center text-[#8B949E]">{t('common.loading')}</div>
                ) : popularTools.length === 0 ? (
                  <div className="text-center text-[#8B949E]">{t('monitoring.tools.empty')}</div>
                ) : (
                  popularTools.map((tool, index) => (
                    <div key={index} className="flex items-center justify-between p-4 bg-[#0E1117] rounded-lg">
                      <div className="flex items-center space-x-3">
                        <Zap className="h-5 w-5 text-yellow-500" />
                        <div>
                          <p className="font-medium text-[#E6EDF3]">{tool.name}</p>
                          <p className="text-sm text-[#8B949E]">{t('monitoring.tools.avgDuration', { duration: tool.avg_duration })}</p>
                        </div>
                      </div>
                      <Badge className="bg-yellow-500/10 text-yellow-500">
                        {t('monitoring.tools.usageCount', { count: tool.usage_count })}
                      </Badge>
                    </div>
                  ))
                )}
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
