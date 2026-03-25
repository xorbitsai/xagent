"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Settings, Cpu, Zap, Eye } from "lucide-react"
import { cn, getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useState, useEffect } from "react"
import { useI18n } from "@/contexts/i18n-context";

interface Model {
  id: number
  model_id: string
  model_provider: string
  model_name: string
  base_url?: string
  temperature?: number
  is_default: boolean
  is_small_fast: boolean
  is_visual: boolean
  is_compact: boolean
  description?: string
  created_at?: string
  updated_at?: string
  is_active: boolean
}

interface Task {
  id: string
  title: string
  status: "pending" | "running" | "completed" | "failed" | "paused"
  description: string
  createdAt: string | number
  updatedAt: string | number
  modelId?: string
  smallFastModelId?: string
  visualModelId?: string
  compactModelId?: string
  modelName?: string
  smallFastModelName?: string
  visualModelName?: string
}

interface ModelInfoDisplayProps {
  currentTask?: Task | null
  onConfigChange?: () => void
  className?: string
}

export function ModelInfoDisplay({ currentTask, onConfigChange, className }: ModelInfoDisplayProps) {
  const [models, setModels] = useState<Model[]>([])
  const { t } = useI18n();

  // Fetch models for mapping
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/models/?category=llm`)
        if (response.ok) {
          const data = await response.json()
          setModels(data)
        }
      } catch (error) {
        console.error('Failed to fetch models:', error)
      }
    }

    // Only fetch if we have a task with model info
    if (
      currentTask &&
      (
        currentTask.modelId ||
        currentTask.smallFastModelId ||
        currentTask.visualModelId ||
        currentTask.compactModelId ||
        currentTask.modelName ||
        currentTask.smallFastModelName ||
        currentTask.visualModelName
      )
    ) {
      fetchModels()
    }
  }, [currentTask])

  const modelIdToNameMap = models.reduce((acc, model) => {
    acc[model.model_id] = model.model_name
    return acc
  }, {} as Record<string, string>)

  const getDisplayName = (modelId?: string, modelName?: string) => {
    if (modelName) return modelName
    if (modelId) return modelIdToNameMap[modelId] || modelId
    return null
  }

  const getTitle = (label: string, modelId?: string, modelName?: string) => {
    const displayName = getDisplayName(modelId, modelName)
    if (!displayName) return label
    if (modelId && modelName && modelId !== modelName) {
      return `${label}: ${modelName} (${modelId})`
    }
    return `${label}: ${displayName}`
  }

  if (
    !currentTask ||
    (
      !currentTask.modelId &&
      !currentTask.smallFastModelId &&
      !currentTask.visualModelId &&
      !currentTask.modelName &&
      !currentTask.smallFastModelName &&
      !currentTask.visualModelName
    )
  ) {
    // If no task or no model info, show config button
    // But if onConfigChange is undefined, don't show button (handled by parent)
    if (!onConfigChange) {
      return null
    }
    return (
      <Button
        variant="ghost"
        size="sm"
        onClick={onConfigChange}
        className={cn(
          "h-7 w-7 p-0 text-muted-foreground hover:text-foreground hover:bg-muted rounded-md",
          className
        )}
        title={t('agent.input.actions.config')}
        aria-label={t('agent.input.actions.config')}
      >
        <Settings className="h-3.5 w-3.5" />
      </Button>
    )
  }

  const mainModelDisplay = getDisplayName(currentTask.modelId, currentTask.modelName)
  const smallFastModelDisplay = getDisplayName(currentTask.smallFastModelId, currentTask.smallFastModelName)
  const visualModelDisplay = getDisplayName(currentTask.visualModelId, currentTask.visualModelName)

  return (
    <div className="flex items-center gap-2">
      {/* Main model display */}
      {mainModelDisplay && (
        <Badge
          variant="secondary"
          className="text-xs bg-blue-500/10 text-blue-600 border-blue-500/20 flex items-center gap-1"
          title={getTitle(t('agent.configDialog.modelSelect.main.label'), currentTask.modelId, currentTask.modelName)}
        >
          <Cpu className="h-3 w-3" />
          {mainModelDisplay}
        </Badge>
      )}

      {/* Fast model display */}
      {smallFastModelDisplay && (
        <Badge
          variant="secondary"
          className="text-xs bg-green-500/10 text-green-600 border-green-500/20 flex items-center gap-1"
          title={getTitle(t('agent.configDialog.modelSelect.smallFast.label'), currentTask.smallFastModelId, currentTask.smallFastModelName)}
        >
          <Zap className="h-3 w-3" />
          {smallFastModelDisplay}
        </Badge>
      )}

      {/* Visual model display */}
      {visualModelDisplay && (
        <Badge
          variant="secondary"
          className="text-xs bg-purple-500/10 text-purple-600 border-purple-500/20 flex items-center gap-1"
          title={getTitle(t('agent.configDialog.modelSelect.visual.label'), currentTask.visualModelId, currentTask.visualModelName)}
        >
          <Eye className="h-3 w-3" />
          {visualModelDisplay}
        </Badge>
      )}

      {/* Config button */}
      {onConfigChange && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onConfigChange}
          className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground hover:bg-muted rounded-md"
          title={t('agent.config.title')}
          aria-label={t('agent.config.title')}
        >
          <Settings className="h-3.5 w-3.5" />
        </Button>
      )}
    </div>
  )
}
