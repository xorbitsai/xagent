"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Settings, X, Loader2, ArrowRight } from "lucide-react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useAuth } from "@/contexts/auth-context"
import { Select } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { useI18n } from "@/contexts/i18n-context"
import {
  InfoTooltip,
} from "@/components/ui/tooltip"

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

interface AgentConfig {
  model: string
  smallFastModel?: string
  visualModel?: string
  compactModel?: string
  memorySimilarityThreshold?: number
}

interface ConfigDialogProps {
  onConfigChange: (config: AgentConfig) => void
  currentConfig?: AgentConfig
  trigger?: React.ReactNode
}

export function ConfigDialog({ onConfigChange, currentConfig, trigger }: ConfigDialogProps) {
  const router = useRouter();
  const [open, setOpen] = useState(false)
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(false)
  const [config, setConfig] = useState<AgentConfig>({
    model: currentConfig?.model || "",
    smallFastModel: currentConfig?.smallFastModel,
    visualModel: currentConfig?.visualModel,
    compactModel: currentConfig?.compactModel,
    memorySimilarityThreshold: currentConfig?.memorySimilarityThreshold ?? 1.5
  })
  const { t } = useI18n()

  // Fetch models when dialog opens
  useEffect(() => {
    if (open) {
      fetchModels()
    }
  }, [open])

  const fetchModels = async () => {
    try {
      setLoading(true)
      const apiUrl = getApiUrl()

      // Fetch user default models first
      const defaultResponse = await apiRequest(`${apiUrl}/api/models/user-default`, {
        headers: {}
      })

      let defaultModels: Record<string, any> = {}
      if (defaultResponse.ok) {
        const defaults = await defaultResponse.json()
        if (Array.isArray(defaults)) {
          defaults.forEach((defaultConfig: any) => {
            if (defaultConfig && defaultConfig.config_type && defaultConfig.model) {
              defaultModels[defaultConfig.config_type] = defaultConfig.model
            }
          })
        }
      }

      // Fetch all models
      const modelsResponse = await apiRequest(`${apiUrl}/api/models/?category=llm`, {
        headers: {}
      })
      if (!modelsResponse.ok) {
        throw new Error("Failed to fetch models")
      }
      const data = await modelsResponse.json()
      setModels(data)

      // Auto-select default models if none selected
      if (data.length > 0) {
        const defaultModel = defaultModels.general || data.find((m: any) => m.is_default) || data[0]
        setConfig(prev => ({
          ...prev,
          model: prev.model || defaultModel.model_id,
          smallFastModel: prev.smallFastModel || defaultModels.small_fast?.model_id,
          visualModel: prev.visualModel || defaultModels.visual?.model_id,
          compactModel: prev.compactModel || defaultModels.compact?.model_id
        }))
      }
    } catch (error) {
      console.error('Failed to fetch models:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleModelSelect = (modelId: string) => {
    const newConfig = { ...config, model: modelId }
    setConfig(newConfig)
    onConfigChange(newConfig)
  }

  const handleSmallFastModelSelect = (modelId: string) => {
    const newConfig = { ...config, smallFastModel: modelId || undefined }
    setConfig(newConfig)
    onConfigChange(newConfig)
  }

  const handleVisualModelSelect = (modelId: string) => {
    const newConfig = { ...config, visualModel: modelId || undefined }
    setConfig(newConfig)
    onConfigChange(newConfig)
  }

  const handleCompactModelSelect = (modelId: string) => {
    const newConfig = { ...config, compactModel: modelId || undefined }
    setConfig(newConfig)
    onConfigChange(newConfig)
  }

  const handleClose = () => {
    setOpen(false)
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {trigger || (
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground hover:bg-muted rounded-md"
            title={t('agent.input.actions.config')}
          >
            <Settings className="h-3.5 w-3.5" />
          </Button>
        )}
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px] max-h-[85vh] flex flex-col p-0 z-[1000]">
        <div className="p-6 pb-0">
          <DialogHeader className="text-left">
            <DialogTitle>{t('agent.configDialog.title')}</DialogTitle>
            <DialogDescription>
              {t('agent.configDialog.description')}
            </DialogDescription>
          </DialogHeader>
        </div>

        <div className="flex-1 overflow-y-auto px-6 space-y-6">
          {/* Model Selection */}
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium">{t('agent.configDialog.modelSelect.label')}</label>
              <p className="text-xs text-muted-foreground mt-1">
                {t('agent.configDialog.modelSelect.description')}
              </p>
            </div>

            {loading ? (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="h-5 w-5 animate-spin text-primary" />
                <span className="ml-2 text-sm text-muted-foreground">{t('agent.configDialog.modelSelect.loading')}</span>
              </div>
            ) : models.length === 0 ? (
              <div className="text-center py-4">
                <p className="text-sm text-muted-foreground">{t('agent.configDialog.modelSelect.empty.title')}</p>
                <p className="text-xs text-muted-foreground mt-1">{t('agent.configDialog.modelSelect.empty.hint')}</p>
                <Button
                  variant="outline"
                  size="icon"
                  className="mt-4"
                  onClick={() => router.push("/models")}
                  title={t('agent.configDialog.modelSelect.empty.button')}
                >
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Main model selection */}
                <div className="space-y-2">
                  <div className="flex items-center gap-1.5">
                    <Label className="text-sm">
                      {t('agent.configDialog.modelSelect.main.label')}
                    </Label>
                    <InfoTooltip content={t('agent.configDialog.modelSelect.main.hint')} />
                  </div>
                  <Select
                    value={config.model}
                    onValueChange={handleModelSelect}
                    options={models.map(m => ({
                      value: m.model_id,
                      label: m.model_name,
                      description: `${m.model_name}${m.description ? ' - ' + m.description : ''} (${m.model_provider}) [${m.model_id}]`,
                      isDefault: m.is_default,
                      isSmallFast: m.is_small_fast,
                      isVisual: m.is_visual
                    }))}
                    placeholder={t('agent.configDialog.modelSelect.main.placeholder')}
                  />
                </div>

                {/* Small/fast model selection (optional) */}
                <div className="space-y-2">
                  <div className="flex items-center gap-1.5">
                    <Label className="text-sm">
                      {t('agent.configDialog.modelSelect.smallFast.label')}
                    </Label>
                    <InfoTooltip content={t('agent.configDialog.modelSelect.smallFast.hint')} />
                  </div>
                  <Select
                    value={config.smallFastModel || ""}
                    onValueChange={handleSmallFastModelSelect}
                    options={[
                      { value: "", label: t('agent.configDialog.modelSelect.smallFast.options.noneLabel'), description: t('agent.configDialog.modelSelect.smallFast.options.noneDescription') },
                      ...models.map(m => ({
                        value: m.model_id,
                        label: m.model_name,
                        description: `${m.model_name}${m.description ? ' - ' + m.description : ''} (${m.model_provider}) [${m.model_id}]${m.is_small_fast ? t('agent.configDialog.modelSelect.smallFast.options.tagFast') : ''}`,
                        isDefault: m.is_default,
                        isSmallFast: m.is_small_fast,
                        isVisual: m.is_visual
                      }))
                    ]}
                    placeholder={t('agent.configDialog.modelSelect.smallFast.placeholder')}
                  />
                </div>

                {/* Visual model selection (optional) */}
                <div className="space-y-2">
                  <div className="flex items-center gap-1.5">
                    <Label className="text-sm">
                      {t('agent.configDialog.modelSelect.visual.label')}
                    </Label>
                    <InfoTooltip content={t('agent.configDialog.modelSelect.visual.hint')} />
                  </div>
                  <Select
                    value={config.visualModel || ""}
                    onValueChange={handleVisualModelSelect}
                    options={[
                      { value: "", label: t('agent.configDialog.modelSelect.visual.options.noneLabel'), description: t('agent.configDialog.modelSelect.visual.options.noneDescription') },
                      ...models.map(m => ({
                        value: m.model_id,
                        label: m.model_name,
                        description: `${m.model_name}${m.description ? ' - ' + m.description : ''} (${m.model_provider})[${m.model_id}]${m.is_visual ? t('agent.configDialog.modelSelect.visual.options.tagVisual') : ''}`,
                        isDefault: m.is_default,
                        isSmallFast: m.is_small_fast,
                        isVisual: m.is_visual
                      }))
                    ]}
                    placeholder={t('agent.configDialog.modelSelect.visual.placeholder')}
                  />
                </div>

                {/* Long context model selection (optional) */}
                <div className="space-y-2">
                  <div className="flex items-center gap-1.5">
                    <Label className="text-sm">
                      {t('agent.configDialog.modelSelect.compact.label')}
                    </Label>
                    <InfoTooltip content={t('agent.configDialog.modelSelect.compact.hint')} />
                  </div>
                  <Select
                    value={config.compactModel || ""}
                    onValueChange={handleCompactModelSelect}
                    options={[
                      { value: "", label: t('agent.configDialog.modelSelect.compact.options.noneLabel'), description: t('agent.configDialog.modelSelect.compact.options.noneDescription') },
                      ...models.map(m => ({
                        value: m.model_id,
                        label: m.model_name,
                        description: `${m.model_name}${m.description ? ' - ' + m.description : ''} (${m.model_provider})[${m.model_id}]${m.is_compact ? t('agent.configDialog.modelSelect.compact.options.tagCompact') : ''}`,
                        isDefault: m.is_default,
                        isSmallFast: m.is_small_fast,
                        isVisual: m.is_visual,
                        isCompact: m.is_compact
                      }))
                    ]}
                    placeholder={t('agent.configDialog.modelSelect.compact.placeholder')}
                  />
                </div>

                {/* Memory similarity threshold configuration */}
                <div className="space-y-2">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">{t('agent.configDialog.memoryThreshold.label')}</Label>
                      <Badge variant="outline" className="text-xs">
                        {config.memorySimilarityThreshold?.toFixed(1) ?? "1.5"}
                      </Badge>
                    </div>
                    <Slider
                      value={[config.memorySimilarityThreshold ?? 1.5]}
                      onValueChange={(value) => setConfig(prev => ({ ...prev, memorySimilarityThreshold: value[0] }))}
                      max={2.0}
                      min={0.1}
                      step={0.1}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>{t('agent.configDialog.memoryThreshold.strict')}</span>
                      <span>{t('agent.configDialog.memoryThreshold.loose')}</span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {t('agent.configDialog.memoryThreshold.hint')}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
          </div>

        <div className="p-6 pt-0">
          <div className="flex justify-end pt-4 border-t">
            <Button variant="outline" onClick={handleClose}>
              <X className="h-4 w-4 mr-1" />
              {t('agent.configDialog.buttons.close')}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
