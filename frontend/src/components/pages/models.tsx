"use client"

import { useState, useEffect, useMemo, ReactNode } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { getApiUrl } from "@/lib/utils"
import { useAuth } from "@/contexts/auth-context"
import { apiRequest } from "@/lib/api-wrapper"
import { getSupportedProviders } from "@/lib/models"
import {
  Plus,
  Brain,
  Image as ImageIcon,
  Star,
  Zap,
  Box,
  Settings,
  CheckCircle2,
} from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { ModelManagementDialog } from "./model-management-dialog"
import { toast } from "sonner"

export function getModelDetailUrl(modelId: string): string {
  return `${getApiUrl()}/api/models/by-id/${encodeURIComponent(modelId)}`
}

function randomHex8(): string {
  try {
    const bytes = new Uint8Array(4)
    crypto.getRandomValues(bytes)
    return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("")
  } catch {
    return Math.random().toString(16).slice(2, 10).padEnd(8, "0")
  }
}

export function generateModelId(modelName: string, modelProvider: string, userId?: string): string {
  const suffix = randomHex8()
  // Fallback to 'user' if ID is missing to prevent "undefined" in string
  const uid = userId || 'unknown'
  return `${modelName}-${modelProvider}-${uid}-${suffix}`
}

// Interfaces from models-1.tsx
export interface Model {
  id: number
  model_id: string
  category: string
  model_provider: string
  model_name: string
  base_url?: string
  temperature?: number
  dimension?: number
  abilities?: string[]
  description?: string
  created_at?: string
  updated_at?: string
  is_active: boolean
  is_owner: boolean
  can_edit: boolean
  can_delete: boolean
  is_shared: boolean
}

export interface ModelCreate {
  model_id: string
  category: string
  model_provider: string
  model_name: string
  api_key: string
  base_url?: string
  temperature?: number
  dimension?: number
  abilities?: string[]
  share_with_users?: boolean
  default_config_types?: string[]
}

// Provider Config from models.tsx
export interface ProviderConfig {
  id: string
  name: string
  description: string
  icon: ReactNode
  defaultBaseUrl?: string
  categoryBaseUrls?: Record<string, string>
  category: string[]
  requires_base_url?: boolean
}

const LOCAL_PROVIDER_CONFIGS: Record<string, Partial<ProviderConfig>> = {
  openai: {
    icon: <img src="/openai.svg" alt="OpenAI" className="w-6 h-6" />,
    category: ["llm", "embedding"],
    defaultBaseUrl: "https://api.openai.com/v1",
  },
  "minimax-coding-plan": {
    icon: <img src="/minimax.svg" alt="MiniMax" className="w-6 h-6" />,
    category: ["llm"],
    defaultBaseUrl: "https://api.minimax.io/anthropic"
  },
  "minimax-cn-coding-plan": {
    icon: <img src="/minimax.svg" alt="MiniMax" className="w-6 h-6" />,
    category: ["llm"],
    defaultBaseUrl: "https://api.minimaxi.com/anthropic"
  },
  "kimi-for-coding": {
    icon: <img src="/kimi.svg" alt="Kimi" className="w-6 h-6" />,
    category: ["llm"],
    defaultBaseUrl: "https://api.kimi.com/coding"
  },
  "zai-coding-plan": {
    icon: <img src="/zhipu.svg" alt="Z.AI" className="w-6 h-6" />,
    category: ["llm"],
    defaultBaseUrl: "https://api.z.ai/api/coding/paas/v4"
  },
  "zhipuai-coding-plan": {
    icon: <img src="/zhipu.svg" alt="Zhipu" className="w-6 h-6" />,
    category: ["llm"],
    defaultBaseUrl: "https://open.bigmodel.cn/api/coding/paas/v4"
  },
  "alibaba-coding-plan": {
    icon: <img src="/dashscope.png" alt="Alibaba Bailian" className="w-6 h-6" />,
    category: ["llm"],
    defaultBaseUrl: "https://coding-intl.dashscope.aliyuncs.com/v1"
  },
  "alibaba-coding-plan-cn": {
    icon: <img src="/dashscope.png" alt="Alibaba Bailian" className="w-6 h-6" />,
    category: ["llm"],
    defaultBaseUrl: "https://coding.dashscope.aliyuncs.com/v1"
  },
  azure_openai: {
    icon: <Zap className="w-6 h-6 text-blue-500" />,
    category: ["llm"]
    // No default base url for Azure, user must provide
  },
  zhipu: {
    icon: <img src="/zhipu.svg" alt="Zhipu" className="w-6 h-6" />,
    category: ["llm"],
    defaultBaseUrl: "https://open.bigmodel.cn/api/paas/v4",
  },
  dashscope: {
    icon: <img src="/dashscope.png" alt="DashScope" className="w-6 h-6" />,
    category: ["embedding", "image"],
    defaultBaseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    categoryBaseUrls: {
      embedding: "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
      image: "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    }
  },
  gemini: {
    icon: <img src="/gemini.svg" alt="Gemini" className="w-6 h-6" />,
    category: ["llm", "image"],
    defaultBaseUrl: "https://generativelanguage.googleapis.com/v1beta",
  },
  claude: {
    icon: <img src="/claude.svg" alt="Claude" className="w-6 h-6" />,
    category: ["llm"],
    defaultBaseUrl: "https://api.anthropic.com/v1",
  },
  xinference: {
    icon: <img src="/xagent_logo.svg" alt="Xinference" className="w-6 h-6" />,
    category: ["llm", "embedding", "image", "speech"],
    defaultBaseUrl: "http://localhost:9997",
  },
  ollama: {
    icon: <Box className="w-6 h-6" />,
    category: ["llm", "embedding"],
    defaultBaseUrl: "http://localhost:11434"
  }
}

export function ModelsPage() {
  const { token } = useAuth()
  const { t } = useI18n()
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState("llm")
  const [dialogState, setDialogState] = useState<{
    isOpen: boolean;
    viewMode: 'list' | 'connect' | 'form';
    providerId?: string;
    editingModel?: Model;
    key: number;
  }>({
    isOpen: false,
    viewMode: 'form',
    key: 0
  })

  // Provider Discovery State
  const [providers, setProviders] = useState<ProviderConfig[]>([])

  // Default models state
  const [defaultModels, setDefaultModels] = useState<{
    general?: Model
    small_fast?: Model
    visual?: Model
    compact?: Model
    embedding?: Model
  }>({})

  useEffect(() => {
    fetchModels()
    loadDefaultModels()
    fetchProviders()
  }, [])

  const fetchProviders = async () => {
    try {
      const apiProviders = await getSupportedProviders()
      const mergedProviders: ProviderConfig[] = apiProviders.map(p => {
        const localConfig = LOCAL_PROVIDER_CONFIGS[p.id] || {}
        return {
          id: p.id,
          name: p.name,
          description: p.description,
          icon: localConfig.icon || <Brain className="w-6 h-6" />,
          defaultBaseUrl: p.default_base_url || localConfig.defaultBaseUrl,
          categoryBaseUrls: localConfig.categoryBaseUrls,
          category: localConfig.category || ["llm", "embedding", "image", "speech"],
          requires_base_url: p.requires_base_url
        }
      })
      setProviders(mergedProviders)
    } catch (err) {
      console.error("Failed to fetch providers:", err)
    }
  }

  const loadDefaultModels = async () => {
    if (!token) return

    try {
      const response = await apiRequest(`${getApiUrl()}/api/models/user-default`, {
        headers: {}
      })

      if (response.ok) {
        const defaults = await response.json()
        const defaultModelMap: Record<string, Model> = {}

        if (Array.isArray(defaults)) {
          defaults.forEach((defaultConfig: any) => {
            if (defaultConfig && defaultConfig.config_type && defaultConfig.model) {
              defaultModelMap[defaultConfig.config_type] = defaultConfig.model
            }
          })
        }

        setDefaultModels(defaultModelMap)
      } else if (response.status === 404 || response.status === 401 || response.status === 403) {
        setDefaultModels({})
      }
    } catch (err) {
      console.error("Failed to load default models:", err)
    }
  }

  const fetchModels = async () => {
    try {
      setLoading(true)
      const apiUrl = getApiUrl()
      const response = await apiRequest(`${apiUrl}/api/models/`, {
        headers: {}
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || t('models.errors.fetchFailed'))
      }
      const data = await response.json()
      setModels(data)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t('models.errors.fetchFailed'))
    } finally {
      setLoading(false)
    }
  }

  const modelCounts = useMemo(() => {
    return {
      llm: models.filter(m => m.category === 'llm').length,
      embedding: models.filter(m => m.category === 'embedding').length,
      image: models.filter(m => m.category === 'image').length,
      speech: models.filter(m => m.category === 'speech').length
    }
  }, [models])

  const filteredModels = useMemo(() => {
    return models.filter(m => m.category === activeTab)
  }, [models, activeTab])

  const enabledProviders = useMemo(() => {
    const groups: Record<string, Model[]> = {}
    filteredModels.forEach(model => {
      const providerKey = model.model_provider.toLowerCase()
      if (!groups[providerKey]) {
        groups[providerKey] = []
      }
      groups[providerKey].push(model)
    })
    return groups
  }, [filteredModels])

  const handleManageProvider = (providerId: string) => {
    setDialogState(prev => ({
      isOpen: true,
      viewMode: 'list',
      providerId,
      editingModel: undefined,
      key: prev.key + 1
    }))
  }

  const getModelDefaultTypes = (modelId: number) => {
    const types: string[] = []
    Object.entries(defaultModels).forEach(([type, model]) => {
      if (model?.id === modelId) {
        types.push(type)
      }
    })
    return types
  }

  const handleAddModel = () => {
    setDialogState(prev => ({
      isOpen: true,
      viewMode: 'connect',
      providerId: undefined,
      editingModel: undefined,
      key: prev.key + 1
    }))
  }

  const handleConnectProvider = (provider: ProviderConfig) => {
    setDialogState(prev => ({
      isOpen: true,
      viewMode: 'connect',
      providerId: provider.id,
      editingModel: undefined,
      key: prev.key + 1
    }))
  }

  const handleDialogSuccess = async () => {
    await fetchModels()
    await loadDefaultModels()
  }

  // Filter explore providers by active tab
  const exploreProviders = useMemo(() => {
    return providers.filter(p => p.category.includes(activeTab as any))
  }, [activeTab, providers])

  if (loading && models.length === 0) {
    return (
      <div style={{ padding: "2rem", textAlign: "center" }}>
        <div>{t('common.loading')}</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background text-foreground overflow-y-auto">
      <div className="w-full p-8 pb-20">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold mb-1">{t('models.header.title')}</h1>
            <p className="text-muted-foreground">{t('models.header.description')}</p>
          </div>

          <Button onClick={handleAddModel} className="flex items-center gap-2">
            <Plus size={16} className="mr-2" />
            {t('models.header.add')}
          </Button>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full mb-8">
          <TabsList className="bg-transparent h-12 p-0 space-x-6 justify-start border-b w-full rounded-none">
            <TabsTrigger
              value="llm"
              className="data-[state=active]:border-primary data-[state=active]:text-primary data-[state=active]:bg-transparent border border-transparent rounded-md px-4 py-2 shadow-none data-[state=active]:shadow-none text-slate-700 font-normal"
            >
              {t('models.tabs.llm')} <Badge variant="secondary" className="ml-2 rounded-full px-2 py-0.5 bg-slate-100 text-slate-600 hover:bg-slate-100 border-none font-normal">{modelCounts.llm}</Badge>
            </TabsTrigger>
            <TabsTrigger
              value="embedding"
              className="data-[state=active]:border-primary data-[state=active]:text-primary data-[state=active]:bg-transparent border border-transparent rounded-md px-4 py-2 shadow-none data-[state=active]:shadow-none text-slate-700 font-normal"
            >
              {t('models.tabs.embedding')} <Badge variant="secondary" className="ml-2 rounded-full px-2 py-0.5 bg-slate-100 text-slate-600 hover:bg-slate-100 border-none font-normal">{modelCounts.embedding}</Badge>
            </TabsTrigger>
            <TabsTrigger
              value="image"
              className="data-[state=active]:border-primary data-[state=active]:text-primary data-[state=active]:bg-transparent border border-transparent rounded-md px-4 py-2 shadow-none data-[state=active]:shadow-none text-slate-700 font-normal"
            >
              {t('models.tabs.image')} <Badge variant="secondary" className="ml-2 rounded-full px-2 py-0.5 bg-slate-100 text-slate-600 hover:bg-slate-100 border-none font-normal">{modelCounts.image}</Badge>
            </TabsTrigger>
            <TabsTrigger
              value="speech"
              className="data-[state=active]:border-primary data-[state=active]:text-primary data-[state=active]:bg-transparent border border-transparent rounded-md px-4 py-2 shadow-none data-[state=active]:shadow-none text-slate-700 font-normal"
            >
              {t('models.tabs.speech')} <Badge variant="secondary" className="ml-2 rounded-full px-2 py-0.5 bg-slate-100 text-slate-600 hover:bg-slate-100 border-none font-normal">{modelCounts.speech}</Badge>
            </TabsTrigger>
          </TabsList>
        </Tabs>

        {/* Enabled Models */}
        {Object.keys(enabledProviders).length > 0 && (
          <div className="mb-12">
            <div className="flex items-center gap-2 mb-4">
              <CheckCircle2 className="w-5 h-5 text-green-500" />
              <h2 className="text-xl font-semibold">{t('models.section.enabledModels')}</h2>
              <span className="text-sm text-muted-foreground ml-auto">
                {t('models.section.configuredCount', { count: filteredModels.length })}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(enabledProviders).map(([providerId, providerModels]) => {
                const providerConfig = providers.find(p => p.id === providerId) || {
                  id: providerId,
                  name: providerId.charAt(0).toUpperCase() + providerId.slice(1),
                  description: "",
                  icon: <Brain className="w-6 h-6" />,
                  category: ["llm"] as any,
                  defaultBaseUrl: "",
                  models: [],
                }

                // Gather unique abilities across all models in this provider
                const allAbilities = new Set<string>()
                providerModels.forEach(m => {
                  if (m.abilities) {
                    m.abilities.forEach(a => allAbilities.add(a))
                  }
                })
                const capabilities = Array.from(allAbilities)
                const labels: Record<string, string> = {
                  chat: t('models.abilities.chat'),
                  vision: t('models.abilities.vision'),
                  thinking_mode: t('models.abilities.thinking_mode'),
                  tool_calling: t('models.abilities.tool_calling'),
                  embedding: t('models.abilities.embedding'),
                  generate: t('models.abilities.generate'),
                  edit: t('models.abilities.edit'),
                  asr: t('models.abilities.asr'),
                  tts: t('models.abilities.tts')
                }
                const icons: Record<string, any> = {
                  chat: <Brain className="w-3 h-3 mr-1" />,
                  vision: <ImageIcon className="w-3 h-3 mr-1" />,
                  thinking_mode: <Box className="w-3 h-3 mr-1" />,
                  tool_calling: <Zap className="w-3 h-3 mr-1" />,
                  embedding: <Box className="w-3 h-3 mr-1" />,
                  generate: <ImageIcon className="w-3 h-3 mr-1" />,
                  edit: <Box className="w-3 h-3 mr-1" />,
                  asr: <Brain className="w-3 h-3 mr-1" />,
                  tts: <Star className="w-3 h-3 mr-1" />
                }

                // Check defaults
                const hasDefault = providerModels.some(m => getModelDefaultTypes(m.id).length > 0)

                return (
                  <Card key={providerId} className="flex flex-col justify-between p-6 hover:shadow-md transition-shadow">
                    <div className="flex flex-col gap-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="p-2 bg-muted rounded-lg shrink-0">
                            {providerConfig.icon}
                          </div>
                          <h3 className="font-semibold text-lg">{providerConfig.name}</h3>
                        </div>
                      </div>

                      <div className="flex flex-wrap gap-2">
                        {capabilities.map(cap => (
                          <Badge key={cap} variant="secondary" className="font-normal bg-muted/50 hover:bg-muted/50">
                            {icons[cap] || <Box className="w-3 h-3 mr-1" />}
                            {labels[cap] || cap}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div className="flex items-center justify-between mt-6 pt-4 border-t">
                      <div className="flex gap-3 items-center text-sm text-muted-foreground">
                        {hasDefault && (
                          <span className="flex items-center gap-1 text-primary bg-primary/10 px-2 py-0.5 rounded-full text-xs">
                            <Star className="w-3 h-3 fill-current" /> {t('models.card.fields.default')}
                          </span>
                        )}
                        {providerModels.length > 0 && (
                          <span className="flex items-center gap-1 text-xs">
                            <Box className="w-3 h-3" /> {t('models.card.usedBy', { count: providerModels.length })}
                          </span>
                        )}
                      </div>

                      <Button variant="ghost" size="sm" onClick={() => handleManageProvider(providerId)} className="h-8 px-2 text-muted-foreground hover:text-foreground">
                        <Settings className="w-4 h-4 mr-2" />
                        {t('models.card.actions.manage')}
                      </Button>
                    </div>
                  </Card>
                )
              })}
            </div>
          </div>
        )}

        {/* Explore Providers */}
        {exploreProviders.length > 0 && (
          <div>
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">{t('models.section.exploreProviders')}</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {exploreProviders.map(provider => (
                <Card key={provider.id} className="p-6 hover:shadow-md transition-shadow flex flex-col justify-between h-full relative overflow-hidden group">
                  <div>
                    <div className="flex justify-between items-start mb-4">
                      <div className="p-3 bg-muted/30 border rounded-lg w-fit group-hover:bg-background transition-colors">
                        {provider.icon}
                      </div>
                    </div>
                    <h3 className="font-bold text-lg mb-2">{provider.name}</h3>
                    <p className="text-sm text-muted-foreground mb-6 line-clamp-2">
                      {provider.description || t('models.card.connectToUse')}
                    </p>
                  </div>
                  <div className="flex justify-end pt-4 border-t">
                    <Button onClick={() => handleConnectProvider(provider)} className="bg-blue-600 hover:bg-blue-700 text-white rounded-full px-6">
                      {t('models.card.actions.connect')} &rarr;
                    </Button>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        )}
        {/* Dialog */}
        <ModelManagementDialog
          key={dialogState.key}
          isOpen={dialogState.isOpen}
          onOpenChange={(open) => setDialogState(prev => ({ ...prev, isOpen: open }))}
          initialViewMode={dialogState.viewMode}
          initialProviderId={dialogState.providerId}
          initialEditingModel={dialogState.editingModel}
          activeTab={activeTab}
          providers={providers}
          enabledModels={models}
          defaultModels={defaultModels}
          onSuccess={handleDialogSuccess}
        />
      </div>
    </div>
  )
}
