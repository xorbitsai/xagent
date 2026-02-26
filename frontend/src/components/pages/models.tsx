"use client"

import { useState, useEffect, useMemo, ReactNode } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select } from "@/components/ui/select"
import { MultiSelect } from "@/components/ui/multi-select"
import { Switch } from "@/components/ui/switch"
import { getApiUrl } from "@/lib/utils"
import { useAuth } from "@/contexts/auth-context"
import { apiRequest } from "@/lib/api-wrapper"
import {
  getSupportedProviders,
  getProviderModels,
  Provider as ApiProvider,
  ProviderModel
} from "@/lib/models"
import {
  ArrowLeft,
  Plus,
  Trash2,
  Edit,
  Brain,
  Image as ImageIcon,
  Hash,
  Star,
  Zap,
  Globe,
  Box,
  Settings,
  CheckCircle2,
  Loader2,
  Search
} from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { ScrollArea } from "@/components/ui/scroll-area"

// Interfaces from models-1.tsx
interface Model {
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

interface ModelCreate {
  model_id: string
  category: string
  model_provider: string
  model_name: string
  model_names?: string[]
  api_key: string
  base_url?: string
  temperature?: number
  dimension?: number
  abilities?: string[]
  share_with_users?: boolean
  default_config_types?: string[]
}

// Provider Config from models.tsx
interface ProviderConfig {
  id: string
  name: string
  description: string
  icon: ReactNode
  defaultBaseUrl?: string
  category: string[]
  requires_base_url?: boolean
}

const LOCAL_PROVIDER_CONFIGS: Record<string, Partial<ProviderConfig>> = {
  openai: {
    icon: <img src="/openai.svg" alt="OpenAI" className="w-6 h-6" />,
    category: ["llm", "embedding"],
    defaultBaseUrl: "https://api.openai.com/v1"
  },
  azure_openai: {
    icon: <Zap className="w-6 h-6 text-blue-500" />,
    category: ["llm"]
    // No default base url for Azure, user must provide
  },
  zhipu: {
    icon: <img src="/zhipu.svg" alt="Zhipu" className="w-6 h-6" />,
    category: ["llm"],
    defaultBaseUrl: "https://open.bigmodel.cn/api/paas/v4"
  },
  dashscope: {
    icon: <img src="/dashscope.png" alt="DashScope" className="w-6 h-6" />,
    category: ["embedding", "image"],
    defaultBaseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  },
  gemini: {
    icon: <img src="/gemini.svg" alt="Gemini" className="w-6 h-6" />,
    category: ["llm", "image"],
    defaultBaseUrl: "https://generativelanguage.googleapis.com/v1beta"
  },
  claude: {
    icon: <img src="/claude.svg" alt="Claude" className="w-6 h-6" />,
    category: ["llm"],
    defaultBaseUrl: "https://api.anthropic.com/v1"
  },
  xinference: {
    icon: <img src="/xagent_logo.svg" alt="Xinference" className="w-6 h-6" />,
    category: ["llm", "embedding", "image"],
    defaultBaseUrl: "http://localhost:9997"
  }
}

export function ModelsPage() {
  const { token, user } = useAuth()
  const { t, locale } = useI18n()
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [editingModel, setEditingModel] = useState<Model | null>(null)
  const [activeTab, setActiveTab] = useState("llm")

  // New state for model management list
  const [viewMode, setViewMode] = useState<'list' | 'form' | 'connect'>('form')
  const [managingProviderModels, setManagingProviderModels] = useState<Model[]>([])
  const [managingProviderId, setManagingProviderId] = useState<string | null>(null)

  // Provider Discovery State
  const [providers, setProviders] = useState<ProviderConfig[]>([])
  const [fetchedModels, setFetchedModels] = useState<ProviderModel[]>([])
  const [selectedFetchedModels, setSelectedFetchedModels] = useState<string[]>([])
  const [isFetchingModels, setIsFetchingModels] = useState(false)
  const [fetchModelsError, setFetchModelsError] = useState<string | null>(null)

  // Default models state
  const [defaultModels, setDefaultModels] = useState<{
    general?: Model
    small_fast?: Model
    visual?: Model
    compact?: Model
    embedding?: Model
  }>({})

  const [formData, setFormData] = useState<ModelCreate>({
    model_id: "",
    category: "llm",
    model_provider: "openai",
    model_name: "",
    api_key: "",
    base_url: "",
    temperature: undefined,
    dimension: undefined,
    abilities: [],
    default_config_types: []
  })

  // Options from models-1.tsx
  const abilityOptions = useMemo(() => [
    { value: "chat", label: t('models.abilities.chat') },
    { value: "vision", label: t('models.abilities.vision') },
    { value: "tool_calling", label: t('models.abilities.tool_calling') },
    { value: "thinking_mode", label: t('models.abilities.thinking_mode') }
  ], [t])

  const embeddingAbilityOptions = useMemo(() => [
    { value: "embedding", label: t('models.abilities.embedding') }
  ], [t])

  const imageAbilityOptions = useMemo(() => [
    { value: "generate", label: t('models.abilities.generate') },
    { value: "edit", label: t('models.abilities.edit') }
  ], [t])

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
          defaultBaseUrl: p.default_base_url || localConfig.defaultBaseUrl, // Use API default or local fallback
          category: localConfig.category || ["llm", "embedding", "image"], // Default to all if unknown
          requires_base_url: p.requires_base_url
        }
      })
      setProviders(mergedProviders)
    } catch (err) {
      console.error("Failed to fetch providers:", err)
      // Fallback to local config if API fails?
      // For now just log error, maybe empty list
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
      // Fetch all models
      const response = await apiRequest(`${apiUrl}/api/models/`, {
        headers: {}
      })

      if (!response.ok) throw new Error("Failed to fetch models")
      const data = await response.json()
      setModels(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }

  // Derived state for counts and filtered models
  const modelCounts = useMemo(() => {
    return {
      llm: models.filter(m => m.category === 'llm').length,
      embedding: models.filter(m => m.category === 'embedding').length,
      image: models.filter(m => m.category === 'image').length
    }
  }, [models])

  const filteredModels = useMemo(() => {
    return models.filter(m => m.category === activeTab)
  }, [models, activeTab])

  // Group filtered models by provider
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

  // Handlers from models-1.tsx
  const handleSubmit = async () => {
    try {
      const payloads: ModelCreate[] = []

      if (!editingModel && formData.model_names && formData.model_names.length > 0) {
        // Batch create mode
        formData.model_names.forEach(name => {
           const payload = { ...formData, model_name: name }
           // Remove array field from payload to match backend expectation
           const { model_names, ...rest } = payload as any
           if (!rest.model_id) {
             rest.model_id = `${name}-${rest.model_provider}`
           }
           payloads.push(rest)
        })
      } else {
        // Single create/edit mode
        const payload = { ...formData }
        // Remove array field
        const { model_names, ...rest } = payload as any

        if (!editingModel && !rest.model_id && rest.model_name && rest.model_provider) {
           rest.model_id = `${rest.model_name}-${rest.model_provider}`
        }
        payloads.push(rest)
      }

      for (const payload of payloads) {
        const url = editingModel
          ? `${getApiUrl()}/api/models/${editingModel.model_id}`
          : `${getApiUrl()}/api/models/`

        const response = await apiRequest(url, {
          method: editingModel ? "PUT" : "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(payload),
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.detail || `Failed to save model ${payload.model_name}`)
        }

        // Handle default configurations
        const modelResponse = await response.json()
        const modelId = modelResponse.id

        const currentDefaults = editingModel ? getModelDefaultTypes(editingModel.id) : []
        const newDefaults = formData.default_config_types || []

        for (const configType of currentDefaults) {
          if (!newDefaults.includes(configType)) {
            await apiRequest(`${getApiUrl()}/api/models/user-default/${configType}`, {
              method: 'DELETE',
              headers: {}
            })
          }
        }

        for (const configType of newDefaults) {
          if (!currentDefaults.includes(configType)) {
            await apiRequest(`${getApiUrl()}/api/models/user-default`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                config_type: configType,
                model_id: modelId
              })
            })
          }
        }
      }

      await fetchModels()
      await loadDefaultModels()

      // If we are in list view, refresh the list
      if (viewMode === 'list' && managingProviderId) {
        closeDialog()
      } else {
        closeDialog()
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }

  const handleManageProvider = (models: Model[], providerId: string) => {
    setManagingProviderModels(models)
    setManagingProviderId(providerId)
    setViewMode('list')
    setIsDialogOpen(true)
  }

  const handleEdit = (model: Model) => {
    setEditingModel(model)
    const currentDefaults = getModelDefaultTypes(model.id)
    setFormData({
      model_id: model.model_id,
      category: model.category,
      model_provider: model.model_provider,
      model_name: model.model_name,
      api_key: "",
      base_url: model.base_url || "",
      temperature: model.temperature,
      dimension: model.dimension,
      abilities: model.abilities || [],
      default_config_types: currentDefaults,
      share_with_users: model.is_shared
    })
    loadDefaultModels()
    setViewMode('form')
    // If we are not coming from the list view (i.e. direct edit, though we changed UI to always list),
    // we should set managing provider if possible, but handleManageProvider sets list view.
    // Here we just open form.
    setIsDialogOpen(true)
  }

  const handleDelete = async (modelId: string) => {
    if (!confirm(t('models.deleteConfirm'))) return

    try {
      const response = await apiRequest(`${getApiUrl()}/api/models/${modelId}`, {
        method: "DELETE",
        headers: {}
      })
      if (!response.ok) throw new Error("Failed to delete model")

      await fetchModels()

      // If in list view, update the local list
      if (viewMode === 'list') {
        setManagingProviderModels(prev => prev.filter(m => m.model_id !== modelId))
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
    }
  }

  const closeDialog = () => {
    setIsDialogOpen(false)
    setEditingModel(null)
    setViewMode('form')
    setManagingProviderModels([])
    setManagingProviderId(null)
    setFormData({
      model_id: "",
      category: "llm",
      model_provider: "openai",
      model_name: "",
      api_key: "",
      base_url: "",
      temperature: undefined,
      dimension: undefined,
      abilities: [],
      default_config_types: []
    })
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
    setFormData({
      model_id: "",
      category: activeTab,
      model_provider: "openai",
      model_name: "",
      api_key: "",
      base_url: "",
      temperature: activeTab === 'llm' ? undefined : undefined,
      dimension: activeTab === 'embedding' ? undefined : undefined,
      abilities: activeTab === 'llm' ? ["chat"] : activeTab === 'embedding' ? ["embedding"] : ["generate"],
      default_config_types: []
    })
    setViewMode('form')
    setIsDialogOpen(true)
  }

  // Auto fetch models when api_key/base_url changes
  useEffect(() => {
    if (viewMode !== 'connect' && !(viewMode === 'form' && !editingModel)) return

    const timer = setTimeout(() => {
      if (formData.api_key && formData.base_url) {
        handleFetchModels()
      } else {
        // Reset if missing required fields
        setFetchedModels([])
        setSelectedFetchedModels([])
        setFetchModelsError(null)
      }
    }, 500) // 500ms debounce

    return () => clearTimeout(timer)
  }, [formData.api_key, formData.base_url, viewMode, editingModel])

  const handleFetchModels = async () => {
    // If no API key and no default base url, don't fetch (unless base url is filled)
    // Actually, user said "When API Key and Base URL are filled".
    // Most providers need API Key. Some like Ollama might only need Base URL (no key).
    // But for now let's assume if API Key is present OR (Base URL present and provider doesn't strictly need key?)
    // Let's stick to: if api_key is present OR (base_url is present AND api_key might be optional/empty allowed?)
    // But usually API Key is the main trigger.

    // For safety, let's just try if at least one is present?
    // Or better, checking formData.api_key as in the debounce.

    // If we are already fetching, maybe skip? But debounce handles that mostly.

    try {
      setIsFetchingModels(true)
      setFetchModelsError(null)
      const models = await getProviderModels(formData.model_provider, {
        api_key: formData.api_key,
        base_url: formData.base_url
      })
      setFetchedModels(models)
    } catch (err) {
      setFetchModelsError(err instanceof Error ? err.message : "Failed to fetch models")
      setFetchedModels([])
    } finally {
      setIsFetchingModels(false)
    }
  }

  const handleSaveSelectedModels = async () => {
    try {
      setLoading(true)
      const selected = fetchedModels.filter(m => selectedFetchedModels.includes(m.id))

      for (const model of selected) {
         const payload: ModelCreate = {
            ...formData,
            model_id: `${model.id}-${formData.model_provider}`,
            model_name: model.id,
            // Ensure capabilities are set?
            // For now use defaults from formData which are set in handleConnectProvider
         }

         const url = `${getApiUrl()}/api/models/`
         await apiRequest(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
         })
      }

      await fetchModels()
      closeDialog()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save models")
    } finally {
      setLoading(false)
    }
  }

  const handleConnectProvider = (provider: ProviderConfig) => {
    setFormData({
      model_id: "",
      category: activeTab,
      model_provider: provider.id,
      model_name: "",
      api_key: "",
      base_url: provider.defaultBaseUrl || "",
      temperature: activeTab === 'llm' ? undefined : undefined,
      dimension: activeTab === 'embedding' ? undefined : undefined,
      abilities: activeTab === 'llm' ? ["chat"] : activeTab === 'embedding' ? ["embedding"] : ["generate"],
      default_config_types: []
    })

    // Reset fetch state
    setFetchedModels([])
    setSelectedFetchedModels([])
    setFetchModelsError(null)
    setIsFetchingModels(false)

    // Open in connect mode
    setViewMode('connect')
    setIsDialogOpen(true)
  }

  const handleAddFromList = () => {
    if (!managingProviderId) return
    const providerConfig = providers.find(p => p.id === managingProviderId)

    setFormData({
      model_id: "",
      category: activeTab,
      model_provider: managingProviderId,
      model_name: "",
      api_key: "",
      base_url: providerConfig?.defaultBaseUrl || "",
      temperature: activeTab === 'llm' ? undefined : undefined,
      dimension: activeTab === 'embedding' ? undefined : undefined,
      abilities: activeTab === 'llm' ? ["chat"] : activeTab === 'embedding' ? ["embedding"] : ["generate"],
      default_config_types: []
    })
    setViewMode('form')
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

        {/* Error */}
        {error && (
          <div className="bg-destructive text-destructive-foreground p-4 rounded-lg mb-6">
            {error}
          </div>
        )}

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="bg-transparent border-b rounded-none w-full justify-start h-auto p-0 mb-8 space-x-6">
            <TabsTrigger
              value="llm"
              className="flex-none rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent px-0 py-2"
            >
              <div className="flex items-center gap-2">
                <span className="font-medium">{t('models.tabs.llm')}</span>
                <Badge variant="secondary" className="rounded-full px-2 py-0.5 text-xs">
                  {modelCounts.llm}
                </Badge>
              </div>
            </TabsTrigger>
            <TabsTrigger
              value="embedding"
              className="flex-none rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent px-0 py-2"
            >
              <div className="flex items-center gap-2">
                <span className="font-medium">{t('models.tabs.embedding')}</span>
                <Badge variant="secondary" className="rounded-full px-2 py-0.5 text-xs">
                  {modelCounts.embedding}
                </Badge>
              </div>
            </TabsTrigger>
            <TabsTrigger
              value="image"
              className="flex-none rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent px-0 py-2"
            >
              <div className="flex items-center gap-2">
                <span className="font-medium">{t('models.tabs.image')}</span>
                <Badge variant="secondary" className="rounded-full px-2 py-0.5 text-xs">
                  {modelCounts.image}
                </Badge>
              </div>
            </TabsTrigger>
          </TabsList>

          <TabsContent value={activeTab} className="space-y-10 flex flex-col">
            {/* Enabled Models */}
            {Object.keys(enabledProviders).length > 0 && (
              <div>
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
                      category: [activeTab] as any,
                      defaultBaseUrl: "",
                      models: []
                    }

                    return (
                      <Card key={providerId} className="flex flex-col justify-between p-6 hover:shadow-md transition-shadow">
                        <div className="flex items-start gap-3">
                          <div className="p-2 bg-muted rounded-lg shrink-0">
                            {providerConfig.icon}
                          </div>
                          <div>
                            <h3 className="font-semibold">{providerConfig.name}</h3>
                            <div className="flex gap-1 flex-wrap">
                              {providerModels.map(m => {
                                const defaultTypes = getModelDefaultTypes(m.id)
                                return (
                                  <div
                                    key={m.id}
                                    className="flex items-center gap-2 cursor-pointer hover:bg-muted/50 p-1 rounded transition-colors"
                                    onClick={() => handleEdit(m)}
                                  >
                                    <Badge variant="secondary" className="text-xs px-2 py-0.5 h-auto whitespace-normal text-left">
                                      <span className="mr-1">{m.model_name}</span>
                                      {defaultTypes.map(type => {
                                        let labelKey = `models.defaults.${type}`
                                        if (type === 'small_fast') labelKey = 'models.defaults.fast'

                                        return (
                                          <span key={type} className="text-[10px] text-primary inline-block whitespace-nowrap">
                                            {t(labelKey)}
                                          </span>
                                        )
                                      })}
                                      {m.is_shared && m.is_owner && (
                                        <span className="text-[10px] text-orange-500 inline-block whitespace-nowrap ml-1">
                                          {t('models.defaults.shared')}
                                        </span>
                                      )}
                                    </Badge>
                                  </div>
                                )
                              })}
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center justify-between mt-4">
                          <div className="flex items-center gap-2 text-sm text-green-600">
                            <div className="w-2 h-2 rounded-full bg-green-500" />
                            {t('models.card.status.connected')}
                          </div>

                          <Button variant="ghost" size="sm" onClick={() => handleManageProvider(providerModels, providerId)}>
                            <Settings className="w-4 h-4 mr-2" />
                            {t('models.card.actions.edit')}
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
                  <h2 className="text-xl font-semibold">{t('models.section.exploreProviders')}</h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {exploreProviders.map(provider => (
                    <Card key={provider.id} className="p-6 hover:shadow-md transition-shadow flex flex-col justify-between h-full">
                    <div>
                      <div className="mb-4 p-2 bg-muted/50 rounded-lg w-fit">
                        {provider.icon}
                      </div>
                      <h3 className="font-semibold text-lg mb-2">{provider.name}</h3>
                      <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                        {t(provider.description)}
                      </p>
                    </div>
                    <div className="flex justify-end items-center mt-auto pt-4 border-t">
                      <Button onClick={() => handleConnectProvider(provider)}>
                        {t('models.card.actions.connect')}
                      </Button>
                    </div>
                  </Card>
                ))}
              </div>
            </div>)}
          </TabsContent>
        </Tabs>

      {/* Dialog */}
      <Dialog open={isDialogOpen} onOpenChange={(open) => {
        if (!open) closeDialog()
        else setIsDialogOpen(true)
      }}>
        {viewMode === 'list' ? (
          <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <div className="flex justify-between items-center pr-8">
                <DialogTitle>
                    {t('models.dialog.providerModelsTitle', { provider: providers.find(p => p.id === managingProviderId)?.name || managingProviderId || "" })}
                  </DialogTitle>
                <Button onClick={handleAddFromList} size="sm">
                  <Plus className="w-4 h-4 mr-2" />
                  {t('models.header.add')}
                </Button>
              </div>
              <DialogDescription>
                {t('models.dialog.manageDescription')}
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4 mt-4">
              {managingProviderModels.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  {t('models.dialog.noModelsConfigured')}
                </div>
              ) : (
                <div className="border rounded-md divide-y">
                  {managingProviderModels.map(model => (
                    <div key={model.model_id} className="flex items-center justify-between p-4">
                      <div>
                        <div className="font-medium flex items-center gap-2">
                          {model.model_name}
                          <Badge variant="outline" className="text-xs font-normal">
                            {model.model_id}
                          </Badge>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button variant="ghost" size="icon" onClick={() => handleEdit(model)}>
                          <Edit className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="text-destructive hover:text-destructive"
                          onClick={() => handleDelete(model.model_id)}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="flex justify-end gap-2 mt-6">
              <Button variant="outline" onClick={closeDialog}>
                {t('models.defaultDialog.close')}
              </Button>
            </div>
          </DialogContent>
        ) : viewMode === 'connect' ? (
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
               <div className="flex justify-between items-center pr-8">
                 <DialogTitle>
                    {t('models.dialog.connectProviderTitle', { provider: providers.find(p => p.id === formData.model_provider)?.name || formData.model_provider })}
                 </DialogTitle>
               </div>
               <DialogDescription>
                 {t('models.dialog.connectProviderDescription')}
               </DialogDescription>
            </DialogHeader>

            <div className="flex flex-col gap-4 mt-4">
               <div className="grid grid-cols-1 gap-4">
                 <div>
                   <Label htmlFor="api_key">{t('models.form.apiKey')}</Label>
                   <Input
                     id="api_key"
                     type="password"
                     value={formData.api_key}
                     onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                     placeholder={t('models.form.apiKeyPlaceholder')}
                   />
                 </div>
                 <div>
                   <Label htmlFor="base_url">{t('models.form.baseUrl')}</Label>
                   <Input
                     id="base_url"
                     value={formData.base_url}
                     onChange={(e) => setFormData({ ...formData, base_url: e.target.value })}
                     placeholder={t('models.form.baseUrlPlaceholder')}
                   />
                 </div>
               </div>

               {fetchModelsError && (
                 <Alert variant="destructive">
                   <AlertDescription>{fetchModelsError}</AlertDescription>
                 </Alert>
               )}

               <div className="mt-4">
                 <Label className="mb-2 block">{t('models.dialog.availableModels')}</Label>
                 {isFetchingModels ? (
                   <div className="flex items-center justify-center p-8 border rounded-md text-muted-foreground">
                     <Loader2 className="w-6 h-6 animate-spin mr-2" />
                     {t('models.dialog.fetchingModels')}
                   </div>
                 ) : fetchedModels.length > 0 ? (
                   <>
                     <ScrollArea className="h-max-[300px] border p-4">
                       <div className="space-y-2">
                         {fetchedModels.map(model => (
                           <div key={model.id} className="flex items-center space-x-2">
                             <input
                               type="checkbox"
                               id={`model-${model.id}`}
                               className="h-4 w-4 rounded border-gray-300"
                               checked={selectedFetchedModels.includes(model.id)}
                               onChange={(e) => {
                                 if (e.target.checked) {
                                   setSelectedFetchedModels([...selectedFetchedModels, model.id])
                                 } else {
                                   setSelectedFetchedModels(selectedFetchedModels.filter(id => id !== model.id))
                                 }
                               }}
                             />
                             <Label htmlFor={`model-${model.id}`} className="font-normal cursor-pointer flex-1">
                               {model.id}
                             </Label>
                           </div>
                         ))}
                       </div>
                     </ScrollArea>
                     <div className="text-sm text-muted-foreground mt-2">
                       {selectedFetchedModels.length} models selected
                     </div>
                   </>
                 ) : (
                   <div className="flex items-center justify-center p-8 border rounded-md text-muted-foreground italic bg-muted/30">
                     {t('models.dialog.modelsFetchedAfterConnect')}
                   </div>
                 )}
               </div>
            </div>

            <div className="flex justify-end gap-2 mt-6">
              <Button variant="outline" onClick={closeDialog}>
                {t('models.dialog.cancel')}
              </Button>
              <Button onClick={handleSaveSelectedModels} disabled={selectedFetchedModels.length === 0 || loading}>
                {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {t('models.dialog.add')}
              </Button>
            </div>
          </DialogContent>
        ) : (
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <div className="flex items-center gap-2">
                {managingProviderId && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="-ml-2 h-8 w-8"
                    onClick={() => setViewMode('list')}
                  >
                    <ArrowLeft className="w-4 h-4" />
                  </Button>
                )}
                <div>
                  <DialogTitle>{editingModel ? t('models.dialog.editTitle') : t('models.dialog.createTitle')}</DialogTitle>
                  <DialogDescription>
                    {editingModel ? t('models.dialog.editDescription') : t('models.dialog.createDescription')}
                  </DialogDescription>
                </div>
              </div>
            </DialogHeader>

            <div className="flex flex-col gap-4 mt-4">
              {error && (
                <Alert>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {/* Category and Provider - Always visible */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="category">{t('models.form.category')}</Label>
                  <Select
                    value={formData.category}
                    onValueChange={(value) => setFormData({ ...formData, category: value })}
                    disabled={!!editingModel}
                    options={[
                      { value: "llm", label: t('models.tabs.llm') },
                      { value: "embedding", label: t('models.tabs.embedding') },
                      { value: "image", label: t('models.tabs.image') }
                    ]}
                  />
                </div>

                <div>
                  <Label htmlFor="model_provider">{t('models.form.provider')}</Label>
                  <Select
                    value={formData.model_provider}
                    onValueChange={(value) => setFormData({ ...formData, model_provider: value })}
                    disabled={!!editingModel}
                    options={providers
                      .filter(p => p.category.includes(formData.category as any))
                      .map((provider) => ({
                        value: provider.id,
                        label: provider.name
                      }))
                    }
                  />
                </div>
              </div>

              {/* API Key and Base URL - Moved up for Create Mode */}
              <div>
                <Label htmlFor="api_key">{t('models.form.apiKey')}</Label>
                <Input
                  id="api_key"
                  type="password"
                  value={formData.api_key}
                  onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                  placeholder={t('models.form.apiKeyPlaceholder')}
                />
              </div>

              <div>
                <Label htmlFor="base_url">{t('models.form.baseUrl')}</Label>
                <Input
                  id="base_url"
                  value={formData.base_url}
                  onChange={(e) => setFormData({ ...formData, base_url: e.target.value })}
                  placeholder={t('models.form.baseUrlPlaceholder')}
                />
              </div>

              {/* Model ID - Only show if editing */}
              {editingModel && (
                <div>
                  <Label htmlFor="model_id">{t('models.form.modelId')}</Label>
                  <Input
                    id="model_id"
                    value={formData.model_id}
                    onChange={(e) => setFormData({ ...formData, model_id: e.target.value })}
                    disabled={!!editingModel}
                  />
                </div>
              )}

              {/* Model Name Selection */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <Label htmlFor="model_name">{t('models.form.name')}</Label>
                  {isFetchingModels && <span className="text-xs text-muted-foreground animate-pulse">Fetching models...</span>}
                  {!isFetchingModels && fetchModelsError && !editingModel && (
                     <span className="text-xs text-destructive">{t('models.error.fetchFailed') || "Fetch failed"}</span>
                  )}
                </div>

                {editingModel ? (
                  <Input
                    id="model_name"
                    value={formData.model_name}
                    onChange={(e) => setFormData({ ...formData, model_name: e.target.value })}
                    placeholder="Enter model name"
                  />
                ) : (
                  <MultiSelect
                    creatable
                    values={formData.model_names || (formData.model_name ? [formData.model_name] : [])}
                    onValuesChange={(values) => setFormData({ ...formData, model_names: values })}
                    options={fetchedModels.map(m => ({ value: m.id, label: m.id }))}
                    placeholder={fetchedModels.length > 0 ? (t('models.form.selectModel') || "Select models...") : "Enter model names..."}
                  />
                )}
              </div>

              {/* Abilities - Always visible as requested */}
              <div>
                <Label className="mb-2 block">{t('models.form.abilities')}</Label>
                <MultiSelect
                  values={formData.abilities || []}
                  onValuesChange={(values) => setFormData({ ...formData, abilities: values })}
                  options={
                    formData.category === 'llm' ? abilityOptions :
                    formData.category === 'embedding' ? embeddingAbilityOptions :
                    imageAbilityOptions
                  }
                  placeholder={t('models.form.abilitiesPlaceholder')}
                />
              </div>

              {/* Advanced Fields - Show only when editing */}
              {editingModel && (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {formData.category === 'llm' && (
                      <div>
                        <Label htmlFor="temperature">{t('models.form.temperature')}</Label>
                        <Input
                          id="temperature"
                          type="number"
                          step="0.1"
                          min="0"
                          max="2"
                          value={formData.temperature ?? ""}
                          onChange={(e) => setFormData({ ...formData, temperature: e.target.value ? parseFloat(e.target.value) : undefined })}
                        />
                      </div>
                    )}

                    {formData.category === 'embedding' && (
                      <div>
                        <Label htmlFor="dimension">{t('models.form.dimension')}</Label>
                        <Input
                          id="dimension"
                          type="number"
                          value={formData.dimension ?? ""}
                          onChange={(e) => setFormData({ ...formData, dimension: e.target.value ? parseInt(e.target.value) : undefined })}
                        />
                      </div>
                    )}
                  </div>

                  <div>
                    <div className="flex items-center gap-2">
                      <Star className="w-4 h-4 text-yellow-500" />
                      <Label className="text-sm font-medium">{t('models.form.setDefault')}</Label>
                    </div>
                    <MultiSelect
                      values={formData.default_config_types || []}
                      onValuesChange={(values) => setFormData({ ...formData, default_config_types: values })}
                      options={[
                        ...(formData.category === 'llm' ? [
                          { value: "general", label: t('models.defaults.general') },
                          { value: "small_fast", label: t('models.defaults.fast') },
                          ...(formData.abilities?.includes('vision') ? [{ value: "visual", label: t('models.defaults.visual') }] : []),
                          { value: "compact", label: t('models.defaults.compact') }
                        ] : []),
                        ...(formData.category === 'embedding' ? [
                          { value: "embedding", label: t('models.defaults.embedding') }
                        ] : []),
                        ...(formData.category === 'image' ? [
                          { value: "image", label: t('models.defaults.image') },
                          ...(formData.abilities?.includes('edit') ? [{ value: "image_edit", label: t('models.defaults.image_edit') }] : [])
                        ] : [])
                      ]}
                      placeholder={t('models.form.defaultPlaceholder')}
                    />
                  </div>

                  {user?.is_admin && (
                  <div className="flex items-center justify-between pt-2">
                    <Label htmlFor="share-with-users">{t('models.form.shareWithUsers')}</Label>
                    <Switch
                      id="share-with-users"
                      checked={formData.share_with_users ?? false}
                      onCheckedChange={(checked) => setFormData(prev => ({ ...prev, share_with_users: checked }))}
                    />
                  </div>
                  )}
                </>
              )}

            </div>

            <div className="flex justify-end gap-2 mt-6">
              <Button variant="outline" onClick={closeDialog}>
                {t('models.defaultDialog.close')}
              </Button>
              <Button onClick={handleSubmit}>
                {editingModel ? t('models.form.update') : t('models.form.create')}
              </Button>
            </div>
          </DialogContent>
        )}
      </Dialog>
      </div>
    </div>
  )
}
