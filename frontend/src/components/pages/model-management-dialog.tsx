"use client"

import { useState, useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Select } from "@/components/ui/select"
import { MultiSelect } from "@/components/ui/multi-select"
import { Switch } from "@/components/ui/switch"
import { getApiUrl } from "@/lib/utils"
import { useAuth } from "@/contexts/auth-context"
import { apiRequest } from "@/lib/api-wrapper"
import {
  DefaultModelType,
  getProviderModels,
  ProviderModel,
  removeUserDefaultModel,
  setUserDefaultModel
} from "@/lib/models"
import {
  ArrowLeft,
  Plus,
  Trash2,
  Edit,
  Brain,
  Image as ImageIcon,
  Box,
  Star,
  Zap,
  CheckCircle2,
  Loader2,
  Search,
  RefreshCw,
  X,
  ChevronRight,
  Check,
  Video
} from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { ScrollArea } from "@/components/ui/scroll-area"
import { toast } from "@/components/ui/sonner"
import { Model, ModelCreate, ProviderConfig, generateModelId, getModelDetailUrl } from "./models"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { Stepper } from "@/components/ui/stepper"

const SPEECH_ABILITY_VALUES = new Set(["asr", "tts"])

function getProviderModelAbilities(model?: ProviderModel): string[] {
  const abilities = model?.abilities?.length
    ? model.abilities
    : model?.model_ability
  return Array.isArray(abilities) ? abilities.map(String) : []
}

function modelSupportsAbilities(
  model: ProviderModel | undefined,
  requiredAbilities: string[] = []
): boolean {
  const required = requiredAbilities.filter((ability) =>
    SPEECH_ABILITY_VALUES.has(ability)
  )
  if (required.length === 0 || !model) return true

  const available = getProviderModelAbilities(model)
  if (available.length === 0) return true

  return required.every((ability) => available.includes(ability))
}

function filterModelsForAbilities(
  models: ProviderModel[],
  category: string,
  abilities: string[] = []
): ProviderModel[] {
  if (category !== "speech") return models
  return models.filter((model) => modelSupportsAbilities(model, abilities))
}

function normalizeAbilitiesForProvider(
  category: string,
  providerId: string,
  abilities: string[]
): string[] {
  if (category === "speech" && providerId === "elevenlabs") {
    const selectedSpeechAbility = [...abilities]
      .reverse()
      .find((ability) => SPEECH_ABILITY_VALUES.has(ability))
    return selectedSpeechAbility ? [selectedSpeechAbility] : []
  }
  return abilities
}

export interface ModelManagementDialogProps {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  initialViewMode: 'list' | 'connect' | 'form'
  initialProviderId?: string
  initialEditingModel?: Model
  activeTab: string
  providers: ProviderConfig[]
  enabledModels: Model[]
  defaultModels: Record<string, Model>
  onSuccess: () => Promise<void>
}

export function ModelManagementDialog({
  isOpen,
  onOpenChange,
  initialViewMode,
  initialProviderId,
  initialEditingModel,
  activeTab,
  providers,
  enabledModels,
  defaultModels,
  onSuccess
}: ModelManagementDialogProps) {
  const { user, token } = useAuth()
  const { t } = useI18n()

  const [viewMode, setViewMode] = useState<'list' | 'connect' | 'form' | 'defaults'>(initialViewMode)
  const [editingModel, setEditingModel] = useState<Model | null>(initialEditingModel || null)
  const [defaultTargetModel, setDefaultTargetModel] = useState<Model | null>(null)
  const managingProviderId = initialProviderId || null
  const [loading, setLoading] = useState(false)

  const [connectStep, setConnectStep] = useState<1 | 2 | 3 | 4>(initialProviderId && initialViewMode === 'connect' ? 2 : 1)
  const [testConnectionStatus, setTestConnectionStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle')
  const [testConnectionError, setTestConnectionError] = useState<string | null>(null)
  const [connectSearchQuery, setConnectSearchQuery] = useState("")

  const [fetchedModels, setFetchedModels] = useState<ProviderModel[]>([])
  const [isFetchingModels, setIsFetchingModels] = useState(false)


  const [modelToDelete, setModelToDelete] = useState<string | null>(null)
  const [isDeletingModel, setIsDeletingModel] = useState(false)
  const [hasInitializedDefaults, setHasInitializedDefaults] = useState(false)
  const [selectedDefaultConfigTypes, setSelectedDefaultConfigTypes] = useState<string[]>([])

  const getDefaultAbilitiesForCategory = (category: string): string[] => {
    if (category === 'llm') return ['chat']
    if (category === 'embedding') return ['embedding']
    if (category === 'image') return ['generate']
    if (category === 'video') return ['generate']
    if (category === 'speech') return ['asr']
    if (category === 'rerank') return ['rerank']
    return []
  }

  const getDefaultAbilitiesForProvider = (category: string, providerId: string): string[] => {
    if (category === 'llm' && providerId === 'deepseek') {
      return ['chat', 'tool_calling', 'thinking_mode']
    }
    if (category === 'speech' && providerId === 'elevenlabs') {
      return ['tts']
    }
    return getDefaultAbilitiesForCategory(category)
  }

  const getPrimaryDefaultType = (category: string, abilities?: string[]): string | null => {
    if (category === 'llm') return 'general'
    if (category === 'embedding') return 'embedding'
    if (category === 'image') return 'image'
    if (category === 'video') return 'video'
    if (category === 'rerank') return 'rerank'
    if (category === 'speech') {
      if (abilities?.includes('asr')) return 'asr'
      if (abilities?.includes('tts')) return 'tts'
      return 'asr'
    }
    return null
  }

  const getDefaultOptionsForModel = (category: string, abilities?: string[]) => {
    return [
      ...(category === 'llm' ? [
        { value: "general", label: t('models.defaults.general') },
        { value: "small_fast", label: t('models.defaults.fast') },
        ...(abilities?.includes('vision') ? [{ value: "visual", label: t('models.defaults.visual') }] : []),
        { value: "compact", label: t('models.defaults.compact') }
      ] : []),
      ...(category === 'embedding' ? [
        { value: "embedding", label: t('models.defaults.embedding') }
      ] : []),
      ...(category === 'image' ? [
        { value: "image", label: t('models.defaults.image') },
        ...(abilities?.includes('edit') ? [{ value: "image_edit", label: t('models.defaults.image_edit') }] : [])
      ] : []),
      ...(category === 'video' ? [
        { value: "video", label: t('models.defaults.video') }
      ] : []),
      ...(category === 'speech' ? [
        ...(abilities?.includes('asr') ? [{ value: "asr", label: t('models.defaults.asr') }] : []),
        ...(abilities?.includes('tts') ? [{ value: "tts", label: t('models.defaults.tts') }] : [])
      ] : []),
      ...(category === 'rerank' ? [
        { value: "rerank", label: t('models.defaults.rerank') }
      ] : [])
    ]
  }

  const resetTestConnectionState = () => {
    setTestConnectionStatus('idle')
    setTestConnectionError(null)
  }

  const resetConnectionState = () => {
    resetTestConnectionState()
    setFetchedModels([])
  }

  const providerAllowsEmptyApiKey = (providerId: string) => providerId === 'xinference'

  const getDefaultBaseUrlForProvider = (providerId: string, category: string) => {
    const provider = providers.find(p => p.id === providerId)
    if (!provider) return ""
    if (provider.categoryBaseUrls && provider.categoryBaseUrls[category]) {
      return provider.categoryBaseUrls[category]
    }
    return provider.defaultBaseUrl || ""
  }

  // Determine initial form data based on editing model or prefill provider
  const getInitialFormData = (): ModelCreate => {
    if (initialEditingModel) {
      const currentDefaults = getModelDefaultTypes(initialEditingModel.id)
      return {
        model_id: initialEditingModel.model_id,
        category: initialEditingModel.category,
        model_provider: initialEditingModel.model_provider,
        model_name: initialEditingModel.model_name,
        api_key: "",
        base_url: initialEditingModel.base_url || "",
        temperature: initialEditingModel.temperature,
        context_window: initialEditingModel.context_window,
        dimension: initialEditingModel.dimension,
        abilities: initialEditingModel.abilities || [],
        default_config_types: currentDefaults,
        share_with_users: initialEditingModel.is_shared
      }
    }

    // Determine default provider based on activeTab if no initialProviderId is given
    let defaultProvider = initialProviderId || "openai";
    if (!initialProviderId) {
      if (activeTab === "image" || activeTab === "embedding") {
        defaultProvider = "dashscope";
      } else if (activeTab === "video") {
        defaultProvider = "volcengine-ark";
      } else if (activeTab === "speech") {
        defaultProvider = "xinference";
      }
    }
    return {
      model_id: "",
      category: activeTab,
      model_provider: defaultProvider,
      model_name: "",
      api_key: "",
      base_url: getDefaultBaseUrlForProvider(defaultProvider, activeTab),
      temperature: activeTab === 'llm' ? undefined : undefined,
      dimension: activeTab === 'embedding' ? undefined : undefined,
      abilities: getDefaultAbilitiesForProvider(activeTab, defaultProvider),
      default_config_types: []
    }
  }

  const [formData, setFormData] = useState<ModelCreate>(getInitialFormData())

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

  const videoAbilityOptions = useMemo(() => [
    { value: "generate", label: t('models.abilities.generate') }
  ], [t])

  const speechAbilityOptions = useMemo(() => [
    { value: "asr", label: t('models.abilities.asr') },
    { value: "tts", label: t('models.abilities.tts') }
  ], [t])

  const rerankAbilityOptions = useMemo(() => [
    { value: "rerank", label: t('models.abilities.rerank') }
  ], [t])

  const getAbilityOptionsForCategory = (category: string) => {
    if (category === 'llm') return abilityOptions
    if (category === 'embedding') return embeddingAbilityOptions
    if (category === 'image') return imageAbilityOptions
    if (category === 'video') return videoAbilityOptions
    if (category === 'speech') return speechAbilityOptions
    if (category === 'rerank') return rerankAbilityOptions
    return []
  }

  const managingProviderModels = useMemo(() => {
    if (!managingProviderId) return []
    return enabledModels.filter(m => m.model_provider === managingProviderId && m.category === activeTab)
  }, [enabledModels, managingProviderId, activeTab])

  const filteredFetchedModels = useMemo(
    () =>
      filterModelsForAbilities(
        fetchedModels,
        formData.category,
        formData.abilities || []
      ),
    [fetchedModels, formData.category, formData.abilities]
  )

  function getModelDefaultTypes(modelId: number) {
    const types: string[] = []
    Object.entries(defaultModels).forEach(([type, model]) => {
      if (model?.id === modelId) {
        types.push(type)
      }
    })
    return types
  }

  const closeDialog = () => {
    setEditingModel(null)
    setDefaultTargetModel(null)
    setSelectedDefaultConfigTypes([])
    onOpenChange(false)
  }

  const handleCancelDefaultsView = () => {
    setDefaultTargetModel(null)
    setSelectedDefaultConfigTypes([])
    setViewMode('list')
  }

  const handleEdit = (model: Model) => {
    setDefaultTargetModel(null)
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
      context_window: model.context_window,
      dimension: model.dimension,
      abilities: model.abilities || [],
      default_config_types: currentDefaults,
      share_with_users: model.is_shared
    })
    setViewMode('form')
  }

  const handleManageDefaults = (model: Model) => {
    setEditingModel(null)
    setDefaultTargetModel(model)
    setSelectedDefaultConfigTypes(getModelDefaultTypes(model.id))
    setViewMode('defaults')
  }

  const handleAddFromList = () => {
    if (!managingProviderId) return
    const providerConfig = providers.find(p => p.id === managingProviderId)
    resetConnectionState()
    setDefaultTargetModel(null)
    setFormData({
      model_id: "",
      category: activeTab,
      model_provider: managingProviderId,
      model_name: "",
      api_key: "",
      base_url: providerConfig?.defaultBaseUrl || "",
      temperature: activeTab === 'llm' ? undefined : undefined,
      dimension: activeTab === 'embedding' ? undefined : undefined,
      abilities: getDefaultAbilitiesForProvider(activeTab, managingProviderId),
      default_config_types: []
    })
    setEditingModel(null)
    setConnectStep(2)
    setViewMode('connect')
  }

  const handleDelete = (modelId: string) => {
    setModelToDelete(modelId)
  }

  const confirmDeleteModel = async () => {
    if (!modelToDelete) return
    setIsDeletingModel(true)

    try {
      const response = await apiRequest(getModelDetailUrl(modelToDelete), {
        method: "DELETE",
        headers: {}
      })
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || t('models.errors.deleteFailed'))
      }

      await onSuccess()
      setModelToDelete(null)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t('models.errors.deleteFailed'))
    } finally {
      setIsDeletingModel(false)
    }
  }

  const handleTestConnection = async (overrideModelName?: string) => {
    const targetModel = overrideModelName || formData.model_name
    if (!targetModel) return

    setTestConnectionStatus('testing')
    setTestConnectionError(null)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/models/test-connection`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model_provider: formData.model_provider,
          model_name: targetModel,
          api_key: formData.api_key,
          base_url: formData.base_url,
          category: formData.category,
          temperature: formData.temperature,
          dimension: formData.dimension,
          abilities: formData.abilities
        })
      })

      const data = await response.json()

      if (!response.ok || data.status === 'failed') {
        throw new Error(data.error || 'Test failed')
      }
      setTestConnectionStatus('success')
    } catch (err) {
      setTestConnectionStatus('error')
      const errorMsg = err instanceof Error ? err.message : 'Unknown error'
      setTestConnectionError(errorMsg)
    }
  }

  const handleFetchModels = async () => {
    try {
      setIsFetchingModels(true)
      const models = await getProviderModels(formData.model_provider, {
        api_key: formData.api_key,
        base_url: formData.base_url,
        category: formData.category,
      })
      // Strip 'models/' prefix from Gemini models returned by the SDK API
      const cleanedModels = models.map(model => {
        if (model.id && model.id.startsWith('models/')) {
          return { ...model, id: model.id.substring(7) }
        }
        return model
      })
      setFetchedModels(cleanedModels)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : t('models.errors.fetchFailed')
      toast.error(errorMessage)
      setFetchedModels([])
      throw err // Rethrow to allow caller to handle failure
    } finally {
      setIsFetchingModels(false)
    }
  }

  const applyFetchedModelSelection = (modelName: string) => {
    const selected = fetchedModels.find(model => model.id === modelName)
    const suggestedBaseUrl = selected?.default_base_url || selected?.base_url
    const suggestedAbilities = getProviderModelAbilities(selected)

    setFormData(prev => ({
      ...prev,
      model_name: modelName,
      base_url: suggestedBaseUrl || prev.base_url,
      abilities: suggestedAbilities?.length
        ? suggestedAbilities
        : prev.abilities?.length
          ? prev.abilities
          : getDefaultAbilitiesForProvider(prev.category, prev.model_provider),
      default_config_types: [],
    }))
    setHasInitializedDefaults(false)
    setTestConnectionStatus('idle')
    setTestConnectionError(null)
  }

  const updateModelAbilities = (abilities: string[]) => {
    resetTestConnectionState()
    setFormData(prev => {
      const nextAbilities = normalizeAbilitiesForProvider(
        prev.category,
        prev.model_provider,
        abilities
      )
      const selectedModel = fetchedModels.find(model => model.id === prev.model_name)
      const nextModelName = modelSupportsAbilities(selectedModel, nextAbilities)
        ? prev.model_name
        : ""

      return {
        ...prev,
        model_name: nextModelName,
        abilities: nextAbilities,
        default_config_types: [],
      }
    })
  }

  const submitModelData = async (data: ModelCreate) => {
    try {
      setLoading(true)

      const payload = { ...data }

      if (!editingModel && !payload.model_id && payload.model_name && payload.model_provider) {
        payload.model_id = generateModelId(payload.model_name, payload.model_provider, user?.id)
      }

      const url = editingModel
        ? getModelDetailUrl(editingModel.model_id)
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
        throw new Error(errorData.detail || t('models.errors.saveFailed'))
      }

      const modelResponse = await response.json()
      const modelId = modelResponse.id

      const currentDefaults = editingModel ? getModelDefaultTypes(editingModel.id) : []
      const newDefaults = payload.default_config_types || []

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

      await onSuccess()

      if (viewMode === 'list' && managingProviderId) {
        // If coming from list view, we just updated or created, keep list view or close?
        // The original code called closeDialog(). We will too.
        closeDialog()
      } else {
        closeDialog()
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t('models.errors.saveFailed'))
    } finally {
      setLoading(false)
    }
  }

  const submitDefaultConfigTypes = async (model: Model, nextDefaultTypes: string[]) => {
    try {
      setLoading(true)

      const currentDefaults = getModelDefaultTypes(model.id)

      const deleteRequests = currentDefaults
        .filter(configType => !nextDefaultTypes.includes(configType))
        .map((configType) => removeUserDefaultModel(token || '', configType as DefaultModelType))

      const addRequests = nextDefaultTypes
        .filter(configType => !currentDefaults.includes(configType))
        .map((configType) => setUserDefaultModel(token || '', configType as DefaultModelType, model.id))

      await Promise.all([...deleteRequests, ...addRequests])

      await onSuccess()
      handleCancelDefaultsView()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t('models.errors.setDefaultFailed'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <Dialog open={isOpen} onOpenChange={onOpenChange}>
        {viewMode === 'list' ? (
          <DialogContent showCloseButton={false} className="max-w-2xl max-h-[80vh] overflow-y-auto">
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
                  {managingProviderModels.map(model => {
                    const defaultTypes = getModelDefaultTypes(model.id)
                    return (
                      <div key={model.model_id} className="flex items-center justify-between p-4">
                        <div>
                          <div className="font-medium flex items-center gap-2 flex-wrap">
                            {model.model_name}
                            {!model.is_owner && (
                              <Badge variant="secondary" className="text-xs px-2 py-0.5 h-auto whitespace-normal text-orange-500">
                                {t('models.defaults.shared_from_others')}
                              </Badge>
                            )}
                            {defaultTypes.map(type => {
                              let labelKey = `models.defaults.${type}`
                              if (type === 'small_fast') labelKey = 'models.defaults.fast'

                              return (
                                <Badge key={type} variant="secondary" className="text-xs px-2 py-0.5 h-auto whitespace-normal text-primary">
                                  {t(labelKey)}
                                </Badge>
                              )
                            })}
                            {model.is_shared && model.is_owner && (
                              <Badge variant="secondary" className="text-xs px-2 py-0.5 h-auto whitespace-normal text-orange-500">
                                {t('models.defaults.shared')}
                              </Badge>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center gap-1">
                          {!model.is_owner && getDefaultOptionsForModel(model.category, model.abilities).length > 0 && (
                            <Button
                              variant={defaultTypes.length > 0 ? "secondary" : "outline"}
                              size="sm"
                              onClick={() => handleManageDefaults(model)}
                              title={t('models.card.actions.setDefault')}
                              className="h-8"
                            >
                              {t('models.card.actions.setDefault')}
                            </Button>
                          )}
                          {model.can_edit && (
                            <Button variant="ghost" size="icon" onClick={() => handleEdit(model)}>
                              <Edit className="w-4 h-4" />
                            </Button>
                          )}
                          {model.can_delete && (
                            <Button
                              variant="ghost"
                              size="icon"
                              className="text-destructive hover:text-destructive"
                              onClick={() => handleDelete(model.model_id)}
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>

            <div className="flex justify-end gap-2 mt-6">
              <Button variant="outline" onClick={closeDialog}>
                {t('models.dialog.cancel')}
              </Button>
            </div>
          </DialogContent>
        ) : viewMode === 'connect' ? (
          <DialogContent className="w-[95vw] sm:max-w-2xl md:max-w-3xl max-h-[85vh] overflow-hidden bg-slate-50 p-0 gap-0 flex flex-col">
            <DialogHeader className="px-4 sm:px-6 py-4 border-b bg-white pr-12">
              <DialogTitle className="text-xl sm:text-2xl font-bold text-left leading-tight">{t('models.dialog.connect.title')}</DialogTitle>
              <DialogDescription className="text-left">{t('models.dialog.connect.description')}</DialogDescription>
            </DialogHeader>

            <Stepper
              steps={[
                {
                  label: t('models.dialog.connect.step1'),
                  content: (
                    <div className="flex flex-col gap-4 h-[400px]">
                      <div className="flex flex-col sm:flex-row gap-3 sm:gap-4">
                        <Select
                          value={formData.category}
                          onValueChange={(value) => {
                            resetConnectionState()
                            setFormData(prev => ({
                              ...prev,
                              category: value,
                              model_provider: "",
                              model_name: "",
                              api_key: "",
                              base_url: "",
                              temperature: value === 'llm' ? prev.temperature : undefined,
                              dimension: value === 'embedding' ? prev.dimension : undefined,
                              abilities: getDefaultAbilitiesForCategory(value),
                              default_config_types: []
                            }))
                          }}
                          options={[
                            { value: "llm", label: t('models.tabs.llm') },
                            { value: "embedding", label: t('models.tabs.embedding') },
                            { value: "image", label: t('models.tabs.image') },
                            { value: "video", label: t('models.tabs.video') },
                            { value: "speech", label: t('models.tabs.speech') },
                            { value: "rerank", label: t('models.tabs.rerank') }
                          ]}
                          className="w-full sm:w-[180px]"
                        />
                        <div className="relative flex-1">
                          <Search className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
                          <Input
                            placeholder={t('models.dialog.connect.searchPlaceholder')}
                            className="pl-9"
                            value={connectSearchQuery}
                            onChange={(e) => setConnectSearchQuery(e.target.value)}
                          />
                        </div>
                      </div>
                      <ScrollArea className="flex-1 border rounded-md overflow-y-scroll">
                        <div className="flex flex-col divide-y">
                          {providers
                            .filter(p => p.category.includes(formData.category))
                            .filter(p => p.name.toLowerCase().includes(connectSearchQuery.toLowerCase()))
                            .map(provider => (
                              <div
                                key={provider.id}
                                className={`flex items-center gap-4 p-4 cursor-pointer hover:bg-muted/50 ${formData.model_provider === provider.id ? 'bg-muted' : ''}`}
                                onClick={() => {
                                  resetConnectionState()
                                  setFormData(prev => ({
                                    ...prev,
                                    model_provider: provider.id,
                                    model_name: "",
                                    base_url: getDefaultBaseUrlForProvider(provider.id, prev.category),
                                    api_key: prev.model_provider === provider.id ? prev.api_key : "",
                                    abilities: getDefaultAbilitiesForProvider(prev.category, provider.id),
                                  }))
                                }}
                              >
                                <div className="p-2 border rounded bg-background">
                                  {provider.icon}
                                </div>
                                <div className="flex-1">
                                  <div className="font-medium flex items-center gap-2">
                                    {provider.name}
                                  </div>
                                  <div className="text-sm text-muted-foreground">{provider.description}</div>
                                </div>
                                {formData.model_provider === provider.id && (
                                  <Check className="w-5 h-5 text-blue-600 ml-auto" />
                                )}
                              </div>
                            ))}
                        </div>
                      </ScrollArea>
                      <div className="flex justify-between mt-2">
                        <Button variant="outline" onClick={closeDialog}>{t('common.cancel')}</Button>
                        <Button
                          onClick={() => setConnectStep(2)}
                          disabled={!formData.model_provider}
                        >
                          {t('models.dialog.connect.next')} &rarr;
                        </Button>
                      </div>
                    </div>
                  )
                },
                {
                  label: t('models.dialog.connect.step2'),
                  content: (
                    <div className="flex flex-col gap-6">
                      {providers.find(p => p.id === formData.model_provider) && (
                        <div className="flex items-center gap-3 p-4 border rounded-md">
                          <div className="p-2 border rounded bg-background">
                            {providers.find(p => p.id === formData.model_provider)?.icon}
                          </div>
                          <span className="font-medium text-lg">{providers.find(p => p.id === formData.model_provider)?.name}</span>
                        </div>
                      )}

                      <div className="space-y-2">
                        <Label className="text-base">{t('models.dialog.connect.apiKeyTitle', { provider: providers.find(p => p.id === formData.model_provider)?.name || '' })}</Label>
                        <Input
                          type="password"
                          placeholder={t('models.dialog.connect.apiKeyPlaceholder')}
                          value={formData.api_key}
                          onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                        />
                        <div className="p-3 bg-muted/50 rounded-md text-sm text-muted-foreground flex items-center gap-2">
                          <div className="w-4 h-4 rounded-full border border-current flex items-center justify-center text-[10px]">i</div>
                          {t('models.dialog.connect.apiKeyHint')}
                        </div>
                      </div>

                      <div className="space-y-2">
                        <details className="group">
                          <summary className="flex items-center gap-1 cursor-pointer text-sm text-muted-foreground font-medium hover:text-black list-none [&::-webkit-details-marker]:hidden">
                            <ChevronRight className="w-4 h-4 transition-transform group-open:rotate-90" />
                            {t('models.dialog.connect.advancedSettings')}
                          </summary>
                          <div className="mt-4 space-y-2 pl-5">
                            <Label className="text-base font-medium">{t('models.form.baseUrl')}</Label>
                            <Input
                              value={formData.base_url}
                              onChange={(e) => setFormData({ ...formData, base_url: e.target.value })}
                            />
                          </div>
                        </details>
                      </div>

                      <div className="flex justify-between mt-4">
                        <Button variant="outline" onClick={() => setConnectStep(1)}>&larr; {t('common.back')}</Button>
                        <Button
                          onClick={async () => {
                            setTestConnectionStatus('testing')
                            try {
                              await handleFetchModels()
                            } catch {
                              // error handled in handleFetchModels
                            } finally {
                              setTestConnectionStatus('idle')
                              setTestConnectionError(null)
                              setConnectStep(3)
                            }
                          }}
                          disabled={
                            (!providerAllowsEmptyApiKey(formData.model_provider) && !formData.api_key)
                            || (providers.find(p => p.id === formData.model_provider)?.requires_base_url ? !formData.base_url : false)
                            || testConnectionStatus === 'testing'
                          }
                        >
                          {testConnectionStatus === 'testing' ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
                          {t('models.dialog.connect.next')} &rarr;
                        </Button>
                      </div>
                    </div>
                  )
                },
                {
                  label: t('models.dialog.connect.step3'),
                  content: (
                    <div className="flex flex-col gap-6">
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <Label className="text-base font-medium">{t('models.form.name')}</Label>
                          <Button variant="ghost" size="sm" className="h-8 text-primary" onClick={handleFetchModels} disabled={isFetchingModels}>
                            <RefreshCw className={`w-3 h-3 mr-1 ${isFetchingModels ? 'animate-spin' : ''}`} /> {t('common.refresh')}
                          </Button>
                        </div>
                        <Select
                          value={formData.model_name}
                          onValueChange={(val) => {
                            applyFetchedModelSelection(val)
                          }}
                          options={filteredFetchedModels.map(m => ({ value: m.id, label: m.id }))}
                          placeholder={filteredFetchedModels.length > 0 ? t('models.form.selectModel') : t('models.form.enterModelName')}
                          allowCustom={formData.model_provider !== 'deepseek'}
                          customPlaceholder={t('models.form.customModel')}
                          customButtonText={t('models.form.addCustom')}
                          onCustomAdd={(val) => {
                            setFetchedModels(prev => prev.find(m => m.id === val)
                              ? prev
                              : [...prev, { id: val, object: "model", created: Date.now(), owned_by: formData.model_provider }]
                            )
                            applyFetchedModelSelection(val)
                            setTestConnectionStatus('idle')
                            setTestConnectionError(null)
                          }}
                        />
                      </div>

                      {formData.model_name && (
                        <div className="space-y-2">
                          <Button
                            variant="outline"
                            className="w-full justify-start"
                            onClick={() => handleTestConnection()}
                            disabled={testConnectionStatus === 'testing'}
                          >
                            {testConnectionStatus === 'testing' ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Zap className="w-4 h-4 mr-2" />}
                            {t('models.dialog.connect.testConnection')}
                          </Button>

                          {testConnectionStatus === 'success' && (
                            <div className="p-3 bg-green-50 text-green-700 border border-green-200 rounded-md flex items-center gap-2 text-sm mt-2">
                              <CheckCircle2 className="w-4 h-4 shrink-0" />
                              <span>{t('models.dialog.connect.testSuccess')}</span>
                            </div>
                          )}

                          {testConnectionStatus === 'error' && testConnectionError && (
                            <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded-md flex items-start gap-2 text-sm mt-2">
                              <X className="w-4 h-4 shrink-0 mt-0.5" />
                              <span className="break-all">{testConnectionError}</span>
                            </div>
                          )}
                        </div>
                      )}

                      <div className="space-y-2">
                        <Label className="text-base font-medium">{t('models.form.abilities')}</Label>
                        <div className="flex gap-2 flex-wrap">
                          {getAbilityOptionsForCategory(formData.category).map(({ value, label }) => {
                            const cap = value
                            const isSelected = formData.abilities?.includes(cap)
                            const icons: Record<string, JSX.Element> = {
                              chat: <Brain className="w-4 h-4 mr-1" />,
                              vision: <ImageIcon className="w-4 h-4 mr-1" />,
                              thinking_mode: <Box className="w-4 h-4 mr-1" />,
                              tool_calling: <Zap className="w-4 h-4 mr-1" />,
                              embedding: <Box className="w-4 h-4 mr-1" />,
                              generate: formData.category === 'video' ? <Video className="w-4 h-4 mr-1" /> : <ImageIcon className="w-4 h-4 mr-1" />,
                              edit: <Edit className="w-4 h-4 mr-1" />,
                              asr: <Brain className="w-4 h-4 mr-1" />,
                              tts: <Star className="w-4 h-4 mr-1" />,
                            }
                            return (
                              <Button
                                key={cap}
                                variant={isSelected ? 'default' : 'outline'}
                                className={`rounded-full ${isSelected ? 'bg-primary text-primary-foreground border-primary hover:bg-primary/90' : ''}`}
                                onClick={() => {
                                  const abilities = formData.abilities || []
                                  const isExclusiveSpeechProvider =
                                    formData.category === "speech" &&
                                    formData.model_provider === "elevenlabs"
                                  if (isExclusiveSpeechProvider) {
                                    updateModelAbilities([cap])
                                  } else if (isSelected) {
                                    updateModelAbilities(abilities.filter(a => a !== cap))
                                  } else {
                                    updateModelAbilities([...abilities, cap])
                                  }
                                }}
                              >
                                {icons[cap]}
                                {label}
                              </Button>
                            )
                          })}
                        </div>
                      </div>

                      {formData.category === 'llm' && (
                        <div className="space-y-2">
                          <Label htmlFor="connect_context_window" className="text-base font-medium">{t('models.form.contextWindow')}</Label>
                          <Input
                            id="connect_context_window"
                            type="number"
                            min="1"
                            value={formData.context_window ?? ""}
                            onChange={(e) => {
                              const val = parseInt(e.target.value, 10)
                              setFormData({ ...formData, context_window: Number.isNaN(val) ? undefined : val })
                            }}
                            placeholder={t('models.form.contextWindowPlaceholder')}
                          />
                          <p className="text-xs text-muted-foreground">{t('models.form.contextWindowHint')}</p>
                        </div>
                      )}

                      <div className="flex justify-between mt-4">
                        <Button variant="outline" onClick={() => setConnectStep(2)}>&larr; {t('common.back')}</Button>
                        <Button
                          onClick={() => {
                            if (!formData.model_name) {
                              toast.error(t('models.dialog.connect.selectModelWarning'))
                              return
                            }

                            if (testConnectionStatus === 'testing') {
                              toast.error(t('models.dialog.connect.testConnection'))
                              return
                            }

                            // Initialize default logic based on whether they have one
                            // only do this once per session so we don't overwrite if they remove it
                            if (!hasInitializedDefaults) {
                              const targetType = getPrimaryDefaultType(formData.category, formData.abilities)
                              if (targetType) {
                                const hasDefault = defaultModels[targetType]
                                if (!hasDefault) {
                                  setFormData(prev => ({
                                    ...prev,
                                    default_config_types: Array.from(new Set([...(prev.default_config_types || []), targetType]))
                                  }))
                                }
                              }
                              setHasInitializedDefaults(true)
                            }

                            setConnectStep(4)
                          }}
                          disabled={loading || !formData.model_name || testConnectionStatus === 'testing'}
                        >
                          {t('models.dialog.connect.next')} &rarr;
                        </Button>
                      </div>
                    </div>
                  )
                },
                {
                  label: t('models.dialog.connect.step4'),
                  content: (
                    <div className="flex flex-col gap-6">
                      <div className="space-y-4">
                        <h3 className="text-lg font-medium">{t('models.dialog.connect.defaultModels')}</h3>
                        <p className="text-sm text-muted-foreground -mt-2">{t('models.dialog.connect.defaultModelsDesc')}</p>

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
                            ] : []),
                            ...(formData.category === 'video' ? [
                              { value: "video", label: t('models.defaults.video') }
                            ] : []),
                            ...(formData.category === 'speech' ? [
                              ...(formData.abilities?.includes('asr') ? [{ value: "asr", label: t('models.defaults.asr') }] : []),
                              ...(formData.abilities?.includes('tts') ? [{ value: "tts", label: t('models.defaults.tts') }] : [])
                            ] : []),
                            ...(formData.category === 'rerank' ? [
                              { value: "rerank", label: t('models.defaults.rerank') }
                            ] : [])
                          ]}
                          placeholder={t('models.form.defaultPlaceholder')}
                        />

                        {(() => {
                          const relevantTypes = (() => {
                            if (formData.category === 'llm') return ['general', 'small_fast', 'visual', 'compact'];
                            if (formData.category === 'embedding') return ['embedding'];
                            if (formData.category === 'image') return ['image', 'image_edit'];
                            if (formData.category === 'video') return ['video'];
                            if (formData.category === 'speech') return ['asr', 'tts'];
                            if (formData.category === 'rerank') return ['rerank'];
                            return [];
                          })();

                          const existingDefaults = relevantTypes
                            .map(type => ({ type, model: defaultModels[type] }))
                            .filter(item => item.model);

                          if (existingDefaults.length === 0) return null;

                          return (
                            <div className="mt-4 p-4 border rounded-md bg-muted/20">
                              <h4 className="text-sm font-medium mb-3 text-foreground">{t('models.dialog.connect.currentDefaults')}</h4>
                              <div className="space-y-2">
                                {existingDefaults.map(({ type, model }) => {
                                  let labelKey = `models.defaults.${type}`;
                                  if (type === 'small_fast') labelKey = 'models.defaults.fast';

                                  return (
                                    <div key={type} className="flex justify-between items-center text-sm">
                                      <span className="text-muted-foreground">{t(labelKey)}</span>
                                      <span className="font-medium truncate max-w-[200px]" title={model.model_name}>
                                        {model.model_name}
                                      </span>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          );
                        })()}

                        {user?.is_admin && (
                          <>
                            <h3 className="text-lg font-medium mt-6">{t('models.dialog.connect.sharing')}</h3>
                            <div className="flex items-start gap-3 p-4 border rounded-md bg-muted/20">
                              <Switch
                                id="share-model"
                                checked={formData.share_with_users ?? false}
                                onCheckedChange={(checked) => setFormData(prev => ({ ...prev, share_with_users: checked }))}
                              />
                              <div className="flex flex-col gap-1">
                                <Label htmlFor="share-model" className="text-base cursor-pointer">{t('models.form.shareWithUsers')}</Label>
                                <p className="text-sm text-muted-foreground">{t('models.dialog.connect.sharingDesc')}</p>
                              </div>
                            </div>
                          </>
                        )}
                      </div>

                      <div className="flex justify-between mt-8">
                        <Button variant="outline" onClick={() => setConnectStep(3)}>&larr; {t('common.back')}</Button>
                        <Button
                          onClick={async () => {
                            if (!formData.model_name) {
                              toast.error(t('models.dialog.connect.selectModelWarning'))
                              return
                            }
                            await submitModelData(formData)
                          }}
                          disabled={loading || !formData.model_name}
                        >
                          {loading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                          {t('models.dialog.connect.activate')} <CheckCircle2 className="w-4 h-4 ml-2" />
                        </Button>
                      </div>
                    </div>
                  )
                }
              ]}
              currentStep={connectStep}
              className="mt-0 px-4 sm:px-6 py-4 flex-1 min-h-0 overflow-hidden"
            />
          </DialogContent>
        ) : viewMode === 'defaults' ? (
          <DialogContent showCloseButton={false} className="max-w-2xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <div className="flex items-center gap-2">
                {managingProviderId && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="-ml-2 h-8 w-8"
                    onClick={handleCancelDefaultsView}
                  >
                    <ArrowLeft className="w-4 h-4" />
                  </Button>
                )}
                <div>
                  <DialogTitle>{t('models.form.setDefault')}</DialogTitle>
                  <DialogDescription>
                    {t('models.dialog.connect.defaultModelsDesc')}
                  </DialogDescription>
                </div>
              </div>
            </DialogHeader>

            {defaultTargetModel && (
              <>
                <div className="flex flex-col gap-4 mt-4">
                  <div className="border rounded-md p-4 bg-muted/20">
                    <div className="font-medium flex items-center gap-2 flex-wrap">
                      <span>{defaultTargetModel.model_name}</span>
                      {!defaultTargetModel.is_owner && (
                        <Badge variant="secondary" className="text-xs px-2 py-0.5 h-auto whitespace-normal text-orange-500">
                          {t('models.defaults.shared_from_others')}
                        </Badge>
                      )}
                      {defaultTargetModel.is_shared && defaultTargetModel.is_owner && (
                        <Badge variant="secondary" className="text-xs px-2 py-0.5 h-auto whitespace-normal text-orange-500">
                          {t('models.defaults.shared')}
                        </Badge>
                      )}
                    </div>
                    <div className="text-sm text-muted-foreground mt-2">
                      {providers.find(p => p.id === defaultTargetModel.model_provider)?.name || defaultTargetModel.model_provider}
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <Star className="w-4 h-4 text-yellow-500" />
                      <Label className="text-sm font-medium">{t('models.form.setDefault')}</Label>
                    </div>
                    <MultiSelect
                      values={selectedDefaultConfigTypes}
                      onValuesChange={setSelectedDefaultConfigTypes}
                      options={getDefaultOptionsForModel(defaultTargetModel.category, defaultTargetModel.abilities)}
                      placeholder={t('models.form.defaultPlaceholder')}
                    />
                  </div>

                  {(() => {
                    const defaultOptions = getDefaultOptionsForModel(defaultTargetModel.category, defaultTargetModel.abilities)
                    const existingDefaults = defaultOptions
                      .map(option => ({ type: option.value, model: defaultModels[option.value] }))
                      .filter(item => item.model)

                    if (existingDefaults.length === 0) return null

                    return (
                      <div className="mt-2 p-4 border rounded-md bg-muted/20">
                        <h4 className="text-sm font-medium mb-3 text-foreground">{t('models.dialog.connect.currentDefaults')}</h4>
                        <div className="space-y-2">
                          {existingDefaults.map(({ type, model }) => {
                            let labelKey = `models.defaults.${type}`
                            if (type === 'small_fast') labelKey = 'models.defaults.fast'

                            return (
                              <div key={type} className="flex justify-between items-center text-sm gap-4">
                                <span className="text-muted-foreground shrink-0">{t(labelKey)}</span>
                                <span className="font-medium truncate" title={model.model_name}>
                                  {model.model_name}
                                </span>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )
                  })()}
                </div>

                <div className="flex justify-end gap-2 mt-6">
                  <Button variant="outline" onClick={handleCancelDefaultsView}>
                    {t('models.dialog.cancel')}
                  </Button>
                  <Button
                    onClick={() => submitDefaultConfigTypes(defaultTargetModel, selectedDefaultConfigTypes)}
                    disabled={loading}
                  >
                    {loading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    {t('common.save')}
                  </Button>
                </div>
              </>
            )}
          </DialogContent>
        ) : (
          <DialogContent showCloseButton={false} className="max-w-2xl max-h-[90vh] overflow-y-auto">
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
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="category">{t('models.form.category')}</Label>
                  <Select
                    value={formData.category}
                    onValueChange={(value) => setFormData({
                      ...formData,
                      category: value,
                      base_url: formData.model_provider ? getDefaultBaseUrlForProvider(formData.model_provider, value) : formData.base_url,
                      abilities: getDefaultAbilitiesForProvider(value, formData.model_provider),
                    })}
                    disabled={!!editingModel}
                    options={[
                      { value: "llm", label: t('models.tabs.llm') },
                      { value: "embedding", label: t('models.tabs.embedding') },
                      { value: "image", label: t('models.tabs.image') },
                      { value: "video", label: t('models.tabs.video') },
                      { value: "speech", label: t('models.tabs.speech') },
                      { value: "rerank", label: t('models.tabs.rerank') }
                    ]}
                  />
                </div>

                <div>
                  <Label htmlFor="model_provider">{t('models.form.provider')}</Label>
                  <Select
                    value={formData.model_provider}
                    onValueChange={(value) => {
                      resetConnectionState()
                      setFormData(prev => ({
                        ...prev,
                        model_provider: value,
                        model_name: "",
                        base_url: getDefaultBaseUrlForProvider(value, prev.category),
                        api_key: prev.model_provider === value ? prev.api_key : "",
                        abilities: getDefaultAbilitiesForProvider(prev.category, value),
                      }))
                    }}
                    disabled={!!editingModel}
                    options={providers
                      .filter(p => p.category.includes(formData.category))
                      .map((provider) => ({
                        value: provider.id,
                        label: provider.name
                      }))
                    }
                  />
                </div>
              </div>

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

              {formData.category === 'llm' && (
                <div>
                  <Label htmlFor="context_window">{t('models.form.contextWindow')}</Label>
                  <Input
                    id="context_window"
                    type="number"
                    min="1"
                    value={formData.context_window ?? ""}
                    onChange={(e) => {
                      const val = parseInt(e.target.value, 10)
                      setFormData({ ...formData, context_window: Number.isNaN(val) ? undefined : val })
                    }}
                    placeholder={t('models.form.contextWindowPlaceholder')}
                  />
                  <p className="text-xs text-muted-foreground mt-1">{t('models.form.contextWindowHint')}</p>
                </div>
              )}

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

              <div>
                <div className="flex justify-between items-center mb-2">
                  <Label htmlFor="model_name">{t('models.form.name')}</Label>
                  {!editingModel && (
                    <div className="flex items-center gap-2">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6"
                        onClick={handleFetchModels}
                        disabled={isFetchingModels || (!formData.api_key && !formData.base_url)}
                        title={t('models.dialog.refreshModels')}
                        type="button"
                      >
                        <RefreshCw className={`h-3 w-3 ${isFetchingModels ? 'animate-spin' : ''}`} />
                      </Button>
                    </div>
                  )}
                </div>

                {editingModel ? (
                  <Input
                    id="model_name"
                    value={formData.model_name}
                    onChange={(e) => setFormData({ ...formData, model_name: e.target.value })}
                    placeholder={t('models.form.enterModelName')}
                  />
                ) : (
                  <Select
                    value={formData.model_name}
                    onValueChange={(value) => applyFetchedModelSelection(value)}
                    options={filteredFetchedModels.map(m => ({ value: m.id, label: m.id }))}
                    placeholder={t('models.form.selectModel')}
                    allowCustom={formData.model_provider !== 'deepseek'}
                    customPlaceholder={t('models.form.enterModelName')}
                    customButtonText={t('models.form.addCustom')}
                    onCustomAdd={(value) => {
                      if (!fetchedModels.find(m => m.id === value)) {
                        setFetchedModels([...fetchedModels, { id: value, object: "model", created: Date.now(), owned_by: formData.model_provider }])
                      }
                      applyFetchedModelSelection(value)
                    }}
                  />
                )}
              </div>

              <div>
                <Label className="mb-2 block">{t('models.form.abilities')}</Label>
                <MultiSelect
                  values={formData.abilities || []}
                  onValuesChange={updateModelAbilities}
                  options={
                    formData.category === 'llm' ? abilityOptions :
                      formData.category === 'embedding' ? embeddingAbilityOptions :
                        formData.category === 'image' ? imageAbilityOptions :
                          formData.category === 'video' ? videoAbilityOptions :
                            formData.category === 'speech' ? speechAbilityOptions :
                              rerankAbilityOptions
                  }
                  placeholder={t('models.form.abilitiesPlaceholder')}
                />
              </div>

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
                    <div className="flex items-center gap-2 mb-2 mt-4">
                      <Star className="w-4 h-4 text-yellow-500" />
                      <Label className="text-sm font-medium">{t('models.form.setDefault')}</Label>
                    </div>
                    <MultiSelect
                      values={formData.default_config_types || []}
                      onValuesChange={(values) => setFormData({ ...formData, default_config_types: values })}
                      options={getDefaultOptionsForModel(formData.category, formData.abilities)}
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
                {t('models.dialog.cancel')}
              </Button>
              <Button onClick={() => submitModelData(formData)}>
                {editingModel ? t('models.form.update') : t('models.form.create')}
              </Button>
            </div>
          </DialogContent >
        )}
      </Dialog>

      <ConfirmDialog
        isOpen={!!modelToDelete}
        onOpenChange={(open) => !open && setModelToDelete(null)}
        onConfirm={confirmDeleteModel}
        isLoading={isDeletingModel}
        description={t('models.deleteConfirm')}
      />
    </>
  )
}
