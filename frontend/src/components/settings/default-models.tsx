"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/select"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import {
  Brain,
  Zap,
  Eye,
  FileText,
  Search,
  Image as ImageIcon,
  Video,
  Mic,
  Volume2,
  Loader2,
  CheckCircle,
  AlertCircle,
} from "lucide-react"
import {
  getUserModels,
  getUserDefaultModels,
  setUserDefaultModel,
  removeUserDefaultModel,
  DefaultModelConfig,
  DefaultModelType,
  Model,
} from "@/lib/models"
import { useAuth } from "@/contexts/auth-context"
import { useI18n } from "@/contexts/i18n-context"

const modelTypeConfig = {
  general: {
    icon: Brain,
    color: "bg-blue-500",
  },
  small_fast: {
    icon: Zap,
    color: "bg-green-500",
  },
  visual: {
    icon: Eye,
    color: "bg-purple-500",
  },
  compact: {
    icon: FileText,
    color: "bg-orange-500",
  },
  embedding: {
    icon: Search,
    color: "bg-red-500",
  },
  image: {
    icon: ImageIcon,
    color: "bg-pink-500",
  },
  image_edit: {
    icon: ImageIcon,
    color: "bg-fuchsia-500",
  },
  video: {
    icon: Video,
    color: "bg-rose-500",
  },
  asr: {
    icon: Mic,
    color: "bg-cyan-500",
  },
  tts: {
    icon: Volume2,
    color: "bg-teal-500",
  },
  speech: {
    icon: Volume2,
    color: "bg-sky-500",
  },
}

const defaultModelTypes = Object.keys(modelTypeConfig) as DefaultModelType[]

const getModelCategory = (model: Model): string => {
  return model.category || ''
}

const getModelAbilities = (model: Model): string[] => {
  return model.abilities || []
}

const getModelDisplayName = (model: Model): string => {
  if (model.model_name) return model.model_name
  return model.name
}

const getModelProviderLabel = (model: Model): string => {
  if (model.model_provider) return model.model_provider
  return model.provider
}

const getCompatibleModels = (models: Model[], configType: DefaultModelType): Model[] => {
  if (configType === 'embedding') {
    return models.filter((model) => getModelCategory(model) === 'embedding')
  }
  if (configType === 'image') {
    return models.filter((model) => getModelCategory(model) === 'image')
  }
  if (configType === 'image_edit') {
    return models.filter((model) => getModelCategory(model) === 'image' && getModelAbilities(model).includes('edit'))
  }
  if (configType === 'video') {
    return models.filter((model) => getModelCategory(model) === 'video' && getModelAbilities(model).includes('generate'))
  }
  if (configType === 'asr') {
    return models.filter((model) => getModelCategory(model) === 'speech' && getModelAbilities(model).includes('asr'))
  }
  if (configType === 'tts') {
    return models.filter((model) => getModelCategory(model) === 'speech' && getModelAbilities(model).includes('tts'))
  }
  if (configType === 'speech') {
    return models.filter((model) => {
      const abilities = getModelAbilities(model)
      return getModelCategory(model) === 'speech' && abilities.includes('asr') && abilities.includes('tts')
    })
  }
  return models.filter((model) => getModelCategory(model) === 'llm')
}

export function DefaultModelsSettings() {
  const { token } = useAuth()
  const [models, setModels] = useState<Model[]>([])
  const [defaultModels, setDefaultModels] = useState<DefaultModelConfig>({})
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState<string | null>(null)
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)
  const { t } = useI18n()

  useEffect(() => {
    loadModelsAndDefaults()
  }, [token])

  const loadModelsAndDefaults = async () => {
    if (!token) return

    try {
      setLoading(true)
      const [modelsData, defaultsData] = await Promise.all([
        getUserModels(token),
        getUserDefaultModels(token),
      ])
      setModels(modelsData)
      setDefaultModels(defaultsData)
    } catch (error) {
      setMessage({ type: 'error', text: t('settings.defaultModels.messages.loadFailed') })
    } finally {
      setLoading(false)
    }
  }

  const handleSetDefault = async (configType: DefaultModelType, modelId: number) => {
    if (!token) return

    try {
      setSaving(configType)
      setMessage(null)

      await setUserDefaultModel(token, configType, modelId)

      // Reload defaults
      const defaultsData = await getUserDefaultModels(token)
      setDefaultModels(defaultsData)

      const typeTitle = t(`settings.defaultModels.types.${configType}.title`)
      setMessage({ type: 'success', text: t('settings.defaultModels.messages.updated', { type: typeTitle }) })
    } catch (error) {
      setMessage({ type: 'error', text: t('settings.defaultModels.messages.setFailed') })
    } finally {
      setSaving(null)
    }
  }

  const handleRemoveDefault = async (configType: DefaultModelType) => {
    if (!token) return

    try {
      setSaving(configType)
      setMessage(null)

      await removeUserDefaultModel(token, configType)

      // Reload defaults
      const defaultsData = await getUserDefaultModels(token)
      setDefaultModels(defaultsData)

      const typeTitle = t(`settings.defaultModels.types.${configType}.title`)
      setMessage({ type: 'success', text: t('settings.defaultModels.messages.removed', { type: typeTitle }) })
    } catch (error) {
      setMessage({ type: 'error', text: t('settings.defaultModels.messages.removeFailed') })
    } finally {
      setSaving(null)
    }
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            {t('settings.defaultModels.title')}
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          {t('settings.defaultModels.title')}
        </CardTitle>
        <CardDescription>
          {t('settings.defaultModels.description')}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {message && (
          <Alert className={message.type === 'error' ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}>
            <AlertDescription className={message.type === 'error' ? 'text-red-800' : 'text-green-800'}>
              {message.text}
            </AlertDescription>
          </Alert>
        )}

        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
          {defaultModelTypes.map((configType) => {
            const config = modelTypeConfig[configType]
            const currentDefault = defaultModels[configType]
            const Icon = config.icon
            const isSaving = saving === configType
            const compatibleModels = getCompatibleModels(models, configType)

            return (
              <Card key={configType} className="relative">
                <CardHeader className="pb-3">
                  <div className="flex items-center gap-2">
                    <div className={`p-2 rounded-lg ${config.color}`}>
                      <Icon className="h-4 w-4 text-white" />
                    </div>
                    <div className="flex-1">
                      <CardTitle className="text-sm">{t(`settings.defaultModels.types.${configType}.title`)}</CardTitle>
                      <CardDescription className="text-xs">
                        {t(`settings.defaultModels.types.${configType}.description`)}
                      </CardDescription>
                    </div>
                    {currentDefault && (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    )}
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  {currentDefault ? (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">
                            {getModelDisplayName(currentDefault.model)}
                          </p>
                          <div className="flex items-center gap-1 mt-1">
                            <Badge variant="secondary" className="text-xs">
                              {getModelProviderLabel(currentDefault.model)}
                            </Badge>
                            <Badge variant="outline" className="text-xs">
                              {t(`models.tabs.${getModelCategory(currentDefault.model)}`)}
                            </Badge>
                          </div>
                        </div>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleRemoveDefault(configType)}
                        disabled={isSaving}
                        className="w-full"
                      >
                        {isSaving ? (
                          <Loader2 className="h-3 w-3 animate-spin mr-1" />
                        ) : null}
                        {t('settings.defaultModels.actions.clearDefault')}
                      </Button>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <Label className="text-sm">{t('settings.defaultModels.labels.selectDefault')}</Label>
                      <Select
                        value=""
                        onValueChange={(value) =>
                          handleSetDefault(configType, parseInt(value))
                        }
                        options={compatibleModels.map((model) => ({
                          value: model.id.toString(),
                          label: `${getModelDisplayName(model)} (${getModelProviderLabel(model)})`,
                        }))}
                        placeholder={t('settings.defaultModels.labels.selectModel')}
                      />
                      {compatibleModels.length === 0 && (
                        <p className="text-xs text-muted-foreground">
                          {t('settings.defaultModels.empty.noModels')}
                        </p>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            )
          })}
        </div>

        <div className="bg-muted/50 p-4 rounded-lg">
          <div className="flex items-start gap-2">
            <AlertCircle className="h-4 w-4 text-muted-foreground mt-0.5" />
            <div className="space-y-1">
              <p className="text-sm font-medium">{t('settings.defaultModels.guide.title')}</p>
              <ul className="text-xs text-muted-foreground space-y-1">
                <li>{t('settings.defaultModels.guide.items.personalOnly')}</li>
                <li>{t('settings.defaultModels.guide.items.fallbackFirstAvailable')}</li>
                <li>{t('settings.defaultModels.guide.items.addOnModelsPage')}</li>
                <li>{t('settings.defaultModels.guide.items.adminShared')}</li>
              </ul>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
