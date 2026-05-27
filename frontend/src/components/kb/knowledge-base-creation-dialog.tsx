import React, { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Stepper } from "@/components/ui/stepper"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select } from "@/components/ui/select"
import { getApiUrl } from "@/lib/utils"
import { appendIngestionConfigToFormData, normalizeIngestionConfigForFilename } from "@/lib/ingestion-form"
import { findMatchingIngestionTask, getKBTaskProgressDetail, getKBTaskProgressPercent, KBProgressTask } from "@/lib/kb-progress"
import {
  buildKnowledgeBaseErrorResult,
  getKnowledgeBaseErrorToastContent,
  KnowledgeBaseIngestionResultLike,
  normalizeKnowledgeBaseIngestionResult,
} from "@/lib/kb-ingest-feedback"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest, getUploadErrorMessage, isJsonRecord, parseApiResponse, UPLOAD_ERROR_MESSAGES } from "@/lib/api-wrapper"
import { Model } from "@/lib/models"
import {
  Upload,
  Globe,
  Settings,
  CheckCircle,
  Clock,
  XCircle,
  AlertCircle,
  FileText,
  Cloud,
  Database,
  ChevronDown,
  ChevronUp,
  ArrowRight,
  ArrowLeft,
} from "lucide-react"
import { toast } from "@/components/ui/sonner"
import { CloudConnectDialog, CloudFile } from "./cloud-connect-dialog"

type IngestionResult = ReturnType<typeof normalizeKnowledgeBaseIngestionResult>

function getKnowledgeBaseToastCopy(
  t: ReturnType<typeof useI18n>["t"],
  genericTitle: string
) {
  return {
    genericTitle,
    embeddingTitle: t("kb.errors.embeddingModelUnavailable"),
    embeddingDescription: t("kb.errors.embeddingModelUnavailableHint"),
    rollbackTitle: t("kb.errors.rollbackFailed"),
    rollbackDescription: t("kb.errors.rollbackFailedHint"),
  }
}

interface WebIngestionResult {
  status: string
  collection: string
  total_urls_found: number
  pages_crawled: number
  pages_failed: number
  documents_created: number
  chunks_created: number
  embeddings_created: number
  crawled_urls: string[]
  failed_urls: Record<string, string>
  message: string
  warnings: string[]
  elapsed_time_ms: number
}

function buildWebIngestionErrorResult(
  collection: string,
  message: string
): WebIngestionResult {
  return {
    status: "error",
    collection,
    total_urls_found: 0,
    pages_crawled: 0,
    pages_failed: 0,
    documents_created: 0,
    chunks_created: 0,
    embeddings_created: 0,
    crawled_urls: [],
    failed_urls: {},
    message,
    warnings: [],
    elapsed_time_ms: 0,
  }
}

interface KnowledgeBaseCreationDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSuccess?: (collectionNames?: string[]) => void
}

export function KnowledgeBaseCreationDialog({ open, onOpenChange, onSuccess }: KnowledgeBaseCreationDialogProps) {
  const { t } = useI18n()

  // State from KnowledgeBasePage
  const [newCollectionName, setNewCollectionName] = useState("")
  const [newCollectionDescription, setNewCollectionDescription] = useState("")
  const [activeImportTab, setActiveImportTab] = useState<"file" | "web" | "cloud">("file")
  const [currentStep, setCurrentStep] = useState(1)
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false)

  // File upload state
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadProgressDetail, setUploadProgressDetail] = useState<string | null>(null)
  const [ingestionResults, setIngestionResults] = useState<IngestionResult[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [currentUploadFileName, setCurrentUploadFileName] = useState<string | null>(null)
  const [currentUploadCollection, setCurrentUploadCollection] = useState<string | null>(null)
  const [completedUploadCount, setCompletedUploadCount] = useState(0)

  // Web ingestion state
  const [isWebIngesting, setIsWebIngesting] = useState(false)
  const [webIngestionProgress, setWebIngestionProgress] = useState(0)
  const [webIngestionResult, setWebIngestionResult] = useState<WebIngestionResult | null>(null)
  const [webIngestionConfig, setWebIngestionConfig] = useState({
    start_url: "",
    max_pages: 100,
    max_depth: 3,
    url_patterns: "",
    exclude_patterns: "",
    same_domain_only: true,
    content_selector: "",
    remove_selectors: "",
    concurrent_requests: 3,
    request_delay: 1.0,
    timeout: 30,
    respect_robots_txt: true,
  })

  // Cloud connect state
  const [selectedCloudProvider, setSelectedCloudProvider] = useState<string | null>(null)
  const [isCloudConnecting, setIsCloudConnecting] = useState(false)
  const [isCloudDialogOpen, setIsCloudDialogOpen] = useState(false)
  const [cloudSelections, setCloudSelections] = useState<Record<string, CloudFile[]>>({})

  const totalCloudFiles = Object.values(cloudSelections).reduce((acc, files) => acc + files.length, 0)

  // Ingestion config state
  const [ingestionConfig, setIngestionConfig] = useState({
    parse_method: "default",
    chunk_strategy: "recursive",
    chunk_size: 1000,
    chunk_overlap: 200,
    separators: "" as string,
    embedding_model_id: "",
    embedding_batch_size: 10,
    max_retries: 3,
    retry_delay: 1.0
  })

  // Embedding models state
  const [embeddingModels, setEmbeddingModels] = useState<Model[]>([])
  const trimmedCollectionName = newCollectionName.trim()
  const requiresExplicitCollectionName =
    (activeImportTab === "file" && selectedFiles.length > 1) ||
    (activeImportTab === "cloud" && totalCloudFiles > 1)

  useEffect(() => {
    if (open) {
      fetchEmbeddingModels()
    }
  }, [open])

  useEffect(() => {
    if (!isUploading || !currentUploadFileName || !currentUploadCollection) return

    let cancelled = false

    const pollProgress = async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/progress?task_type=ingestion`)
        if (!response.ok) return
        const data = await response.json()
        const tasks = (data.tasks || []) as KBProgressTask[]
        const task = findMatchingIngestionTask(tasks, currentUploadCollection, currentUploadFileName)
        if (!task || cancelled) return

        const detail = getKBTaskProgressDetail(task)
        const taskPercent = getKBTaskProgressPercent(task)
        if (detail) setUploadProgressDetail(detail)
        if (typeof taskPercent === "number") {
          const overall = ((completedUploadCount + taskPercent / 100) / Math.max(selectedFiles.length, 1)) * 100
          setUploadProgress(Math.max(0, Math.min(100, overall)))
        }
      } catch {
        // Ignore transient progress polling failures; upload request remains source of truth.
      }
    }

    pollProgress()
    const interval = window.setInterval(pollProgress, 1000)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [isUploading, currentUploadFileName, currentUploadCollection, completedUploadCount, selectedFiles.length])

  const fetchEmbeddingModels = async () => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/models/?category=embedding`)

      if (!response.ok) {
        throw new Error("Failed to fetch embedding models")
      }

      const models = await response.json() || []
      setEmbeddingModels(models)

      // Get user's default embedding model
      const defaultResponse = await apiRequest(`${getApiUrl()}/api/models/user-default`)
      if (defaultResponse.ok) {
        const defaultData = await defaultResponse.json()
        if (defaultData.embedding?.model?.model_id) {
          const defaultModelId = defaultData.embedding.model.model_id
          setIngestionConfig(prev => ({ ...prev, embedding_model_id: defaultModelId }))
        } else if (models.length > 0) {
          setIngestionConfig(prev => ({ ...prev, embedding_model_id: models[0].model_id }))
        }
      } else if (models.length > 0) {
        setIngestionConfig(prev => ({ ...prev, embedding_model_id: models[0].model_id }))
      }
    } catch (err) {
      console.error("Failed to fetch embedding models:", err)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()

    if (e.relatedTarget && e.currentTarget.contains(e.relatedTarget as Node)) {
      return
    }

    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const files = Array.from(e.dataTransfer.files)
      const allowedExtensions = [".pdf", ".txt", ".html", ".htm", ".md", ".doc", ".docx", ".xlsx", ".ppt", ".pptx", ".csv"]
      const validFiles = files.filter(file => {
        const fileName = file.name.toLowerCase()
        return allowedExtensions.some(ext => fileName.endsWith(ext))
      })

      if (validFiles.length !== files.length) {
        toast.error(t("kb.errors.unsupportedFileType"))
      }

      if (validFiles.length > 0) {
        setSelectedFiles(prev => [...prev, ...validFiles])
      }
    }
  }

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    setSelectedFiles(prev => [...prev, ...files])
  }

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "success":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "processing":
        return <Clock className="h-4 w-4 text-yellow-500" />
      case "error":
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return <AlertCircle className="h-4 w-4 text-gray-500" />
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 B"
    const k = 1024
    const sizes = ["B", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  const resetState = () => {
    setSelectedFiles([])
    setUploadProgress(0)
    setIngestionResults([])
    setWebIngestionResult(null)
    setNewCollectionName("")
    setNewCollectionDescription("")
    setActiveImportTab("file")
    setSelectedCloudProvider(null)
    setIsCloudDialogOpen(false)
    setCloudSelections({})
    setWebIngestionConfig({
      start_url: "",
      max_pages: 100,
      max_depth: 3,
      url_patterns: "",
      exclude_patterns: "",
      same_domain_only: true,
      content_selector: "",
      remove_selectors: "",
      concurrent_requests: 3,
      request_delay: 1.0,
      timeout: 30,
      respect_robots_txt: true,
    })
    setCurrentStep(1)
  }

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      toast.error(t("kb.errors.uploadFileRequired"))
      return
    }

    if (selectedFiles.length > 1 && !trimmedCollectionName) {
      toast.error(t("kb.errors.multiFileNameRequired"))
      return
    }

    setIsUploading(true)
    setUploadProgress(0)
    setUploadProgressDetail(null)
    setIngestionResults([])
    setCompletedUploadCount(0)

    const successfulCollections: string[] = []

    try {
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i]
        const formData = new FormData()

        const collectionName = trimmedCollectionName || file.name.replace(/\.[^/.]+$/, "")
        setCurrentUploadFileName(file.name)
        setCurrentUploadCollection(collectionName)
        setUploadProgressDetail(null)

        formData.append("file", file)
        formData.append("collection", collectionName)
        appendIngestionConfigToFormData(
          formData,
          normalizeIngestionConfigForFilename(ingestionConfig, file.name)
        )

        const response = await apiRequest(`${getApiUrl()}/api/kb/ingest`, {
          method: "POST",
          body: formData
        })

        const parsed = await parseApiResponse(response)

        if (!response.ok) {
          const errorData = isJsonRecord(parsed.data) ? parsed.data : {}
          if (errorData.status === 'error') {
            setIngestionResults(prev => [
              ...prev,
              normalizeKnowledgeBaseIngestionResult(
                errorData as unknown as KnowledgeBaseIngestionResultLike,
                { collection: collectionName, fileName: file.name }
              ),
            ])
            throw new Error((typeof errorData.message === 'string' && errorData.message) || t("kb.errors.uploadFailedFile", { name: file.name }))
          }
          const errorMessage = getUploadErrorMessage(response, parsed, {
            generic: t("kb.errors.uploadFailedFile", { name: file.name }) || `Failed to upload file: ${file.name}`,
            ...UPLOAD_ERROR_MESSAGES,
          })
          setIngestionResults(prev => [
            ...prev,
            normalizeKnowledgeBaseIngestionResult(
              buildKnowledgeBaseErrorResult(collectionName, errorMessage, undefined, file.name),
              { collection: collectionName, fileName: file.name }
            ),
          ])
          throw new Error(errorMessage)
        }

        const result = isJsonRecord(parsed.data)
          ? parsed.data as unknown as KnowledgeBaseIngestionResultLike
          : null
        if (!result) {
          throw new Error(t("kb.errors.uploadFailedFile", { name: file.name }))
        }
        const normalizedResult = normalizeKnowledgeBaseIngestionResult(
          result as unknown as KnowledgeBaseIngestionResultLike,
          { collection: collectionName, fileName: file.name }
        )
        setIngestionResults(prev => [...prev, normalizedResult])

        if (normalizedResult.status === "partial" && normalizedResult.failed_step) {
          throw new Error(normalizedResult.message || t("kb.errors.failedAtStep", { step: normalizedResult.failed_step }))
        }

        successfulCollections.push(collectionName)
        setCompletedUploadCount(i + 1)
        setUploadProgress(((i + 1) / selectedFiles.length) * 100)
      }

      resetState()
      onOpenChange(false)
      onSuccess?.(successfulCollections)

    } catch (err) {
      const rawMessage = err instanceof Error ? err.message : t("kb.errors.uploadFailed")
      const toastContent = getKnowledgeBaseErrorToastContent(
        rawMessage,
        getKnowledgeBaseToastCopy(t, t("kb.errors.uploadFailed"))
      )
      toast.error(toastContent.title, {
        description: toastContent.description,
      })
      if (successfulCollections.length > 0) {
        onSuccess?.(successfulCollections)
      }
    } finally {
      setIsUploading(false)
      setCurrentUploadFileName(null)
      setCurrentUploadCollection(null)
      setUploadProgressDetail(null)
    }
  }

  const handleWebIngest = async () => {
    if (!webIngestionConfig.start_url.trim()) {
      toast.error(t("kb.errors.startUrlRequired"))
      return
    }

    setIsWebIngesting(true)
    setWebIngestionProgress(0)
    setWebIngestionResult(null)

    try {
      const formData = new FormData()

      const collectionName = trimmedCollectionName || "web_collection"

      formData.append("collection", collectionName)
      formData.append("start_url", webIngestionConfig.start_url)
      formData.append("max_pages", webIngestionConfig.max_pages.toString())
      formData.append("max_depth", webIngestionConfig.max_depth.toString())
      if (webIngestionConfig.url_patterns) {
        formData.append("url_patterns", webIngestionConfig.url_patterns)
      }
      if (webIngestionConfig.exclude_patterns) {
        formData.append("exclude_patterns", webIngestionConfig.exclude_patterns)
      }
      formData.append("same_domain_only", webIngestionConfig.same_domain_only.toString())
      if (webIngestionConfig.content_selector) {
        formData.append("content_selector", webIngestionConfig.content_selector)
      }
      if (webIngestionConfig.remove_selectors) {
        formData.append("remove_selectors", webIngestionConfig.remove_selectors)
      }
      formData.append("concurrent_requests", webIngestionConfig.concurrent_requests.toString())
      formData.append("request_delay", webIngestionConfig.request_delay.toString())
      formData.append("timeout", webIngestionConfig.timeout.toString())
      formData.append("respect_robots_txt", webIngestionConfig.respect_robots_txt.toString())

      appendIngestionConfigToFormData(formData, ingestionConfig)

      setWebIngestionProgress(10)

      const response = await apiRequest(`${getApiUrl()}/api/kb/ingest-web`, {
        method: "POST",
        body: formData
      })

      const parsed = await parseApiResponse(response)

      setWebIngestionProgress(50)

      if (!response.ok) {
        const errorData = isJsonRecord(parsed.data) ? parsed.data : {}
        if (errorData.status === 'error') {
          setWebIngestionResult(errorData as unknown as WebIngestionResult)
          throw new Error((typeof errorData.message === 'string' && errorData.message) || t("kb.errors.webIngestFailed"))
        }
        const errorMessage = getUploadErrorMessage(response, parsed, {
          generic: t("kb.errors.webIngestFailed") || "Website import failed",
          ...UPLOAD_ERROR_MESSAGES,
        })
        setWebIngestionResult(buildWebIngestionErrorResult(collectionName, errorMessage))
        throw new Error(errorMessage)
      }

      const result: WebIngestionResult | null = isJsonRecord(parsed.data)
        ? (parsed.data as unknown as WebIngestionResult)
        : null
      if (!result) {
        throw new Error(t("kb.errors.webIngestFailed"))
      }
      setWebIngestionResult(result)
      setWebIngestionProgress(100)
      if (result.status !== "success") {
        throw new Error(result.message || t("kb.errors.webIngestFailed"))
      }

      resetState()
      onOpenChange(false)
      onSuccess?.([collectionName])

    } catch (err) {
      const rawMessage = err instanceof Error ? err.message : t("kb.errors.webIngestFailed")
      const toastContent = getKnowledgeBaseErrorToastContent(
        rawMessage,
        getKnowledgeBaseToastCopy(t, t("kb.errors.webIngestFailed"))
      )
      toast.error(toastContent.title, {
        description: toastContent.description,
      })
    } finally {
      setIsWebIngesting(false)
      setWebIngestionProgress(0)
    }
  }

  const handleCloudIngest = async () => {
    if (totalCloudFiles === 0) return

    setIsCloudConnecting(true)
    setIngestionResults([])

    try {
      // Aggregate all selected files from all providers
      const filesToIngest = Object.entries(cloudSelections).flatMap(([provider, files]) =>
        files.map(file => ({ provider, fileId: file.id, fileName: file.name }))
      )

      // Determine collection name
      if (filesToIngest.length > 1 && !trimmedCollectionName) {
        toast.error(t("kb.errors.multiFileNameRequired"))
        return
      }

      let collectionName = trimmedCollectionName
      if (!collectionName && filesToIngest.length > 0) {
        // Use first file name without extension as default collection name
        collectionName = filesToIngest[0].fileName.replace(/\.[^/.]+$/, "")
      }
      if (!collectionName) collectionName = "cloud_collection"

      // Prepare separators
      let separators: string[] | undefined = undefined
      if (ingestionConfig.separators) {
        try {
          const parsed = JSON.parse(ingestionConfig.separators)
          if (Array.isArray(parsed) && parsed.every(s => typeof s === 'string')) {
            separators = parsed
          }
        } catch (e) {
          console.warn("Invalid separators JSON", e)
        }
      }

      const requestBody = {
        files: filesToIngest,
        collection: collectionName,
        parse_method: ingestionConfig.parse_method,
        chunk_strategy: ingestionConfig.chunk_strategy,
        chunk_size: ingestionConfig.chunk_size,
        chunk_overlap: ingestionConfig.chunk_overlap,
        separators: separators,
        embedding_model_id: ingestionConfig.embedding_model_id,
        embedding_batch_size: ingestionConfig.embedding_batch_size,
        max_retries: ingestionConfig.max_retries,
        retry_delay: ingestionConfig.retry_delay
      }

      const response = await apiRequest(`${getApiUrl()}/api/kb/ingest-cloud`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(requestBody)
      })

      const parsed = await parseApiResponse(response)

      if (!response.ok) {
        const errorMessage = getUploadErrorMessage(response, parsed, {
          generic: t("kb.errors.cloudIngestFailed") || "Cloud ingest failed",
          ...UPLOAD_ERROR_MESSAGES,
        })
        setIngestionResults([
          normalizeKnowledgeBaseIngestionResult(
            buildKnowledgeBaseErrorResult(
              collectionName,
              errorMessage,
              undefined,
              filesToIngest.length === 1 ? filesToIngest[0].fileName : undefined
            ),
            {
              collection: collectionName,
              fileName: filesToIngest.length === 1 ? filesToIngest[0].fileName : undefined,
            }
          ),
        ])
        throw new Error(errorMessage)
      }

      const results: IngestionResult[] = Array.isArray(parsed.data)
        ? (parsed.data as unknown as KnowledgeBaseIngestionResultLike[]).map((result, index) =>
            normalizeKnowledgeBaseIngestionResult(result, {
              collection: collectionName,
              fileName: filesToIngest[index]?.fileName,
            })
          )
        : []
      setIngestionResults(results)

      const failedResults = results.filter(result => result.status !== "success")
      if (failedResults.length > 0) {
        throw new Error(failedResults[0].message || t("kb.errors.cloudIngestFailed"))
      }

      toast.success(t("kb.dialog.fileUpload.processSuccess"))

      // Reset and close
      resetState()
      onOpenChange(false)
      onSuccess?.()
    } catch (error) {
      console.error("Cloud ingest error:", error)
      const rawMessage = error instanceof Error
        ? error.message
        : t("kb.dialog.fileUpload.processFailed")
      const toastContent = getKnowledgeBaseErrorToastContent(
        rawMessage,
        getKnowledgeBaseToastCopy(
          t,
          t("kb.errors.cloudIngestFailed")
        )
      )
      toast.error(toastContent.title, {
        description: toastContent.description,
      })
    } finally {
      setIsCloudConnecting(false)
    }
  }

  const cloudProviders = [
    {
      id: "google-drive",
      name: t("kb.dialog.cloudConnect.googleDrive"),
      hasDrives: true,
      authPath: "google",
      logo: "/google-drive.svg"
    },
  ]

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="sm:max-w-[600px] max-h-[85vh] flex flex-col p-0 bg-slate-50">
          <div className="p-6 pb-0">
            <DialogHeader>
              <DialogTitle className="text-xl font-bold">{t("kb.dialog.createTitle")}</DialogTitle>
              <DialogDescription className="text-primary mt-1">
                {t("kb.dialog.steps.stepCount", { currentStep })} — {
                  currentStep === 1 ? t("kb.dialog.steps.nameIt") :
                    currentStep === 2 ? t("kb.dialog.steps.addContentTitle") :
                      t("kb.dialog.steps.reviewTitle")
                }
              </DialogDescription>
            </DialogHeader>
            <div className="mt-6">
              <Stepper
                currentStep={currentStep}
                steps={[
                  { label: t("kb.dialog.steps.nameIt"), content: <div /> },
                  { label: t("kb.dialog.steps.addContent"), content: <div /> },
                  { label: t("kb.dialog.steps.reviewTitle"), content: <div /> }
                ]}
              />
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-6 pb-6">
            {currentStep === 1 && (
              <div className="space-y-6 mt-4">
                <div>
                  <Label htmlFor="collection_name" className="text-sm font-medium">{t("kb.dialog.basicInfo.nameLabel")} {t("common.optional")}</Label>
                  <Input
                    id="collection_name"
                    value={newCollectionName}
                    onChange={(e) => setNewCollectionName(e.target.value)}
                    placeholder={t("kb.dialog.basicInfo.namePlaceholder")}
                    className="mt-1.5"
                  />
                  {requiresExplicitCollectionName && !trimmedCollectionName && (
                    <p className="mt-2 text-sm text-destructive">
                      {t("kb.dialog.basicInfo.multiFileRequiredHint")}
                    </p>
                  )}
                </div>
                <div>
                  <Label htmlFor="collection_description" className="text-sm font-medium">{t("kb.dialog.basicInfo.descriptionLabel")} {t("common.optional")}</Label>
                  <Textarea
                    id="collection_description"
                    value={newCollectionDescription}
                    onChange={(e) => setNewCollectionDescription(e.target.value)}
                    placeholder={t("kb.dialog.basicInfo.descriptionPlaceholder")}
                    className="mt-1.5 h-32"
                  />
                </div>
              </div>
            )}

            {currentStep === 2 && (
              <div className="space-y-6 mt-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <Card
                    className={`p-6 cursor-pointer flex flex-col items-center justify-center gap-2 transition-all text-center ${activeImportTab === 'file' ? 'border-primary bg-primary/5 border-2' : 'hover:bg-muted'}`}
                    onClick={() => setActiveImportTab('file')}
                  >
                    <Upload className="w-6 h-6 text-primary mb-2" />
                    <span className="font-bold text-base">{t("kb.dialog.tabs.file")}</span>
                    <span className="text-xs text-muted-foreground">{t("kb.dialog.fileUpload.supportedFormats")}</span>
                  </Card>
                  <Card
                    className={`p-6 cursor-pointer flex flex-col items-center justify-center gap-2 transition-all text-center ${activeImportTab === 'web' ? 'border-primary bg-primary/5 border-2' : 'hover:bg-muted'}`}
                    onClick={() => setActiveImportTab('web')}
                  >
                    <Globe className="w-6 h-6 text-primary mb-2" />
                    <span className="font-bold text-base">{t("kb.dialog.tabs.web")}</span>
                    <span className="text-xs text-muted-foreground">{t("kb.dialog.tabs.webDesc")}</span>
                  </Card>
                  <Card
                    className={`p-6 cursor-pointer flex flex-col items-center justify-center gap-2 transition-all text-center ${activeImportTab === 'cloud' ? 'border-primary bg-primary/5 border-2' : 'hover:bg-muted'}`}
                    onClick={() => setActiveImportTab('cloud')}
                  >
                    <Cloud className="w-6 h-6 text-primary mb-2" />
                    <span className="font-bold text-base">{t("kb.dialog.tabs.cloud")}</span>
                    <span className="text-xs text-muted-foreground">{t("kb.dialog.tabs.cloudDesc")}</span>
                  </Card>

                </div>

                {activeImportTab === 'file' && (
                  <div className="space-y-4 w-full bg-white rounded-lg p-4 border border-dashed">
                    <div
                      className={`w-full rounded-lg p-8 text-center cursor-pointer transition-colors ${isDragging ? "bg-primary/10" : ""}`}
                      onClick={() => fileInputRef.current?.click()}
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                    >
                      <Cloud className={`h-8 w-8 mx-auto mb-4 ${isDragging ? "text-primary" : "text-blue-500"}`} />
                      <p className="text-sm font-bold mb-2">{t("kb.dialog.fileUpload.dropOrClick")}</p>
                      <p className="text-xs text-muted-foreground mb-4">
                        {t("kb.dialog.fileUpload.supportedFormats")}
                      </p>
                      <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        accept=".pdf,.txt,.html,.htm,.md,.doc,.docx,.xlsx,.ppt,.pptx,.csv"
                        onChange={handleFileSelect}
                        className="hidden"
                        id="file-upload"
                      />
                    </div>

                    {selectedFiles.length > 0 && (
                      <div className="mt-4">
                        <Label className="text-sm font-medium">{t("kb.dialog.fileUpload.selectedTitle")}</Label>
                        <ScrollArea className="h-32 border rounded-md p-2 mt-2 bg-slate-50">
                          <div className="space-y-2">
                            {selectedFiles.map((file, index) => (
                              <div key={index} className="flex items-center justify-between p-2 bg-white rounded border">
                                <div className="flex items-center gap-2">
                                  <FileText className="h-4 w-4 text-muted-foreground" />
                                  <span className="text-sm font-medium">{file.name}</span>
                                  <Badge variant="secondary" className="text-xs font-normal">
                                    {formatFileSize(file.size)}
                                  </Badge>
                                </div>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
                                  onClick={(e) => { e.stopPropagation(); removeFile(index); }}
                                >
                                  X
                                </Button>
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      </div>
                    )}
                  </div>
                )}


                {activeImportTab === 'cloud' && (
                  <div className="space-y-4 w-full bg-white rounded-lg p-6 border">
                    <div className="flex items-center gap-2">
                      <Cloud className="h-5 w-5 text-blue-500" />
                      <h3 className="text-lg font-medium">{t("kb.dialog.cloudConnect.title")}</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {t("kb.dialog.cloudConnect.description")}
                    </p>

                    <div className="grid grid-cols-2 gap-4">
                      {cloudProviders.map((provider) => (
                        <Card
                          key={provider.id}
                          className={`p-4 cursor-pointer transition-all hover:border-blue-500 relative ${cloudSelections[provider.id]?.length > 0 ? "border-blue-500 border-2" : ""}`}
                          onClick={() => {
                            setSelectedCloudProvider(provider.id)
                            setIsCloudDialogOpen(true)
                          }}
                        >
                          <div className="flex items-center gap-2">
                            <img src={provider.logo} alt={provider.name} className="h-8 w-8" />
                            <span className="font-medium">{provider.name}</span>
                          </div>
                          {cloudSelections[provider.id]?.length > 0 && (
                            <Badge variant="default" className="absolute top-2 right-2 w-4 h-4 flex items-center justify-center rounded-full text-[10px]">
                              {cloudSelections[provider.id].length}
                            </Badge>
                          )}
                        </Card>
                      ))}
                    </div>

                    {totalCloudFiles > 0 && (
                      <div className="mt-6">
                        <Label>{t("kb.dialog.fileUpload.selectedTitle")}</Label>
                        <ScrollArea className="h-32 border rounded-md p-2 mt-2">
                          <div className="space-y-2">
                            {Object.entries(cloudSelections)
                              .flatMap(([providerId, files]) => {
                                const provider = cloudProviders.find(p => p.id === providerId)
                                return files.map(file => ({ ...file, providerId, provider }))
                              })
                              .map((file) => (
                                <div key={`${file.providerId}-${file.id}`} className="flex items-center justify-between p-2 bg-muted rounded">
                                  <div className="flex items-center gap-2">
                                    {file.provider ? (
                                      <img src={file.provider.logo} alt={file.provider.name} className="h-4 w-4" />
                                    ) : (
                                      <Cloud className="h-4 w-4 text-blue-500" />
                                    )}
                                    <span className="text-xs text-muted-foreground">
                                      {file.provider ? file.provider.name : file.providerId}:
                                    </span>
                                    <span className="text-sm truncate max-w-[200px]" title={file.name}>{file.name}</span>
                                    {file.size && (
                                      <Badge variant="outline" className="text-xs">
                                        {file.size}
                                      </Badge>
                                    )}
                                  </div>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-6 w-6 p-0"
                                    onClick={() => {
                                      setCloudSelections(prev => ({
                                        ...prev,
                                        [file.providerId]: prev[file.providerId].filter(f => f.id !== file.id)
                                      }))
                                    }}
                                  >
                                    <XCircle className="h-4 w-4 text-muted-foreground hover:text-destructive" />
                                  </Button>
                                </div>
                              ))}
                          </div>
                        </ScrollArea>
                      </div>
                    )}
                  </div>
                )}

                {activeImportTab === 'web' && (
                  <div className="space-y-4 w-full bg-white rounded-lg p-6 border">
                    <div>
                      <Label htmlFor="start_url" className="text-sm font-medium">{t("kb.dialog.webImport.basic.startUrl")} <span className="text-destructive">*</span></Label>
                      <Input
                        id="start_url"
                        placeholder="https://help.example.com"
                        value={webIngestionConfig.start_url}
                        onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, start_url: e.target.value }))}
                        className="mt-1.5"
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label htmlFor="max_pages" className="text-sm font-medium">{t("kb.dialog.webImport.basic.maxPages")}</Label>
                        <Input
                          id="max_pages"
                          type="number"
                          value={webIngestionConfig.max_pages}
                          onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, max_pages: parseInt(e.target.value) || 100 }))}
                          className="mt-1.5"
                        />
                      </div>
                      <div>
                        <Label htmlFor="max_depth" className="text-sm font-medium">{t("kb.dialog.webImport.basic.crawlDepth")}</Label>
                        <Input
                          id="max_depth"
                          type="number"
                          min="1"
                          max="10"
                          value={webIngestionConfig.max_depth}
                          onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, max_depth: parseInt(e.target.value) || 3 }))}
                          className="mt-1.5"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {currentStep === 3 && (
              <div className="space-y-6 mt-4">
                <div className="bg-primary/5 rounded-lg p-4 flex flex-col gap-2 border border-primary/20">
                  <div className="flex items-center gap-2 font-bold text-sm">
                    <Database className="w-4 h-4 text-primary" />
                    {newCollectionName || "KB " + new Date().toLocaleString()}
                  </div>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground ml-6">
                    <FileText className="w-4 h-4" />
                    {activeImportTab === 'file'
                      ? `${selectedFiles.length} ${t("kb.dialog.steps.filesSelected")}`
                      : `${t("kb.dialog.steps.crawlStartingFrom")} ${webIngestionConfig.start_url}`}
                  </div>
                </div>

                <div className="space-y-4">
                  <h3 className="text-base font-bold">{t("kb.index.title")}</h3>

                  <div className="space-y-4">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                      <div>
                        <Label htmlFor="parse_method" className="text-sm font-bold">{t("kb.index.parseMethod")}</Label>
                        <p className="text-xs text-muted-foreground">{t("kb.index.parseMethodDesc")}</p>
                      </div>
                      <Select
                        value={ingestionConfig.parse_method}
                        onValueChange={(value) => setIngestionConfig(prev => ({ ...prev, parse_method: value }))}
                        options={[
                          { value: "default", label: t("kb.index.parseOptions.default") },
                          { value: "pypdf", label: t("kb.index.parseOptions.pypdf") },
                          { value: "pdfplumber", label: t("kb.index.parseOptions.pdfplumber") },
                          { value: "unstructured", label: t("kb.index.parseOptions.unstructured") },
                          { value: "pymupdf", label: t("kb.index.parseOptions.pymupdf") },
                          { value: "deepdoc", label: t("kb.index.parseOptions.deepdoc") },
                        ]}
                        className="w-full md:w-64"
                      />
                    </div>

                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                      <div>
                        <Label htmlFor="chunk_strategy" className="text-sm font-bold">{t("kb.index.chunkStrategy")}</Label>
                        <p className="text-xs text-muted-foreground">{t("kb.index.chunkStrategyDesc")}</p>
                      </div>
                      <Select
                        value={ingestionConfig.chunk_strategy}
                        onValueChange={(value) => setIngestionConfig(prev => ({ ...prev, chunk_strategy: value }))}
                        options={[
                          { value: "recursive", label: t("kb.index.chunkOptions.recursive") },
                          { value: "fixed_size", label: t("kb.index.chunkOptions.fixed_size") },
                          { value: "markdown", label: t("kb.index.chunkOptions.markdown") },
                        ]}
                        className="w-full md:w-64"
                      />
                    </div>

                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                      <div>
                        <Label htmlFor="embedding_model_id" className="text-sm font-bold">{t("kb.index.embeddingModelId")}</Label>
                        <p className="text-xs text-muted-foreground">{t("kb.index.embeddingModelDesc")}</p>
                      </div>
                      <Select
                        value={ingestionConfig.embedding_model_id}
                        onValueChange={(value: string) => setIngestionConfig(prev => ({ ...prev, embedding_model_id: value }))}
                        options={embeddingModels.map(model => ({ value: model.model_id, label: model.name || model.model_id }))}
                        className="w-full md:w-64"
                      />
                    </div>
                  </div>

                  <div className="pt-2">
                    <button
                      className="flex items-center gap-2 text-sm font-bold"
                      onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                    >
                      <Settings className="w-4 h-4" />
                      {t("kb.dialog.webImport.advanced.title")}
                      {showAdvancedSettings ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>

                    {showAdvancedSettings && (
                      <div className="grid grid-cols-2 gap-4 mt-4 bg-slate-50 p-4 rounded-lg border">
                        <div>
                          <Label htmlFor="chunk_size" className="text-sm font-bold">{t("kb.index.chunkSize")}</Label>
                          <Input
                            id="chunk_size"
                            type="number"
                            value={ingestionConfig.chunk_size}
                            onChange={(e) => setIngestionConfig(prev => ({ ...prev, chunk_size: parseInt(e.target.value) || 1000 }))}
                            className="mt-1.5"
                          />
                        </div>
                        <div>
                          <Label htmlFor="chunk_overlap" className="text-sm font-bold">{t("kb.index.chunkOverlap")}</Label>
                          <Input
                            id="chunk_overlap"
                            type="number"
                            value={ingestionConfig.chunk_overlap}
                            onChange={(e) => setIngestionConfig(prev => ({ ...prev, chunk_overlap: parseInt(e.target.value) || 200 }))}
                            className="mt-1.5"
                          />
                        </div>
                        <div>
                          <Label htmlFor="embedding_batch_size" className="text-sm font-bold">{t("kb.index.embeddingBatchSize")}</Label>
                          <Input
                            id="embedding_batch_size"
                            type="number"
                            value={ingestionConfig.embedding_batch_size}
                            onChange={(e) => setIngestionConfig(prev => ({ ...prev, embedding_batch_size: parseInt(e.target.value) || 10 }))}
                            className="mt-1.5"
                          />
                        </div>
                        {ingestionConfig.chunk_strategy === "recursive" && (
                          <div>
                            <Label htmlFor="separators" className="text-sm font-bold">{t("kb.index.separators")}</Label>
                            <Input
                              id="separators"
                              type="text"
                              value={ingestionConfig.separators ?? ""}
                              onChange={(e) => setIngestionConfig(prev => ({ ...prev, separators: e.target.value }))}
                              placeholder="\n\n, \n, ..."
                              className="mt-1.5"
                            />
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Progress/Results overlays would go here if needed, but we usually show them as toast or disable UI. We'll add them if uploading is true */}
                  {(isUploading || isWebIngesting || isCloudConnecting) && (
                    <div className="mt-4 p-4 bg-white rounded-lg border">
                      <div className="flex justify-between text-sm mb-2">
                        <span className="font-medium">
                          {isUploading ? t("kb.dialog.fileUpload.progressTitle") :
                            isWebIngesting ? t("kb.dialog.webImport.status.progressTitle") :
                              t("kb.dialog.cloudConnect.connecting")}
                        </span>
                        <span>{Math.round(isUploading ? uploadProgress : webIngestionProgress)}%</span>
                      </div>
                      <Progress value={isUploading ? uploadProgress : webIngestionProgress} className="w-full" />
                      {(uploadProgressDetail || (isWebIngesting && t("kb.dialog.webImport.status.crawling"))) && (
                        <p className="text-xs text-muted-foreground mt-2">
                          {uploadProgressDetail || t("kb.dialog.webImport.status.crawling")}
                        </p>
                      )}
                    </div>
                  )}

                  {ingestionResults.length > 0 && (
                    <div className="mt-4">
                      <Label className="text-sm font-bold">{t("kb.detail.process.title")}</Label>
                      <ScrollArea className="h-32 border rounded-md p-2 mt-2 bg-white">
                        <div className="space-y-2">
                          {ingestionResults.map((result, index) => (
                            <div key={index} className="flex flex-col gap-1 p-2 bg-slate-50 rounded border">
                              <div className="flex items-center gap-2">
                                {getStatusIcon(result.status)}
                                <span className="text-sm font-medium">{result.file_name || result.collection}</span>
                                {result.status === 'success' && (
                                  <>
                                    <Badge variant="secondary" className="text-xs font-normal">
                                      {result.document_count} {t("kb.dialog.fileUpload.processResult.createDocuments")}
                                    </Badge>
                                    <Badge variant="secondary" className="text-xs font-normal">
                                      {result.chunks_count} {t("kb.dialog.fileUpload.processResult.textChunks")}
                                    </Badge>
                                  </>
                                )}
                              </div>
                              {result.status !== 'success' && result.message && (
                                <p className="text-xs text-destructive ml-6 break-all">{result.message}</p>
                              )}
                            </div>
                          ))}
                        </div>
                      </ScrollArea>
                    </div>
                  )}

                  {webIngestionResult && (
                    <div className="mt-4">
                      <Label className="text-sm font-bold">{t("kb.detail.process.title")}</Label>
                      <Card className="p-4 mt-2">
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            {getStatusIcon(webIngestionResult.status)}
                            <span className="font-medium">{t(webIngestionResult.status === "success" ? "kb.dialog.webImport.status.success" : "kb.dialog.webImport.status.failed")}</span>
                          </div>
                          <p className="text-sm text-muted-foreground">{webIngestionResult.message}</p>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                            <div>
                              <div className="text-2xl font-bold">{webIngestionResult.pages_crawled}</div>
                              <div className="text-xs text-muted-foreground">{t("kb.dialog.webImport.result.pages")}</div>
                            </div>
                            <div>
                              <div className="text-2xl font-bold">{webIngestionResult.documents_created}</div>
                              <div className="text-xs text-muted-foreground">{t("kb.dialog.fileUpload.processResult.createDocuments")}</div>
                            </div>
                            <div>
                              <div className="text-2xl font-bold">{webIngestionResult.chunks_created}</div>
                              <div className="text-xs text-muted-foreground">{t("kb.dialog.fileUpload.processResult.textChunks")}</div>
                            </div>
                            <div>
                              <div className="text-2xl font-bold">{webIngestionResult.embeddings_created}</div>
                              <div className="text-xs text-muted-foreground">{t("kb.dialog.fileUpload.processResult.vectors")}</div>
                            </div>
                          </div>
                          {webIngestionResult.warnings && webIngestionResult.warnings.length > 0 && (
                            <details className="mt-4">
                              <summary className="cursor-pointer text-sm font-medium">{t("kb.dialog.webImport.result.viewWarnings")}</summary>
                              <div className="mt-2 space-y-1">
                                {webIngestionResult.warnings.map((warning, index) => (
                                  <div key={index} className="text-xs text-yellow-600 bg-yellow-50 dark:bg-yellow-950 p-2 rounded">
                                    {warning}
                                  </div>
                                ))}
                              </div>
                            </details>
                          )}
                        </div>
                      </Card>
                    </div>
                  )}

                </div>
              </div>
            )}
          </div>

          <div className="p-6 pt-4 flex justify-between border-t bg-white rounded-b-lg">
            <Button variant="outline" onClick={() => {
              resetState()
              onOpenChange(false)
            }}>
              {t("common.cancel")}
            </Button>
            <div className="flex gap-2">
              {currentStep > 1 && (
                <Button
                  variant="outline"
                  onClick={() => setCurrentStep(prev => prev - 1)}
                  disabled={isUploading || isWebIngesting || isCloudConnecting}
                >
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  {t("common.back")}
                </Button>
              )}
              {currentStep < 3 ? (
                <Button
                  onClick={() => setCurrentStep(prev => prev + 1)}
                  disabled={
                    (currentStep === 2 && activeImportTab === "file" && selectedFiles.length === 0) ||
                    (currentStep === 2 && activeImportTab === "web" && !webIngestionConfig.start_url)
                  }
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  {t("common.next")}
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              ) : (
                <Button
                  onClick={() => {
                    if (activeImportTab === "web") {
                      handleWebIngest()
                    } else if (activeImportTab === "cloud") {
                      handleCloudIngest()
                    } else {
                      handleUpload()
                    }
                  }}
                  disabled={
                    isUploading ||
                    isWebIngesting ||
                    isCloudConnecting ||
                    (activeImportTab === "file" && selectedFiles.length === 0)
                  }
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  {isUploading || isWebIngesting || isCloudConnecting ? (
                    <span className="flex items-center gap-2">
                      <Clock className="w-4 h-4 animate-spin" />
                      {t("kb.dialog.fileUpload.processing")}
                    </span>
                  ) : (
                    <span className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4" />
                      {t("kb.dialog.createButton")}
                    </span>
                  )}
                </Button>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Cloud Connect Dialog */}
      <CloudConnectDialog
        open={isCloudDialogOpen}
        onOpenChange={setIsCloudDialogOpen}
        provider={cloudProviders.find(p => p.id === selectedCloudProvider) || null}
        initialSelectedFiles={
          selectedCloudProvider ? cloudSelections[selectedCloudProvider] || [] : []
        }
        onConfirm={(files) => {
          if (selectedCloudProvider) {
            setCloudSelections((prev) => ({
              ...prev,
              [selectedCloudProvider]: files,
            }))
          }
        }}
      />
    </>
  )
}
