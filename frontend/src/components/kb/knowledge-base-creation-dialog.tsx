import React, { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select } from "@/components/ui/select"
import { getApiUrl } from "@/lib/utils"
import { appendIngestionConfigToFormData, normalizeIngestionConfigForFilename } from "@/lib/ingestion-form"
import { findMatchingIngestionTask, getKBTaskProgressDetail, getKBTaskProgressPercent, KBProgressTask } from "@/lib/kb-progress"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
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
} from "lucide-react"
import { toast } from "sonner"
import { CloudConnectDialog, CloudFile } from "./cloud-connect-dialog"

interface IngestionResult {
  collection: string
  document_count: number
  chunks_count: number
  status: string
  message: string
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

        if (!response.ok) {
          const errorData = await response.json()
          if (errorData.status === 'error') {
            setIngestionResults(prev => [...prev, errorData])
            throw new Error(errorData.message || t("kb.errors.uploadFailedFile", { name: file.name }))
          }
          throw new Error(errorData.detail || t("kb.errors.uploadFailedFile", { name: file.name }))
        }

        const result = await response.json()
        setIngestionResults(prev => [...prev, result])

        if (result.status === "partial" && result.failed_step) {
          throw new Error(result.message || t("kb.errors.failedAtStep", { step: result.failed_step }))
        }

        successfulCollections.push(collectionName)
        setCompletedUploadCount(i + 1)
        setUploadProgress(((i + 1) / selectedFiles.length) * 100)
      }

      resetState()
      onOpenChange(false)
      onSuccess?.(successfulCollections)

    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("kb.errors.uploadFailed"))
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

      setWebIngestionProgress(50)

      if (!response.ok) {
        const errorData = await response.json()
        if (errorData.status === 'error') {
          setWebIngestionResult(errorData)
          throw new Error(errorData.message || t("kb.errors.webIngestFailed"))
        }
        throw new Error(errorData.detail || t("kb.errors.webIngestFailed"))
      }

      const result: WebIngestionResult = await response.json()
      setWebIngestionResult(result)
      setWebIngestionProgress(100)

      resetState()
      onOpenChange(false)
      onSuccess?.([collectionName])

    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("kb.errors.webIngestFailed"))
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

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || t("kb.errors.cloudIngestFailed"))
      }

      const results: IngestionResult[] = await response.json()
      setIngestionResults(results)

      // Check for errors
      const errors = results.filter(r => r.status === 'error')
      if (errors.length > 0) {
        toast.error(t("kb.errors.someFilesFailed"))
        // Don't close dialog so user can see errors
      } else {
        toast.success(t("kb.dialog.fileUpload.processSuccess"))

        // Reset and close
        resetState()
        onOpenChange(false)
        onSuccess?.()
      }
    } catch (error) {
      console.error("Cloud ingest error:", error)
      toast.error(error instanceof Error ? error.message : t("kb.dialog.fileUpload.processFailed"))
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
        <DialogContent className="sm:max-w-[500px] max-h-[85vh] flex flex-col p-0">
          <div className="p-6 pb-0">
            <DialogHeader>
              <DialogTitle>{t("kb.dialog.createTitle")}</DialogTitle>
              <DialogDescription>
                {t("kb.dialog.createDescription")}
              </DialogDescription>
            </DialogHeader>
          </div>
          <div className="flex-1 overflow-y-auto px-6 space-y-6 flex flex-col gap-6">
            {/* Basic Information */}
            <div>
              <h3 className="text-lg font-medium">{t("kb.dialog.basicInfo.title")}</h3>
              <div>
                <Label htmlFor="collection_name">{t("kb.dialog.basicInfo.nameLabel")}</Label>
                <Input
                  id="collection_name"
                  value={newCollectionName}
                  onChange={(e) => setNewCollectionName(e.target.value)}
                  placeholder={t("kb.dialog.basicInfo.namePlaceholder")}
                />
                {requiresExplicitCollectionName && !trimmedCollectionName && (
                  <p className="mt-2 text-sm text-destructive">
                    {t("kb.dialog.basicInfo.multiFileRequiredHint")}
                  </p>
                )}
              </div>
              <div>
                <Label htmlFor="collection_description">{t("kb.dialog.basicInfo.descriptionLabel")}</Label>
                <Textarea
                  id="collection_description"
                  value={newCollectionDescription}
                  onChange={(e) => setNewCollectionDescription(e.target.value)}
                  placeholder={t("kb.dialog.basicInfo.descriptionPlaceholder")}
                />
              </div>
            </div>

            {/* Tabs: File Upload / Web Import / Cloud Connect */}
            <Tabs value={activeImportTab} onValueChange={(v) => setActiveImportTab(v as "file" | "web" | "cloud")} className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="file">
                  <FileText size={16} className="mr-2" />
                  {t("kb.dialog.tabs.file")}
                </TabsTrigger>
                <TabsTrigger value="web">
                  <Globe size={16} className="mr-2" />
                  {t("kb.dialog.tabs.web")}
                </TabsTrigger>
                <TabsTrigger value="cloud">
                  <Cloud size={16} className="mr-2" />
                  {t("kb.dialog.tabs.cloud")}
                </TabsTrigger>
              </TabsList>

              {/* File Upload Tab */}
              <TabsContent value="file" className="space-y-4 w-full">
                {/* File Upload */}
                <div className="space-y-4 w-full">
                  <div className="flex items-center gap-2">
                    <FileText className="h-5 w-5 text-blue-500" />
                    <h3 className="text-lg font-medium">{t("kb.dialog.fileUpload.title")}</h3>
                  </div>
                  {/* File Selection Area */}
                  <div
                    className={`w-full border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:bg-muted/50 transition-colors ${
                      isDragging ? "border-primary bg-primary/10" : "border-border"
                    }`}
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                  >
                    <Upload className={`h-12 w-12 mx-auto mb-4 ${isDragging ? "text-primary" : "text-muted-foreground"}`} />
                    <p className="text-sm font-medium mb-2">{t("kb.dialog.fileUpload.dropOrClick")}</p>
                    <p className="text-sm text-muted-foreground mb-4">
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

                  {/* Selected Files List */}
                  {selectedFiles.length > 0 && (
                    <div>
                      <Label>{t("kb.dialog.fileUpload.selectedTitle")}</Label>
                      <ScrollArea className="h-32 border rounded-md p-2">
                        <div className="space-y-2">
                          {selectedFiles.map((file, index) => (
                            <div key={index} className="flex items-center justify-between p-2 bg-muted rounded">
                              <div className="flex items-center gap-2">
                                <FileText className="h-4 w-4" />
                                <span className="text-sm">{file.name}</span>
                                <Badge variant="outline" className="text-xs">
                                  {formatFileSize(file.size)}
                                </Badge>
                              </div>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => removeFile(index)}
                              >
                                X
                              </Button>
                            </div>
                          ))}
                        </div>
                      </ScrollArea>
                    </div>
                  )}

                  {/* Upload Progress */}
                  {isUploading && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>{t("kb.dialog.fileUpload.progressTitle")}</span>
                        <span>{Math.round(uploadProgress)}%</span>
                      </div>
                      <Progress value={uploadProgress} className="w-full" />
                      {uploadProgressDetail && (
                        <p className="text-xs text-muted-foreground">{uploadProgressDetail}</p>
                      )}
                    </div>
                  )}

                  {/* Upload Result */}
                  {ingestionResults.length > 0 && (
                    <div>
                      <Label>{t("kb.detail.process.title")}</Label>
                      <ScrollArea className="h-32 border rounded-md p-2">
                        <div className="space-y-2">
                          {ingestionResults.map((result, index) => (
                            <div key={index} className="flex flex-col gap-1 p-2 bg-muted rounded">
                              <div className="flex items-center gap-2">
                                {getStatusIcon(result.status)}
                                <span className="text-sm">{result.collection}</span>
                                {result.status === 'success' && (
                                  <>
                                    <Badge variant="outline" className="text-xs">
                                      {result.document_count} {t("kb.dialog.fileUpload.processResult.createDocuments")}
                                    </Badge>
                                    <Badge variant="outline" className="text-xs">
                                      {result.chunks_count} {t("kb.dialog.fileUpload.processResult.textChunks")}
                                    </Badge>
                                  </>
                                )}
                              </div>
                              {result.status === 'error' && result.message && (
                                <p className="text-xs text-red-500 ml-6 break-all">{result.message}</p>
                              )}
                            </div>
                          ))}
                        </div>
                      </ScrollArea>
                    </div>
                  )}
                </div>
              </TabsContent>

              {/* Website Import Tab */}
              <TabsContent value="web" className="space-y-6">
                <div className="space-y-4 w-full">
                  <div className="flex items-center gap-2">
                    <Globe className="h-5 w-5 text-blue-500" />
                    <h3 className="text-lg font-medium">{t("kb.dialog.webImport.title")}</h3>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {t("kb.dialog.webImport.description")}
                  </p>

                  {/* Basic Configuration */}
                  <div className="space-y-4">
                    <h4 className="font-medium">{t("kb.dialog.webImport.basic.title")}</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <Label htmlFor="start_url">{t("kb.dialog.webImport.basic.startUrl")} *</Label>
                        <Input
                          id="start_url"
                          placeholder="https://help.example.com"
                          value={webIngestionConfig.start_url}
                          onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, start_url: e.target.value }))}
                        />
                      </div>
                      <div>
                        <Label htmlFor="max_pages">{t("kb.dialog.webImport.basic.maxPages")}</Label>
                        <Input
                          id="max_pages"
                          type="number"
                          value={webIngestionConfig.max_pages}
                          onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, max_pages: parseInt(e.target.value) || 100 }))}
                        />
                      </div>
                      <div>
                        <Label htmlFor="max_depth">{t("kb.dialog.webImport.basic.crawlDepth")}</Label>
                        <Input
                          id="max_depth"
                          type="number"
                          min="1"
                          max="10"
                          value={webIngestionConfig.max_depth}
                          onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, max_depth: parseInt(e.target.value) || 3 }))}
                        />
                      </div>
                      <div>
                        <Label htmlFor="concurrent_requests">{t("kb.dialog.webImport.basic.concurrentRequests")}</Label>
                        <Input
                          id="concurrent_requests"
                          type="number"
                          min="1"
                          max="10"
                          value={webIngestionConfig.concurrent_requests}
                          onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, concurrent_requests: parseInt(e.target.value) || 3 }))}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Advanced Configuration */}
                  <details className="space-y-4">
                    <summary className="cursor-pointer font-medium flex items-center gap-2">
                      <Settings size={16} />
                      {t("kb.dialog.webImport.advanced.title")}
                    </summary>
                    <div className="space-y-4 pt-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor="url_patterns">{t("kb.dialog.webImport.advanced.urlPatterns")}</Label>
                          <Input
                            id="url_patterns"
                            placeholder=".*help\\.example\\.com.*"
                            value={webIngestionConfig.url_patterns}
                            onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, url_patterns: e.target.value }))}
                          />
                          <p className="text-xs text-muted-foreground mt-1">{t("kb.dialog.webImport.advanced.hintMultiple")}</p>
                        </div>
                        <div>
                          <Label htmlFor="exclude_patterns">{t("kb.dialog.webImport.advanced.excludePatterns")}</Label>
                          <Input
                            id="exclude_patterns"
                            placeholder=".*\\.pdf$,.*\\.jpg$"
                            value={webIngestionConfig.exclude_patterns}
                            onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, exclude_patterns: e.target.value }))}
                          />
                          <p className="text-xs text-muted-foreground mt-1">{t("kb.dialog.webImport.advanced.hintMultiple")}</p>
                        </div>
                        <div>
                          <Label htmlFor="content_selector">{t("kb.dialog.webImport.advanced.contentSelector")}</Label>
                          <Input
                            id="content_selector"
                            placeholder="main article"
                            value={webIngestionConfig.content_selector}
                            onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, content_selector: e.target.value }))}
                          />
                          <p className="text-xs text-muted-foreground mt-1">{t("kb.dialog.webImport.advanced.hintContentSelector")}</p>
                        </div>
                        <div>
                          <Label htmlFor="remove_selectors">{t("kb.dialog.webImport.advanced.removeSelectors")}</Label>
                          <Input
                            id="remove_selectors"
                            placeholder="nav, footer, .sidebar"
                            value={webIngestionConfig.remove_selectors}
                            onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, remove_selectors: e.target.value }))}
                          />
                          <p className="text-xs text-muted-foreground mt-1">{t("kb.dialog.webImport.advanced.hintMultiple")}</p>
                        </div>
                        <div>
                          <Label htmlFor="request_delay">{t("kb.dialog.webImport.advanced.requestDelaySeconds")}</Label>
                          <Input
                            id="request_delay"
                            type="number"
                            step="0.1"
                            min="0"
                            value={webIngestionConfig.request_delay}
                            onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, request_delay: parseFloat(e.target.value) || 1.0 }))}
                          />
                        </div>
                        <div>
                          <Label htmlFor="timeout">{t("kb.dialog.webImport.advanced.timeoutSeconds")}</Label>
                          <Input
                            id="timeout"
                            type="number"
                            value={webIngestionConfig.timeout}
                            onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, timeout: parseInt(e.target.value) || 30 }))}
                          />
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          id="same_domain_only"
                          checked={webIngestionConfig.same_domain_only}
                          onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, same_domain_only: e.target.checked }))}
                          className="w-4 h-4"
                        />
                        <Label htmlFor="same_domain_only" className="cursor-pointer">{t("kb.dialog.webImport.advanced.sameDomainOnly")}</Label>
                      </div>
                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          id="respect_robots_txt"
                          checked={webIngestionConfig.respect_robots_txt}
                          onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, respect_robots_txt: e.target.checked }))}
                          className="w-4 h-4"
                        />
                        <Label htmlFor="respect_robots_txt" className="cursor-pointer">{t("kb.dialog.webImport.advanced.respectRobotsTxt")}</Label>
                      </div>
                    </div>
                  </details>

                  {/* Crawl Progress */}
                  {isWebIngesting && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>{t("kb.dialog.webImport.status.progressTitle")}</span>
                        <span>{Math.round(webIngestionProgress)}%</span>
                      </div>
                      <Progress value={webIngestionProgress} className="w-full" />
                      <p className="text-xs text-muted-foreground">{t("kb.dialog.webImport.status.crawling")}</p>
                    </div>
                  )}

                  {/* Crawl Result */}
                  {webIngestionResult && (
                    <Card className="p-4">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          {getStatusIcon(webIngestionResult.status)}
                          <span className="font-medium">{t(webIngestionResult.status === "success" ? "kb.dialog.webImport.status.success" : "kb.dialog.webImport.status.done")}</span>
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
                  )}
                </div>
              </TabsContent>

              {/* Cloud Connect Tab */}
              <TabsContent value="cloud" className="space-y-6">
                <div className="space-y-4 w-full">
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

                  {/* Selected Cloud Files List */}
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
              </TabsContent>
            </Tabs>

            {/* Index Configuration */}
            <div>
              <h3 className="text-lg font-medium">{t("kb.index.title")}</h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="parse_method">{t("kb.index.parseMethod")}</Label>
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
                  />
                </div>

                <div>
                  <Label htmlFor="chunk_strategy">{t("kb.index.chunkStrategy")}</Label>
                  <Select
                    value={ingestionConfig.chunk_strategy}
                    onValueChange={(value) => setIngestionConfig(prev => ({ ...prev, chunk_strategy: value }))}
                    options={[
                      { value: "recursive", label: t("kb.index.chunkOptions.recursive") },
                      { value: "fixed_size", label: t("kb.index.chunkOptions.fixed_size") },
                      { value: "markdown", label: t("kb.index.chunkOptions.markdown") },
                    ]}
                  />
                </div>

                <div>
                  <Label htmlFor="chunk_size">{t("kb.index.chunkSize")}</Label>
                  <Input
                    id="chunk_size"
                    type="number"
                    value={ingestionConfig.chunk_size}
                    onChange={(e) => setIngestionConfig(prev => ({ ...prev, chunk_size: parseInt(e.target.value) || 1000 }))}
                  />
                </div>

                <div>
                  <Label htmlFor="chunk_overlap">{t("kb.index.chunkOverlap")}</Label>
                  <Input
                    id="chunk_overlap"
                    type="number"
                    value={ingestionConfig.chunk_overlap}
                    onChange={(e) => setIngestionConfig(prev => ({ ...prev, chunk_overlap: parseInt(e.target.value) || 200 }))}
                  />
                </div>

                {ingestionConfig.chunk_strategy === "recursive" && (
                  <div>
                    <Label htmlFor="separators" title={t("kb.index.separatorsTip")}>
                      {t("kb.index.separators")}
                    </Label>
                    <Input
                      id="separators"
                      type="text"
                      value={ingestionConfig.separators ?? ""}
                      onChange={(e) => setIngestionConfig(prev => ({ ...prev, separators: e.target.value }))}
                      placeholder={t("kb.index.separatorsPlaceholder")}
                    />
                  </div>
                )}

                <div>
                  <Label htmlFor="embedding_model_id">{t("kb.index.embeddingModelId")}</Label>
                  <Select
                    value={ingestionConfig.embedding_model_id}
                    onValueChange={(value: string) => setIngestionConfig(prev => ({ ...prev, embedding_model_id: value }))}
                    options={embeddingModels.map(model => ({ value: model.model_id, label: model.name || model.model_id }))}
                  />
                </div>

                <div>
                  <Label htmlFor="embedding_batch_size">{t("kb.index.embeddingBatchSize")}</Label>
                  <Input
                    id="embedding_batch_size"
                    type="number"
                    value={ingestionConfig.embedding_batch_size}
                    onChange={(e) => setIngestionConfig(prev => ({ ...prev, embedding_batch_size: parseInt(e.target.value) || 10 }))}
                  />
                </div>
              </div>
            </div>
          </div>
          {/* Action Buttons */}
          <div className="p-6 pt-4 flex justify-end border-t gap-2">
            <Button variant="outline" onClick={() => {
              resetState()
              onOpenChange(false)
            }}>
              {t("common.cancel")}
            </Button>
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
                (activeImportTab === "file" && (selectedFiles.length === 0 || (selectedFiles.length > 1 && !trimmedCollectionName))) ||
                (activeImportTab === "web" && !webIngestionConfig.start_url) ||
                (activeImportTab === "cloud" && (totalCloudFiles === 0 || (totalCloudFiles > 1 && !trimmedCollectionName))) ||
                isUploading ||
                isWebIngesting ||
                isCloudConnecting
              }
            >
              {isUploading || isWebIngesting || isCloudConnecting
                ? activeImportTab === "cloud"
                  ? t("kb.dialog.cloudConnect.connecting")
                  : t("kb.dialog.fileUpload.processing")
                : t("kb.index.startImport")}
            </Button>
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
