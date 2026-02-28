
import { useState, useEffect, useRef } from "react"
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
import { SelectRadix, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { getApiUrl } from "@/lib/utils"
import { useAuth } from "@/contexts/auth-context"
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
  FileText
} from "lucide-react"
import { toast } from "sonner"

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
  onSuccess?: () => void
}

export function KnowledgeBaseCreationDialog({ open, onOpenChange, onSuccess }: KnowledgeBaseCreationDialogProps) {
  const { t } = useI18n()

  // State from KnowledgeBasePage
  const [newCollectionName, setNewCollectionName] = useState("")
  const [newCollectionDescription, setNewCollectionDescription] = useState("")
  const [activeImportTab, setActiveImportTab] = useState<"file" | "web">("file")

  // File upload state
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [ingestionResults, setIngestionResults] = useState<IngestionResult[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

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

  // Ingestion config state
  const [ingestionConfig, setIngestionConfig] = useState({
    parse_method: "default",
    chunk_strategy: "recursive",
    chunk_size: 1000,
    chunk_overlap: 200,
    embedding_model_id: "",
    embedding_batch_size: 10,
    max_retries: 3,
    retry_delay: 1.0
  })

  // Embedding models state
  const [embeddingModels, setEmbeddingModels] = useState<Model[]>([])

  useEffect(() => {
    if (open) {
      fetchEmbeddingModels()
    }
  }, [open])

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
        toast.error(t("kb.errors.unsupportedFileType") || "部分文件格式不支持，已跳过")
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

    setIsUploading(true)
    setUploadProgress(0)
    setIngestionResults([])

    try {
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i]
        const formData = new FormData()

        const collectionName = newCollectionName || file.name.replace(/\.[^/.]+$/, "")

        formData.append("file", file)
        formData.append("collection", collectionName)
        formData.append("parse_method", ingestionConfig.parse_method)
        formData.append("chunk_strategy", ingestionConfig.chunk_strategy)
        formData.append("chunk_size", ingestionConfig.chunk_size.toString())
        formData.append("chunk_overlap", ingestionConfig.chunk_overlap.toString())
        formData.append("embedding_model_id", ingestionConfig.embedding_model_id)
        formData.append("embedding_batch_size", ingestionConfig.embedding_batch_size.toString())
        formData.append("max_retries", ingestionConfig.max_retries.toString())
        formData.append("retry_delay", ingestionConfig.retry_delay.toString())

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
          throw new Error(result.message || `Failed at step: ${result.failed_step}`)
        }

        setUploadProgress(((i + 1) / selectedFiles.length) * 100)
      }

      resetState()
      onOpenChange(false)
      onSuccess?.()

    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("kb.errors.uploadFailed"))
    } finally {
      setIsUploading(false)
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

      const collectionName = newCollectionName || "web_collection"

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

      formData.append("parse_method", ingestionConfig.parse_method)
      formData.append("chunk_strategy", ingestionConfig.chunk_strategy)
      formData.append("chunk_size", ingestionConfig.chunk_size.toString())
      formData.append("chunk_overlap", ingestionConfig.chunk_overlap.toString())
      formData.append("embedding_model_id", ingestionConfig.embedding_model_id)
      formData.append("embedding_batch_size", ingestionConfig.embedding_batch_size.toString())
      formData.append("max_retries", ingestionConfig.max_retries.toString())
      formData.append("retry_delay", ingestionConfig.retry_delay.toString())

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
      onSuccess?.()

    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("kb.errors.webIngestFailed"))
    } finally {
      setIsWebIngesting(false)
      setWebIngestionProgress(0)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{t("kb.dialog.createTitle")}</DialogTitle>
          <DialogDescription>
            {t("kb.dialog.createDescription")}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-6">
          {/* 基本信息 */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium">{t("kb.dialog.basicInfo.title")}</h3>
            <div>
              <Label htmlFor="collection_name">{t("kb.dialog.basicInfo.nameLabel")}</Label>
              <Input
                id="collection_name"
                value={newCollectionName}
                onChange={(e) => setNewCollectionName(e.target.value)}
                placeholder={t("kb.dialog.basicInfo.namePlaceholder")}
              />
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

          {/* Tabs: 文件上传 / 网站导入 */}
          <Tabs value={activeImportTab} onValueChange={(v) => setActiveImportTab(v as "file" | "web")} className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="file">
                <FileText size={16} className="mr-2" />
                {t("kb.dialog.tabs.file")}
              </TabsTrigger>
              <TabsTrigger value="web">
                <Globe size={16} className="mr-2" />
                {t("kb.dialog.tabs.web")}
              </TabsTrigger>
            </TabsList>

            {/* 文件上传 Tab */}
            <TabsContent value="file" className="space-y-4 w-full">
              {/* 文件上传 */}
              <div className="space-y-4 w-full">
                <h3 className="text-lg font-medium">{t("kb.dialog.fileUpload.title")}</h3>

                {/* 文件选择区域 */}
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

                {/* 已选择文件列表 */}
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

                {/* 上传进度 */}
                {isUploading && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>{t("kb.dialog.fileUpload.progressTitle")}</span>
                      <span>{Math.round(uploadProgress)}%</span>
                    </div>
                    <Progress value={uploadProgress} className="w-full" />
                  </div>
                )}

                {/* 上传结果 */}
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

            {/* 网站导入 Tab */}
            <TabsContent value="web" className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Globe className="h-5 w-5 text-blue-500" />
                  <h3 className="text-lg font-medium">{t("kb.dialog.webImport.title")}</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  {t("kb.dialog.webImport.description")}
                </p>

                {/* 基础配置 */}
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

                {/* 高级配置 */}
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

                {/* 爬取进度 */}
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

                {/* 爬取结果 */}
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
          </Tabs>

          {/* 索引配置 */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium">{t("kb.index.title")}</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="parse_method">{t("kb.index.parseMethod")}</Label>
                <SelectRadix
                  value={ingestionConfig.parse_method}
                  onValueChange={(value) => setIngestionConfig(prev => ({ ...prev, parse_method: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="default">{t("kb.index.parseOptions.default")}</SelectItem>
                    <SelectItem value="pypdf">{t("kb.index.parseOptions.pypdf")}</SelectItem>
                    <SelectItem value="pdfplumber">{t("kb.index.parseOptions.pdfplumber")}</SelectItem>
                    <SelectItem value="unstructured">{t("kb.index.parseOptions.unstructured")}</SelectItem>
                    <SelectItem value="pymupdf">{t("kb.index.parseOptions.pymupdf")}</SelectItem>
                    <SelectItem value="deepdoc">{t("kb.index.parseOptions.deepdoc")}</SelectItem>
                  </SelectContent>
                </SelectRadix>
              </div>

              <div>
                <Label htmlFor="chunk_strategy">{t("kb.index.chunkStrategy")}</Label>
                <SelectRadix
                  value={ingestionConfig.chunk_strategy}
                  onValueChange={(value) => setIngestionConfig(prev => ({ ...prev, chunk_strategy: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="recursive">{t("kb.index.chunkOptions.recursive")}</SelectItem>
                    <SelectItem value="fixed_size">{t("kb.index.chunkOptions.fixed_size")}</SelectItem>
                    <SelectItem value="markdown">{t("kb.index.chunkOptions.markdown")}</SelectItem>
                  </SelectContent>
                </SelectRadix>
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

              <div>
                <Label htmlFor="embedding_model_id">{t("kb.index.embeddingModelId")}</Label>
                <SelectRadix value={ingestionConfig.embedding_model_id} onValueChange={(value) => setIngestionConfig(prev => ({ ...prev, embedding_model_id: value }))}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {embeddingModels.map((model) => (
                      <SelectItem key={model.id} value={model.model_id}>
                        {model.name || model.model_id}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </SelectRadix>
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

          {/* 操作按钮 */}
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => {
              resetState()
              onOpenChange(false)
            }}>
              {t("common.cancel")}
            </Button>
            <Button
              onClick={() => {
                if (activeImportTab === 'web') {
                  handleWebIngest()
                } else {
                  handleUpload()
                }
              }}
              disabled={
                (activeImportTab === 'file' && selectedFiles.length === 0) ||
                (activeImportTab === 'web' && !webIngestionConfig.start_url) ||
                isUploading ||
                isWebIngesting
              }
            >
              {(isUploading || isWebIngesting) ? t("kb.dialog.fileUpload.processing") : t("kb.index.startImport")}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
