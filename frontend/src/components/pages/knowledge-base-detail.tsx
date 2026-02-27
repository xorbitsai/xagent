"use client"

import { useState, useEffect, useRef } from "react"
import * as TabsPrimitive from "@radix-ui/react-tabs"
import { ArrowLeft, HardDrive, Search, Upload, Plus, Trash2, FileIcon, CheckCircle, XCircle, AlertCircle, Globe, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card } from "@/components/ui/card"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { SelectRadix, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "sonner"

interface CollectionInfo {
  name: string
  documents: number
  chunks: number
  embeddings: number
  parses: number
  document_names?: string[]
}

interface IngestionConfig {
  parse_method: string
  chunk_strategy: string
  chunk_size: number
  chunk_overlap: number
  embedding_model_id: string
  embedding_batch_size: number
  max_retries: number
  retry_delay: number
}

interface SearchResult {
  score: number
  text: string
  document: string
  metadata?: any
}

interface SearchConfig {
  search_type: string
  top_k: number
  embedding_model_id: string
  rerank_model_id: string
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

export function KnowledgeBaseDetailContent({ collectionName }: { collectionName: string }) {
  const { t } = useI18n()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [collectionInfo, setCollectionInfo] = useState<CollectionInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState("files")

  // Edit dialog states
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false)
  const [editCollectionName, setEditCollectionName] = useState("")
  const [isUpdating, setIsUpdating] = useState(false)

  // File upload states
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [ingestionResults, setIngestionResults] = useState<any[]>([])
  const [newCollectionName, setNewCollectionName] = useState("")
  const [newCollectionDescription, setNewCollectionDescription] = useState("")
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
  const [isAddSourceOpen, setIsAddSourceOpen] = useState(false)
  const [activeAddSourceMode, setActiveAddSourceMode] = useState<"web" | "file" | null>(null)
  const [isDragging, setIsDragging] = useState(false)

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
        toast.error(t("kb.errors.unsupportedFileType") || "Unsupported file type")
      }

      if (validFiles.length > 0) {
        setSelectedFiles(prev => [...prev, ...validFiles])
        setActiveAddSourceMode("file")
      }
    }
  }


  // Web ingestion states
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
  const [uploadMethod, setUploadMethod] = useState<"file" | "web">("file")

  // Embedding models state
  const [embeddingModels, setEmbeddingModels] = useState<any[]>([])
  const [defaultEmbeddingModel, setDefaultEmbeddingModel] = useState<string | null>(null)

  // Ingestion configuration
  const [ingestionConfig, setIngestionConfig] = useState<IngestionConfig>({
    parse_method: "pypdf",
    chunk_strategy: "recursive",
    chunk_size: 1000,
    chunk_overlap: 200,
    embedding_model_id: "",
    embedding_batch_size: 10,
    max_retries: 3,
    retry_delay: 1.0,
  })

  // Search states
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [searchConfig, setSearchConfig] = useState<SearchConfig>({
    search_type: "hybrid",
    top_k: 5,
    embedding_model_id: "",
    rerank_model_id: "",
  })

  useEffect(() => {
    fetchCollectionInfo()
    fetchEmbeddingModels()
  }, [collectionName])

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
          setDefaultEmbeddingModel(defaultModelId)
          // Update configs to use default model
          setIngestionConfig(prev => ({ ...prev, embedding_model_id: defaultModelId }))
          setSearchConfig(prev => ({ ...prev, embedding_model_id: defaultModelId }))
        } else if (models.length > 0) {
          // Fallback to first model if no default set
          const firstModelId = models[0].model_id
          setDefaultEmbeddingModel(firstModelId)
          setIngestionConfig(prev => ({ ...prev, embedding_model_id: firstModelId }))
          setSearchConfig(prev => ({ ...prev, embedding_model_id: firstModelId }))
        }
      } else if (models.length > 0) {
        // Fallback to first model
        const firstModelId = models[0].model_id
        setDefaultEmbeddingModel(firstModelId)
        setIngestionConfig(prev => ({ ...prev, embedding_model_id: firstModelId }))
        setSearchConfig(prev => ({ ...prev, embedding_model_id: firstModelId }))
      }
    } catch (err) {
      console.error("Failed to fetch embedding models:", err)
    }
  }

  const fetchCollectionInfo = async () => {
    try {
      setLoading(true)
      const response = await apiRequest(`${getApiUrl()}/api/kb/collections`)

      if (!response.ok) {
        throw new Error("Failed to fetch collection info")
      }

      const data = await response.json()
      const collection = data.collections?.find((c: CollectionInfo) => c.name === collectionName)

      if (!collection) {
        throw new Error("Collection not found")
      }

      setCollectionInfo(collection)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    setSelectedFiles(prev => [...prev, ...files])
    setActiveAddSourceMode("file")
  }

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const handleDeleteDocument = async (fileName: string) => {
    if (!confirm(t("kb.detail.uploaded.confirmDelete") || "Are you sure you want to delete this document?")) {
      return
    }

    try {
      console.log("Deleting document:", fileName)
      const response = await apiRequest(
        `${getApiUrl()}/api/kb/collections/${encodeURIComponent(collectionName)}/documents/${encodeURIComponent(fileName)}`,
        { method: "DELETE" }
      )

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || t("kb.detail.errors.deleteFailed"))
      }

      const result = await response.json()
      console.log("Delete result:", result)

      // Handle partial success or failure
      if (result.status === "partial_success") {
        toast.error(t("kb.detail.errors.partialSuccess", { message: result.message }))
      } else if (result.status === "failed") {
        toast.error(t("kb.detail.errors.deleteFailedWithMessage", { message: result.message }))
        return
      }

      // Add brief delay to ensure backend data is updated
      await new Promise(resolve => setTimeout(resolve, 500))

      // Reload collection info
      await fetchCollectionInfo()
      console.log("Collection info refreshed")
    } catch (error) {
      console.error("Delete error:", error)
      toast.error(error instanceof Error ? error.message : t("kb.detail.errors.deleteFailed"))
    }
  }

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      toast.error(t("kb.detail.errors.pleaseSelectFiles"))
      return
    }

    setIsUploading(true)
    setUploadProgress(0)
    setIngestionResults([])

    try {
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i]
        const formData = new FormData()

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
          throw new Error(errorData.detail || t("kb.detail.errors.uploadFailedWithName", { name: file.name }))
        }

        const result = await response.json()
        setIngestionResults(prev => [...prev, result])
        setUploadProgress(((i + 1) / selectedFiles.length) * 100)
      }

      await fetchCollectionInfo()
      setSelectedFiles([])
      setUploadProgress(0)
      setIsAddSourceOpen(false)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("kb.detail.errors.uploadFailedGeneric"))
    } finally {
      setIsUploading(false)
    }
  }

  const handleWebIngest = async () => {
    if (!webIngestionConfig.start_url.trim()) {
      toast.error(t("kb.detail.errors.enterStartUrl"))
      return
    }

    setIsWebIngesting(true)
    setWebIngestionProgress(0)
    setWebIngestionResult(null)

    try {
      const formData = new FormData()

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

      // 添加索引配置
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
        throw new Error(errorData.detail || t("kb.detail.errors.webImportFailed"))
      }

      const result: WebIngestionResult = await response.json()
      setWebIngestionResult(result)
      setWebIngestionProgress(100)

      // 导入成功后刷新信息
      await fetchCollectionInfo()

      // 关闭对话框
      setIsAddSourceOpen(false)

      // 重置配置
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

    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("kb.detail.errors.webImportFailed"))
    } finally {
      setIsWebIngesting(false)
      setWebIngestionProgress(0)
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setSearching(true)
    try {
      const formData = new FormData()
      formData.append("collection", collectionName)
      formData.append("query_text", searchQuery)
      formData.append("search_type", searchConfig.search_type)
      formData.append("top_k", searchConfig.top_k.toString())
      formData.append("embedding_model_id", searchConfig.embedding_model_id)

      if (searchConfig.rerank_model_id) {
        formData.append("rerank_model_id", searchConfig.rerank_model_id)
      }

      const response = await apiRequest(`${getApiUrl()}/api/kb/search`, {
        method: "POST",
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || t("kb.detail.errors.searchFailed"))
      }

      const result = await response.json()
      setSearchResults(result.results || [])
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("kb.detail.errors.searchFailed"))
    } finally {
      setSearching(false)
    }
  }

  const handleOpenEditDialog = () => {
    setEditCollectionName(collectionName)
    setIsEditDialogOpen(true)
  }

  const handleUpdateCollectionName = async () => {
    if (!editCollectionName.trim() || editCollectionName === collectionName) {
      setIsEditDialogOpen(false)
      return
    }

    setIsUpdating(true)
    try {
      const formData = new FormData()
      formData.append("new_name", editCollectionName)

      const response = await apiRequest(`${getApiUrl()}/api/kb/collections/${encodeURIComponent(collectionName)}`, {
        method: "PUT",
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || t("kb.detail.edit.errors.renameFailed"))
      }

      const result = await response.json()

      // 重命名成功后跳转到新的 URL
      window.location.href = `/kb/${encodeURIComponent(editCollectionName)}`
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("kb.detail.edit.errors.updateFailed"))
    } finally {
      setIsUpdating(false)
    }
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <HardDrive className="h-12 w-12 mx-auto mb-4 animate-spin text-muted-foreground" />
          <p>{t("kb.detail.loadingDetail")}</p>
        </div>
      </div>
    )
  }

  if (!collectionInfo) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
          <p className="text-lg mb-2">{t("kb.detail.notFound")}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col space-y-6">
        {/* Main Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="lex-1 w-full">
          <TabsList className="flex w-full justify-start rounded-none border-b bg-transparent px-6 mb-6">
            <TabsPrimitive.Trigger
              value="files"
              className="flex-none relative h-10 px-4 pb-3 pt-2 font-semibold text-muted-foreground hover:text-foreground border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:text-primary transition-colors outline-none ring-0 focus-visible:ring-0"
            >
              {t("kb.detail.tabs.files")}
            </TabsPrimitive.Trigger>
            <TabsPrimitive.Trigger
              value="search"
              className="flex-none relative h-10 px-4 pb-3 pt-2 font-semibold text-muted-foreground hover:text-foreground border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:text-primary transition-colors outline-none ring-0 focus-visible:ring-0"
            >
              {t("kb.detail.tabs.search")}
            </TabsPrimitive.Trigger>
            <TabsPrimitive.Trigger
              value="settings"
              className="flex-none relative h-10 px-4 pb-3 pt-2 font-semibold text-muted-foreground hover:text-foreground border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:text-primary transition-colors outline-none ring-0 focus-visible:ring-0"
            >
              {t("kb.detail.tabs.settings")}
            </TabsPrimitive.Trigger>
          </TabsList>

          {/* Files Management Tab */}
          <TabsContent value="files" className="space-y-6 w-full">
            {/* Uploaded Files Section */}
            <div className="p-6 w-full">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">{t("kb.detail.files.title")}</h3>
                <Button onClick={() => setIsAddSourceOpen(true)} size="sm">
                  <Plus size={16} className="mr-2" />
                  {t("kb.detail.files.addSource")}
                </Button>
              </div>

              {collectionInfo.document_names && collectionInfo.document_names.length > 0 ? (
                <ScrollArea className="h-96">
                  <div className="space-y-3">
                    {collectionInfo.document_names.map((fileName, index) => (
                      <div key={index} className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50 transition-colors">
                        <div className="flex items-center gap-3">
                          <FileIcon className="h-5 w-5 text-blue-500" />
                          <div className="flex-1">
                            <p className="font-medium text-sm">{fileName.split('/').pop() || fileName}</p>
                            <p className="text-xs text-muted-foreground">{fileName}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant="secondary" className="text-xs">
                            {t("kb.detail.uploaded.indexed")}
                          </Badge>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDeleteDocument(fileName)}
                            title={t("kb.detail.uploaded.delete") || "Delete document"}
                          >
                            <Trash2 size={14} />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
                  <FileIcon className="h-16 w-16 mb-4 opacity-30" />
                  <p className="text-lg font-medium mb-2">{t("kb.detail.uploaded.emptyTitle")}</p>
                  <p className="text-sm text-center">{t("kb.detail.uploaded.emptyHint")}</p>
                </div>
              )}
            </div>

            {/* Hidden Input for File Selection */}
            <input
                type="file"
                multiple
                ref={fileInputRef}
                accept=".pdf,.txt,.html,.htm,.md,.doc,.docx,.xlsx,.ppt,.pptx,.csv"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload-detail"
            />

            {/* Upload Results Section - Full Width Below Columns */}
            {ingestionResults.length > 0 && (
              <Card className="p-6 w-full">
                <h3 className="text-lg font-semibold mb-4">{t("kb.detail.process.title")}</h3>
                <ScrollArea className="h-96">
                  <div className="space-y-4">
                    {ingestionResults.map((result, index) => (
                      <div key={index} className="p-4 border rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          {result.status === "success" ? (
                            <CheckCircle className="h-4 w-4 text-green-500" />
                          ) : (
                            <XCircle className="h-4 w-4 text-red-500" />
                          )}
                          <span className="font-medium">{result.file_name || `${t("kb.detail.process.labels.file")} ${index + 1}`}</span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
                          <div>{t("kb.detail.process.labels.document")}: {result.documents_processed || 0}</div>
                          <div>{t("kb.detail.process.labels.chunk")}: {result.chunks_created || 0}</div>
                          <div>{t("kb.detail.process.labels.parse")}: {result.parses_completed || 0}</div>
                          <div>{t("kb.detail.process.labels.vector")}: {result.embeddings_created || 0}</div>
                        </div>
                        {result.error && (
                          <div className="mt-2 text-sm text-red-600">
                            {t("kb.detail.process.labels.error")}: {result.error}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </Card>
            )}
          </TabsContent>

          {/* Search Test Tab */}
          <TabsContent value="search" className="space-y-6 w-full flex-1">
            <div className="p-6 w-full">
              <h3 className="text-lg font-semibold mb-4">{t("kb.detail.search.title")}</h3>

              {/* Search Configuration */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <Label htmlFor="search_type">{t("kb.detail.search.typeLabel")}</Label>
                  <SelectRadix value={searchConfig.search_type} onValueChange={(value) => setSearchConfig(prev => ({ ...prev, search_type: value }))}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="hybrid">{t("kb.detail.search.types.hybrid")}</SelectItem>
                      <SelectItem value="dense">{t("kb.detail.search.types.dense")}</SelectItem>
                      <SelectItem value="sparse">{t("kb.detail.search.types.sparse")}</SelectItem>
                    </SelectContent>
                  </SelectRadix>
                </div>
                <div>
                  <Label htmlFor="top_k">{t("kb.detail.search.topKLabel")}</Label>
                  <Input
                    id="top_k"
                    type="number"
                    value={searchConfig.top_k}
                    onChange={(e) => setSearchConfig(prev => ({ ...prev, top_k: parseInt(e.target.value) || 5 }))}
                  />
                </div>
                <div>
                  <Label htmlFor="embedding_model_id">{t("kb.detail.search.embeddingModelIdLabel")}</Label>
                  <SelectRadix value={searchConfig.embedding_model_id} onValueChange={(value) => setSearchConfig(prev => ({ ...prev, embedding_model_id: value }))}>
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
                  <Label htmlFor="rerank_model_id">{t("kb.detail.search.rerankModelIdLabel")}</Label>
                  <Input
                    id="rerank_model_id"
                    value={searchConfig.rerank_model_id}
                    onChange={(e) => setSearchConfig(prev => ({ ...prev, rerank_model_id: e.target.value }))}
                    placeholder={t("kb.detail.search.rerankPlaceholder")}
                  />
                </div>
              </div>

              {/* Search Input */}
              <div className="flex gap-2 mb-6">
                <div className="flex-1">
                  <Input
                    placeholder={t("kb.detail.search.queryPlaceholder")}
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  />
                </div>
                <Button
                  onClick={handleSearch}
                  disabled={!searchQuery.trim() || searching}
                >
                  {searching ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  ) : (
                    <>
                      <Search size={16} className="mr-2" />
                      {t("kb.detail.search.searchButton")}
                    </>
                  )}
                </Button>
              </div>

              {/* Search Results */}
              {searchResults.length > 0 && (
                <div>
                  <h4 className="font-medium mb-4">{t("kb.detail.search.resultsTitle", { count: searchResults.length })}</h4>
                  <div className="space-y-4">
                    {searchResults.map((result, index) => (
                      <Card key={index} className="p-4">
                        <div className="flex justify-between items-start mb-2">
                          <div className="flex items-center gap-2">
                            <Badge variant="outline">
                              {t("kb.detail.search.score")}: {result.score.toFixed(3)}
                            </Badge>
                            <Badge variant="secondary">
                              {result.document}
                            </Badge>
                          </div>
                        </div>
                        <p className="text-sm leading-relaxed">{result.text}</p>
                        {result.metadata && (
                          <div className="mt-2 text-xs text-muted-foreground">
                            <details>
                              <summary className="cursor-pointer">{t("kb.detail.search.metadata")}</summary>
                              <pre className="mt-1 p-2 bg-muted rounded text-xs overflow-x-auto">
                                {JSON.stringify(result.metadata, null, 2)}
                              </pre>
                            </details>
                          </div>
                        )}
                      </Card>
                    ))}
                  </div>
                </div>
              )}

              {searchResults.length === 0 && searchQuery && !searching && (
                <div className="text-center py-8">
                  <Search className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
                  <p className="text-muted-foreground">{t("kb.detail.search.noResults")}</p>
                </div>
              )}
            </div>
          </TabsContent>

          {/* Index Settings Tab */}
          <TabsContent value="settings" className="space-y-6 w-full">
            <div className="p-6 w-full">
              <h3 className="text-lg font-semibold mb-4">{t("kb.index.title")}</h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="parse_method">{t("kb.index.parseMethod")}</Label>
                  <SelectRadix value={ingestionConfig.parse_method} onValueChange={(value) => setIngestionConfig(prev => ({ ...prev, parse_method: value }))}>
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
                  <SelectRadix value={ingestionConfig.chunk_strategy} onValueChange={(value) => setIngestionConfig(prev => ({ ...prev, chunk_strategy: value }))}>
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
                  <Label htmlFor="embedding_model_id_settings">{t("kb.index.embeddingModelId")}</Label>
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
                  <Label htmlFor="embedding_batch_size_settings">{t("kb.index.embeddingBatchSize")}</Label>
                  <Input
                    id="embedding_batch_size_settings"
                    type="number"
                    value={ingestionConfig.embedding_batch_size}
                    onChange={(e) => setIngestionConfig(prev => ({ ...prev, embedding_batch_size: parseInt(e.target.value) || 10 }))}
                  />
                </div>
              </div>

              <div className="mt-6">
                <Button>{t("kb.index.saveConfig")}</Button>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        {/* Edit Collection Dialog */}
        <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>{t("kb.detail.edit.title")}</DialogTitle>
              <DialogDescription>
                {t("kb.detail.edit.description")}
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4 py-4">
              <div>
                <Label htmlFor="edit-collection-name">{t("kb.detail.edit.nameLabel")}</Label>
                <Input
                  id="edit-collection-name"
                  value={editCollectionName}
                  onChange={(e) => setEditCollectionName(e.target.value)}
                  placeholder={t("kb.detail.edit.namePlaceholder")}
                />
              </div>
            </div>

            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setIsEditDialogOpen(false)}>
                {t("common.cancel")}
              </Button>
              <Button onClick={handleUpdateCollectionName} disabled={isUpdating}>
                {isUpdating ? t("kb.detail.edit.updating") : t("common.save")}
              </Button>
            </div>
          </DialogContent>
        </Dialog>

        {/* Add Source Dialog */}
        <Dialog open={isAddSourceOpen} onOpenChange={(open) => {
          setIsAddSourceOpen(open)
          if (!open) setActiveAddSourceMode(null)
        }}>
          <DialogContent className="sm:max-w-[600px]">
            {!activeAddSourceMode ? (
              <>
                <DialogHeader>
                  <DialogTitle>{t("kb.detail.files.addDialogTitle")}</DialogTitle>
                  <DialogDescription>
                    {t("kb.detail.files.addDialogDescription")}
                  </DialogDescription>
                </DialogHeader>
                <div className="grid grid-cols-2 gap-4 py-4">
                  <Button
                    variant="outline"
                    className={`h-32 flex flex-col gap-3 hover:bg-muted/50 hover:border-primary transition-all ${
                      isDragging ? "border-primary bg-primary/10" : ""
                    }`}
                    onClick={() => {
                      fileInputRef.current?.click()
                    }}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                  >
                    <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-full">
                      <Upload size={24} className="text-blue-600 dark:text-blue-400" />
                    </div>
                    <span className="font-medium">{t("kb.dialog.tabs.file")}</span>
                  </Button>
                  <Button
                    variant="outline"
                    className="h-32 flex flex-col gap-3 hover:bg-muted/50 hover:border-primary transition-all"
                    onClick={() => setActiveAddSourceMode("web")}
                  >
                    <div className="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-full">
                      <Globe size={24} className="text-purple-600 dark:text-purple-400" />
                    </div>
                    <span className="font-medium">{t("kb.dialog.tabs.web")}</span>
                  </Button>
                </div>
              </>
            ) : activeAddSourceMode === "file" ? (
              <div className="space-y-4">
                 <div className="flex items-center gap-2 mb-2">
                    <Button variant="ghost" size="icon" className="h-8 w-8 -ml-2" onClick={() => setActiveAddSourceMode(null)}>
                      <ArrowLeft size={16} />
                    </Button>
                    <DialogTitle>{t("kb.dialog.fileUpload.title")}</DialogTitle>
                 </div>

                 {selectedFiles.length === 0 ? (
                    <div
                      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:bg-muted/50 transition-colors ${
                        isDragging ? "border-primary bg-primary/10" : ""
                      }`}
                      onClick={() => fileInputRef.current?.click()}
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                    >
                      <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">{t("kb.dialog.fileUpload.dropOrClick")}</p>
                      <p className="text-xs text-muted-foreground mt-2">{t("kb.dialog.fileUpload.supportedFormats")}</p>
                    </div>
                 ) : (
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <h4 className="font-medium text-sm">{t("kb.dialog.fileUpload.selectedTitle")} ({selectedFiles.length})</h4>
                        <Button variant="ghost" size="sm" onClick={() => fileInputRef.current?.click()}>
                          <Plus size={14} className="mr-1" />
                          {t("kb.dialog.fileUpload.selectFiles")}
                        </Button>
                      </div>

                      <ScrollArea className="h-48 border rounded-md p-2">
                        <div className="space-y-2">
                          {selectedFiles.map((file, index) => (
                            <div key={index} className="flex items-center justify-between p-2 bg-muted/30 rounded text-sm">
                              <span className="truncate max-w-[200px]" title={file.name}>{file.name}</span>
                              <div className="flex items-center gap-2">
                                <span className="text-xs text-muted-foreground">{(file.size / 1024).toFixed(1)} KB</span>
                                <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => removeFile(index)}>
                                  <XCircle size={14} />
                                </Button>
                              </div>
                            </div>
                          ))}
                        </div>
                      </ScrollArea>

                      <Button
                        onClick={handleUpload}
                        disabled={isUploading}
                        className="w-full"
                      >
                        {isUploading ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            {t("kb.detail.files.uploading")} ({Math.round(uploadProgress)}%)
                          </>
                        ) : (
                          t("kb.detail.files.upload")
                        )}
                      </Button>
                    </div>
                 )}
              </div>
            ) : (
              <div className="space-y-4">
                 <div className="flex items-center gap-2 mb-2">
                    <Button variant="ghost" size="icon" className="h-8 w-8 -ml-2" onClick={() => setActiveAddSourceMode(null)}>
                      <ArrowLeft size={16} />
                    </Button>
                    <DialogTitle>{t("kb.dialog.tabs.web")}</DialogTitle>
                 </div>

                 <div className="space-y-4">
                    <div>
                      <Label htmlFor="dialog-start-url">{t("kb.dialog.webImport.basic.startUrl")} *</Label>
                      <Input
                        id="dialog-start-url"
                        placeholder="https://help.example.com"
                        value={webIngestionConfig.start_url}
                        onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, start_url: e.target.value }))}
                        className="mt-1"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor="dialog-max-pages">{t("kb.dialog.webImport.basic.maxPages")}</Label>
                          <Input
                            id="dialog-max-pages"
                            type="number"
                            value={webIngestionConfig.max_pages}
                            onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, max_pages: parseInt(e.target.value) || 100 }))}
                            className="mt-1"
                          />
                        </div>
                        <div>
                          <Label htmlFor="dialog-max-depth">{t("kb.dialog.webImport.basic.crawlDepth")}</Label>
                          <Input
                            id="dialog-max-depth"
                            type="number"
                            min="1"
                            max="10"
                            value={webIngestionConfig.max_depth}
                            onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, max_depth: parseInt(e.target.value) || 3 }))}
                            className="mt-1"
                          />
                        </div>
                    </div>

                    <details className="text-sm">
                      <summary className="cursor-pointer font-medium text-muted-foreground hover:text-foreground transition-colors mb-2 select-none">
                        {t("kb.dialog.webImport.advanced.title")}
                      </summary>
                      <div className="space-y-4 pt-2 pl-2 border-l-2 border-muted ml-1">
                        <div>
                          <Label htmlFor="dialog-url-patterns">{t("kb.dialog.webImport.advanced.urlPatterns")}</Label>
                          <Input
                            id="dialog-url-patterns"
                            placeholder="https://example.com/blog/*"
                            value={webIngestionConfig.url_patterns}
                            onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, url_patterns: e.target.value }))}
                            className="mt-1"
                          />
                        </div>
                        <div>
                          <Label htmlFor="dialog-exclude-patterns">{t("kb.dialog.webImport.advanced.excludePatterns")}</Label>
                          <Input
                            id="dialog-exclude-patterns"
                            placeholder="*.png, *.jpg"
                            value={webIngestionConfig.exclude_patterns}
                            onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, exclude_patterns: e.target.value }))}
                            className="mt-1"
                          />
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <Label htmlFor="dialog-content-selector">{t("kb.dialog.webImport.advanced.contentSelector")}</Label>
                            <Input
                              id="dialog-content-selector"
                              placeholder="main, article, .content"
                              value={webIngestionConfig.content_selector}
                              onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, content_selector: e.target.value }))}
                              className="mt-1"
                            />
                          </div>
                          <div>
                            <Label htmlFor="dialog-remove-selectors">{t("kb.dialog.webImport.advanced.removeSelectors")}</Label>
                            <Input
                              id="dialog-remove-selectors"
                              placeholder="nav, footer, .ads"
                              value={webIngestionConfig.remove_selectors}
                              onChange={(e) => setWebIngestionConfig(prev => ({ ...prev, remove_selectors: e.target.value }))}
                              className="mt-1"
                            />
                          </div>
                        </div>
                      </div>
                    </details>

                    <Button
                      onClick={() => handleWebIngest()}
                      disabled={!webIngestionConfig.start_url || isWebIngesting}
                      className="w-full mt-2"
                    >
                      {isWebIngesting ? (
                         <div className="flex items-center gap-2">
                            <Loader2 className="h-4 w-4 animate-spin" />
                            {t("kb.dialog.webImport.status.crawling")}
                         </div>
                      ) : (
                         t("kb.index.startImport")
                      )}
                    </Button>
                 </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
    </div>
  )
}
