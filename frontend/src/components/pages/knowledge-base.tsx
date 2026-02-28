"use client"

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
import { Select, SelectRadix, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { getApiUrl } from "@/lib/utils"
import { useAuth } from "@/contexts/auth-context"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import Link from "next/link"
import {
  ArrowLeft,
  Plus,
  Trash2,
  Edit,
  Search,
  FileText,
  Upload,
  Clock,
  CheckCircle,
  AlertCircle,
  XCircle,
  Eye,
  FolderOpen,
  HardDrive,
  AlertTriangle,
  Loader2,
  Globe,
  Settings
} from "lucide-react"

interface Collection {
  name: string
  documents: number
  parses: number
  chunks: number
  embeddings: number
  document_names: string[]
}

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

import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet"
import { KnowledgeBaseDetailContent } from "./knowledge-base-detail"
import { toast } from "sonner"

export function KnowledgeBasePage() {
  const { token } = useAuth()
  const { t, locale } = useI18n()
  const [collections, setCollections] = useState<Collection[]>([])
  const [loading, setLoading] = useState(true)
  const [deletingCollection, setDeletingCollection] = useState<string | null>(null)
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [filteredCollections, setFilteredCollections] = useState<Collection[]>([])
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null)
  const [isDrawerOpen, setIsDrawerOpen] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // 文件上传相关状态
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [ingestionResults, setIngestionResults] = useState<IngestionResult[]>([])
  const [isDragging, setIsDragging] = useState(false)

  // 网站导入相关状态
  const [isWebIngesting, setIsWebIngesting] = useState(false)
  const [webIngestionProgress, setWebIngestionProgress] = useState(0)
  const [webIngestionResult, setWebIngestionResult] = useState<WebIngestionResult | null>(null)
  const [activeImportTab, setActiveImportTab] = useState<"file" | "web">("file")
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

  // 新建知识库表单状态
  const [newCollectionName, setNewCollectionName] = useState("")
  const [newCollectionDescription, setNewCollectionDescription] = useState("")

  // Embedding models state
  const [embeddingModels, setEmbeddingModels] = useState<any[]>([])

  // 索引配置状态
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

  useEffect(() => {
    fetchCollections()
    fetchEmbeddingModels()
  }, [])

  useEffect(() => {
    if (searchQuery) {
      setFilteredCollections(
        collections.filter(collection =>
          collection.name.toLowerCase().includes(searchQuery.toLowerCase())
        )
      )
    } else {
      setFilteredCollections(collections)
    }
  }, [searchQuery, collections])

  const fetchCollections = async () => {
    try {
      setLoading(true)
      const response = await apiRequest(`${getApiUrl()}/api/kb/collections`)

      if (!response.ok) {
        throw new Error("Failed to fetch collections")
      }

      const data = await response.json()
      setCollections(data.collections || [])
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }

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
          // Fallback to first model if no default set
          setIngestionConfig(prev => ({ ...prev, embedding_model_id: models[0].model_id }))
        }
      } else if (models.length > 0) {
        // Fallback to first model
        setIngestionConfig(prev => ({ ...prev, embedding_model_id: models[0].model_id }))
      }
    } catch (err) {
      console.error("Failed to fetch embedding models:", err)
    }
  }

  const handleCreateCollection = async () => {
    if (!newCollectionName.trim()) {
      toast.error(t("kb.errors.nameRequired"))
      return
    }

    try {
      setIsCreateDialogOpen(false)

      // 这里可以扩展为真正的创建collection API
      // 目前通过上传文件来创建collection

      // Reset form
      setNewCollectionName("")
      setNewCollectionDescription("")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("kb.errors.createFailed"))
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

        // 使用文件名作为collection名称，如果没有指定新的collection
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

        // 如果结果是partial，且有failed_step，则认为是失败
        if (result.status === "partial" && result.failed_step) {
          throw new Error(result.message || `Failed at step: ${result.failed_step}`)
        }

        setUploadProgress(((i + 1) / selectedFiles.length) * 100)
      }

      // 上传成功后刷新列表
      await fetchCollections()

      // 重置状态
      setSelectedFiles([])
      setUploadProgress(0)
      setIsCreateDialogOpen(false)
      setNewCollectionName("")
      setNewCollectionDescription("")
      setActiveImportTab("file")

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
        throw new Error(errorData.detail || t("kb.errors.webIngestFailed"))
      }

      const result: WebIngestionResult = await response.json()
      setWebIngestionResult(result)
      setWebIngestionProgress(100)

      // 导入成功后刷新列表
      await fetchCollections()

      // 重置状态
      setIsCreateDialogOpen(false)
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

    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("kb.errors.webIngestFailed"))
    } finally {
      setIsWebIngesting(false)
      setWebIngestionProgress(0)
    }
  }

  const handleViewDetail = (collectionName: string) => {
    setSelectedCollection(collectionName)
    setIsDrawerOpen(true)
  }

  const handleDeleteCollection = async (collectionName: string) => {
    if (!confirm(t("kb.actions.deleteConfirm", { name: collectionName }))) {
      return
    }

    setDeletingCollection(collectionName)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/kb/collections/${encodeURIComponent(collectionName)}`, {
        method: "DELETE"
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || t("kb.errors.deleteFailed", { name: collectionName }))
      }

      const result = await response.json()
      console.log("删除成功:", result)

      // 删除成功后刷新列表
      await fetchCollections()

    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("kb.errors.deleteFailedGeneric"))
    } finally {
      setDeletingCollection(null)
    }
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

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString(locale)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="text-center">
          <HardDrive className="h-12 w-12 mx-auto mb-4 animate-spin text-muted-foreground" />
          <p>{t("kb.loading.loadingKB")}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="w-full p-8">
        {/* Header */}
        <div className="flex justify-between items-start mb-8">
          <div>
            <h1 className="text-3xl font-bold mb-1">{t("kb.header.title")}</h1>
            <p className="text-muted-foreground">{t("kb.header.description")}</p>
          </div>

          <div className="flex items-center gap-4">
            <div className="relative w-64">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
              <Input
                placeholder={t("kb.search.placeholder")}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <Button onClick={() => { setIsCreateDialogOpen(true) }} className="flex items-center gap-2">
              <Plus size={16} className="mr-2" />
              {t("kb.header.new")}
            </Button>
          </div>
        </div>

        {/* Collections Grid */}
        {filteredCollections.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredCollections.map((collection) => (
              <Card
                key={collection.name}
                className="py-0 hover:shadow-lg transition-shadow cursor-pointer overflow-hidden flex flex-col"
                onClick={() => handleViewDetail(collection.name)}
              >
                <div className="p-6 flex-1">
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex gap-4 min-w-0 flex-1 mr-2">
                      <div className="h-10 w-10 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center flex-shrink-0">
                        <FolderOpen className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                      </div>
                      <div className="min-w-0">
                        <h3 className="text-lg font-semibold truncate" title={collection.name}>{collection.name}</h3>
                        <p className="text-sm text-muted-foreground truncate" title={collection.document_names && collection.document_names.length > 0 ? collection.document_names.join(", ") : t("kb.card.noDescription")}>
                          {collection.document_names && collection.document_names.length > 0
                            ? collection.document_names.join(", ")
                            : t("kb.card.noDescription")}
                        </p>
                      </div>
                    </div>
                    <Badge variant="outline" className="text-green-600 bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-900 ml-2 whitespace-nowrap flex-shrink-0">
                      {t("kb.card.status.active")}
                    </Badge>
                  </div>
                </div>

                <div className="px-6 py-4 bg-muted/30 border-t flex justify-between items-center text-sm text-muted-foreground">
                  <div className="flex items-center">
                    <FileText className="h-4 w-4 mr-2" />
                    {collection.documents} {t("kb.card.documentsLabel")}
                  </div>
                  <div className="flex items-center">
                    <HardDrive className="h-4 w-4 mr-2" />
                    {collection.chunks} {t("kb.card.chunksLabel")}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <Card className="p-12 text-center">
            <FolderOpen size={48} className="mx-auto mb-4 opacity-50 text-muted-foreground" />
            <p className="text-lg mb-2 text-muted-foreground">
              {searchQuery ? t("kb.empty.searchNoMatch") : t("kb.empty.noKB")}
            </p>
            <p className="text-sm text-muted-foreground mb-4">
              {searchQuery ? t("kb.empty.hintSearch") : t("kb.empty.hintCreate")}
            </p>
            {!searchQuery && (
              <Button onClick={() => { setIsCreateDialogOpen(true) }} className="flex items-center gap-2">
                <Plus size={16} className="mr-2" />
                {t("kb.header.new")}
              </Button>
            )}
          </Card>
        )}

        {/* Create Collection Dialog */}
        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
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
                  setIsCreateDialogOpen(false)
                  // 重置状态
                  setSelectedFiles([])
                  setUploadProgress(0)
                  setIngestionResults([])
                  setWebIngestionResult(null)
                  setNewCollectionName("")
                  setNewCollectionDescription("")
                  setActiveImportTab("file")
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

        {/* Detail Drawer */}
        <Sheet open={isDrawerOpen} onOpenChange={setIsDrawerOpen}>
          <SheetContent className="w-[90vw] sm:max-w-[85vw] md:max-w-[1000px] overflow-y-auto">
            <SheetHeader>
              <SheetTitle>{selectedCollection || ""}</SheetTitle>
              <SheetDescription>
                {selectedCollection ? t("kb.detail.viewingDetails", { name: selectedCollection }) : ""}
              </SheetDescription>
            </SheetHeader>
            <div className="h-full pb-10">
              {selectedCollection && (
                <KnowledgeBaseDetailContent collectionName={selectedCollection} />
              )}
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </div>
  )
}
