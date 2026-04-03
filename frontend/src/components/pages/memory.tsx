"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { SearchInput } from "@/components/ui/search-input"
import { Badge } from "@/components/ui/badge"
import { InfoTooltip } from "@/components/ui/tooltip"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { cn } from "@/lib/utils"
import {
  Database,
  Plus,
  Edit,
  Trash2,
  Save,
  X,
  Globe,
  User,
  Wrench,
  Layers,
  FileText,
  Download
} from "lucide-react"

interface MemoryItem {
  id: string
  content: string
  keywords: string[]
  tags: string[]
  category: string
  timestamp: string
  mime_type: string
  metadata: Record<string, any>
}

interface MemoryStats {
  total_count: number
  category_counts: Record<string, number>
  tag_counts: Record<string, number>
  memory_store_type: string
  error?: string
}

interface MemoryFilters {
  category?: string
  tags?: string[]
  keywords?: string[]
  date_from?: string
  date_to?: string
  search?: string
}

export function MemoryPage() {
  const { t, locale } = useI18n()
  const [memories, setMemories] = useState<MemoryItem[]>([])
  const [stats, setStats] = useState<MemoryStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false)
  const [isViewDialogOpen, setIsViewDialogOpen] = useState(false)
  const [viewingMemory, setViewingMemory] = useState<MemoryItem | null>(null)
  const [editingMemory, setEditingMemory] = useState<MemoryItem | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [isBulkDeleting, setIsBulkDeleting] = useState(false)
  const [similarityThreshold, setSimilarityThreshold] = useState([1.5])

  const [filters, setFilters] = useState<MemoryFilters>({})
  const [limit, setLimit] = useState(50)
  const [offset, setOffset] = useState(0)

  const [formData, setFormData] = useState({
    content: "",
    keywords: "",
    tags: "",
    category: "general",
    metadata: ""
  })

  const [categories, setCategories] = useState<string[]>(['execution_memory', 'react_memory', 'general'])

  // Search debouncing
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const isComposingRef = useRef(false)
  const [searchInput, setSearchInput] = useState(filters.search || "")

  const debouncedSearch = useCallback((searchTerm: string) => {
    if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current)
    searchTimeoutRef.current = setTimeout(() => {
      setFilters(prev => ({ ...prev, search: searchTerm }))
    }, 500)
  }, [])

  const handleSearchChange = (value: string) => {
    setSearchInput(value)
    if (!isComposingRef.current) debouncedSearch(value)
  }

  const handleCompositionStart = () => { isComposingRef.current = true }

  const handleCompositionEnd = (e: React.CompositionEvent<HTMLInputElement>) => {
    isComposingRef.current = false
    debouncedSearch(e.currentTarget.value)
  }

  useEffect(() => {
    return () => { if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current) }
  }, [])

  useEffect(() => { setSearchInput(filters.search || "") }, [filters.search])

  const handleToggleSelect = (id: string) => {
    const newSelected = new Set(selectedIds)
    if (newSelected.has(id)) {
      newSelected.delete(id)
    } else {
      newSelected.add(id)
    }
    setSelectedIds(newSelected)
  }

  const handleSelectAll = () => {
    if (selectedIds.size === memories.length && memories.length > 0) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(memories.map(m => m.id)))
    }
  }

  const handleBulkDelete = async () => {
    try {
      await Promise.all(Array.from(selectedIds).map(id =>
        apiRequest(`${getApiUrl()}/api/memory/${id}`, { method: "DELETE", headers: {} })
      ))
      setSelectedIds(new Set())
      setIsBulkDeleting(false)
      fetchData()
    } catch (err) {
      setError(err instanceof Error ? err.message : t("memory.errors.deleteBatchFailed"))
    }
  }

  const handleBulkExport = () => {
    const targetMemories = selectedIds.size > 0
      ? memories.filter(m => selectedIds.has(m.id))
      : memories

    if (targetMemories.length === 0) return

    const csvContent = "data:text/csv;charset=utf-8,"
      + "ID,Content,Keywords,Tags,Category,Timestamp,Mime Type,Metadata\n"
      + targetMemories.map(m => {
        const content = m.content.replace(/"/g, '""')
        const keywords = (m.keywords || []).join("|").replace(/"/g, '""')
        const tags = (m.tags || []).join("|").replace(/"/g, '""')
        const metadata = JSON.stringify(m.metadata || {}).replace(/"/g, '""')
        return `"${m.id}","${content}","${keywords}","${tags}","${m.category}","${m.timestamp}","${m.mime_type || ''}","${metadata}"`
      }).join("\n")

    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", `memories_export_${new Date().toISOString().slice(0, 10)}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const fetchData = async () => {
    try {
      setLoading(true)
      setError(null)

      const params = new URLSearchParams()
      if (filters.category) params.append("category", filters.category)
      if (filters.tags?.length) params.append("tags", filters.tags.join(","))
      if (filters.keywords?.length) params.append("keywords", filters.keywords.join(","))
      if (filters.date_from) params.append("date_from", filters.date_from)
      if (filters.date_to) params.append("date_to", filters.date_to)
      if (filters.search) params.append("search", filters.search)
      params.append("similarity_threshold", similarityThreshold[0].toString())
      if (limit) params.append("limit", limit.toString())
      if (offset) params.append("offset", offset.toString())

      const [memoriesResponse, statsResponse] = await Promise.all([
        apiRequest(`${getApiUrl()}/api/memory/list?${params.toString()}`, { headers: {} }),
        apiRequest(`${getApiUrl()}/api/memory/stats`, { headers: {} })
      ])

      if (!memoriesResponse.ok) throw new Error("Failed to fetch memories")
      if (!statsResponse.ok) throw new Error("Failed to fetch stats")

      const memoriesData = await memoriesResponse.json()
      const statsData = await statsResponse.json()

      // Client-side filtering for the hardcoded data
      let filteredMemories = memoriesData.memories
      if (filters.category) {
        filteredMemories = filteredMemories.filter((m: any) => m.category === filters.category)
      }

      setMemories(filteredMemories)
      setStats(statsData)
    } catch (err) {
      setError(err instanceof Error ? err.message : t("memory.errors.unknown"))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [filters, limit, offset])

  const handleCreateMemory = async () => {
    try {
      const memoryData = {
        content: formData.content,
        keywords: formData.keywords.split(",").map(k => k.trim()).filter(k => k),
        tags: formData.tags.split(",").map(t => t.trim()).filter(t => t),
        category: formData.category,
        metadata: formData.metadata ? JSON.parse(formData.metadata) : {}
      }

      const response = await apiRequest(`${getApiUrl()}/api/memory`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(memoryData)
      })

      if (!response.ok) throw new Error("Failed to create memory")

      setIsCreateDialogOpen(false)
      setFormData({ content: "", keywords: "", tags: "", category: "general", metadata: "" })
      fetchData()
    } catch (err) {
      setError(err instanceof Error ? err.message : t("memory.errors.createFailed"))
    }
  }

  const handleUpdateMemory = async () => {
    if (!editingMemory) return
    try {
      const updateData = {
        content: formData.content,
        keywords: formData.keywords.split(",").map(k => k.trim()).filter(k => k),
        tags: formData.tags.split(",").map(t => t.trim()).filter(t => t),
        category: formData.category,
        metadata: formData.metadata ? JSON.parse(formData.metadata) : {}
      }

      const response = await apiRequest(`${getApiUrl()}/api/memory/${editingMemory.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updateData)
      })

      if (!response.ok) throw new Error("Failed to update memory")

      setIsEditDialogOpen(false)
      setEditingMemory(null)
      setFormData({ content: "", keywords: "", tags: "", category: "general", metadata: "" })
      fetchData()
    } catch (err) {
      setError(err instanceof Error ? err.message : t("memory.errors.updateFailed"))
    }
  }

  const handleDeleteMemory = async (id: string) => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/memory/${id}`, { method: "DELETE", headers: {} })
      if (!response.ok) throw new Error("Failed to delete memory")
      setDeletingId(null)
      fetchData()
    } catch (err) {
      setError(err instanceof Error ? err.message : t("memory.errors.deleteFailed"))
    }
  }

  const openViewDialog = (memory: MemoryItem) => {
    setViewingMemory(memory)
    setIsViewDialogOpen(true)
  }

  const openEditDialog = (memory: MemoryItem) => {
    setEditingMemory(memory)
    setFormData({
      content: memory.content,
      keywords: memory.keywords.join(", "),
      tags: memory.tags.join(", "),
      category: memory.category,
      metadata: JSON.stringify(memory.metadata, null, 2)
    })
    setIsEditDialogOpen(true)
  }

  const formatDate = (dateString: string) => {
    const lang = locale === "zh" ? "zh-CN" : "en-US"
    return new Date(dateString).toLocaleString(lang)
  }

  // Helper to determine icon based on category
  const getCategoryIcon = (category: string) => {
    const lower = category.toLowerCase()
    if (lower === 'react_memory') return Wrench
    if (lower === 'execution_memory') return FileText
    if (lower === 'general') return User
    return Layers
  }

  // Helper to determine badge color
  const getCategoryColor = (category: string) => {
    const lower = category.toLowerCase()
    if (lower === 'react_memory') return "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300"
    if (lower === 'execution_memory') return "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300"
    if (lower === 'general') return "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-300"
    return "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-300"
  }

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Top Header */}
      <div className="border-b flex justify-between items-center p-8">
        <div>
          <h1 className="text-3xl font-bold mb-1">{t("memory.header.title")}</h1>
          <p className="text-muted-foreground">{t("memory.header.description")}</p>
        </div>

        <div className="flex items-center gap-3">
          <SearchInput
            placeholder={t("memory.filters.search.placeholder")}
            className="h-9"
            containerClassName="w-64 hidden sm:block"
            value={searchInput}
            onChange={handleSearchChange}
            onCompositionStart={handleCompositionStart}
            onCompositionEnd={handleCompositionEnd}
          />

          <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button size="sm" className="bg-primary hover:bg-primary/90 text-primary-foreground">
                <Plus className="h-4 w-4 mr-2" />
                {t("memory.header.create")}
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>{t("memory.createDialog.title")}</DialogTitle>
                <DialogDescription>{t("memory.createDialog.description")}</DialogDescription>
              </DialogHeader>
              <MemoryForm
                formData={formData}
                setFormData={setFormData}
                categories={categories}
                onSubmit={handleCreateMemory}
                onCancel={() => setIsCreateDialogOpen(false)}
              />
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <aside className="w-64 border-r bg-muted/10 flex flex-col flex-shrink-0 overflow-y-auto">
          <div className="p-4 space-y-8">
            <div className="space-y-3">
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-2">
                {t("memory.sidebar.contextSource")}
              </h3>
              <div className="space-y-1">
                <Button
                  variant={!filters.category ? "secondary" : "ghost"}
                  className={cn("w-full justify-start", !filters.category && "font-medium")}
                  onClick={() => setFilters(prev => ({ ...prev, category: undefined }))}
                >
                  <Database className="h-4 w-4 mr-2" />
                  {t("memory.sidebar.allMemories")}
                </Button>
              </div>
            </div>

            <div className="space-y-3">
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-2">
                {t("memory.sidebar.byCategory")}
              </h3>
              <div className="space-y-1">
                {categories.map(cat => (
                  <Button
                    key={cat}
                    variant={filters.category === cat ? "secondary" : "ghost"}
                    className={cn("w-full justify-start", filters.category === cat && "font-medium")}
                    onClick={() => setFilters(prev => ({ ...prev, category: cat }))}
                  >
                    <div className={cn("h-2 w-2 rounded-full shrink-0",
                      cat === 'general' ? 'bg-slate-400' :
                        cat === 'react_memory' ? 'bg-purple-400' :
                          'bg-blue-400'
                    )} />
                    {t(`memory.filters.categoryOptions.${cat}`)}
                    {stats?.category_counts[cat] ? (
                      <span className="ml-auto text-xs text-muted-foreground">
                        {stats.category_counts[cat]}
                      </span>
                    ) : null}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col bg-slate-50/50 dark:bg-background overflow-hidden">
          {/* Main Content Header */}
          <div className="h-24 flex items-center justify-between px-8 bg-card flex-shrink-0">
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary accent-primary"
                checked={memories.length > 0 && selectedIds.size === memories.length}
                onChange={handleSelectAll}
              />
              <h2 className="text-lg font-semibold">
                {filters.category ? t(`memory.filters.categoryOptions.${filters.category}`) : t("memory.sidebar.allMemories")}
              </h2>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" className="h-8 text-xs" onClick={handleBulkExport}>
                <Download className="h-3.5 w-3.5 mr-2" />
                {selectedIds.size > 0 ? t("memory.actions.exportSelected", { count: selectedIds.size }) : t("memory.actions.exportCsv")}
              </Button>
              {selectedIds.size > 0 ? (
                <Button variant="destructive" size="sm" className="h-8 text-xs" onClick={() => setIsBulkDeleting(true)}>
                  <Trash2 className="h-3.5 w-3.5 mr-2" />
                  {t("memory.actions.deleteSelected", { count: selectedIds.size })}
                </Button>
              ) : (
                filters.category && (
                  <Button variant="ghost" size="sm" className="h-8 text-xs text-muted-foreground hover:text-foreground" onClick={() => setFilters(prev => ({ ...prev, category: undefined }))}>
                    <X className="h-3.5 w-3.5 mr-2" />
                    {t("memory.actions.clearFilter")}
                  </Button>
                )
              )}
            </div>
          </div>

          {/* Memory List */}
          <div className="flex-1 overflow-y-auto">
            {error && (
              <Alert variant="destructive" className="mb-6">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {loading && !memories.length ? (
              <div className="flex items-center justify-center h-64">
                <div className="text-center">
                  <Database className="h-12 w-12 text-blue-400 mx-auto mb-4 animate-pulse" />
                  <p className="text-muted-foreground">{t("memory.loading")}</p>
                </div>
              </div>
            ) : memories.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                <div className="h-16 w-16 bg-muted rounded-full flex items-center justify-center mb-4">
                  <Database className="h-8 w-8 opacity-50" />
                </div>
                <h3 className="text-lg font-medium mb-1">{t("memory.empty.title")}</h3>
                <p className="text-sm max-w-sm text-center">
                  {filters.search ? t("memory.empty.searchHint") : t("memory.empty.categoryHint")}
                </p>
              </div>
            ) : (
              <div className="space-y-4 mx-auto px-8">
                {memories.map((memory) => {
                  const Icon = getCategoryIcon(memory.category)
                  const badgeColorClass = getCategoryColor(memory.category)

                  return (
                    <Card
                      key={memory.id}
                      className={cn(
                        "group cursor-pointer hover:shadow-md transition-all duration-200 border-border/60",
                        selectedIds.has(memory.id) && "border-primary/50 bg-accent/50"
                      )}
                      onClick={() => openViewDialog(memory)}
                    >
                      <CardContent className="p-5 flex gap-5">
                        {/* Checkbox */}
                        <div
                          className={cn(
                            "flex items-center justify-center pt-1 transition-opacity duration-200",
                            selectedIds.has(memory.id) ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                          )}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <input
                            type="checkbox"
                            className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary accent-primary cursor-pointer"
                            checked={selectedIds.has(memory.id)}
                            onChange={() => handleToggleSelect(memory.id)}
                          />
                        </div>

                        {/* Icon Box */}
                        <div className={cn("h-12 w-12 rounded-xl flex items-center justify-center flex-shrink-0 mt-1", badgeColorClass.split(' ')[0])}>
                          <Icon className={cn("h-6 w-6", badgeColorClass.split(' ')[1])} />
                        </div>

                        {/* Content */}
                        <div className="flex-1 min-w-0 space-y-2">
                          <div className="flex items-center gap-3">
                            <Badge variant="outline" className={cn("rounded-md border-0 px-2 py-0.5 text-xs font-medium uppercase tracking-wide", badgeColorClass)}>
                              {t(`memory.filters.categoryOptions.${memory.category}`) === `memory.filters.categoryOptions.${memory.category}`
                                ? memory.category.replace(/_/g, ' ')
                                : t(`memory.filters.categoryOptions.${memory.category}`)}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {memory.category === 'general' ? t("memory.item.addedManually") : t("memory.item.addedAutomatically")} • {formatDate(memory.timestamp)}
                            </span>
                          </div>

                          <p className="text-sm text-foreground/90 leading-relaxed line-clamp-3">
                            {memory.content}
                          </p>

                          {/* Footer / Source */}
                          <div className="pt-1 flex items-center gap-2 flex-wrap">
                            {/* Keywords */}
                            {memory.keywords.length > 0 && (
                              <div className="flex items-center gap-1 ml-2">
                                <span className="text-xs text-muted-foreground">{t("memory.item.keywordsLabel")}</span>
                                {memory.keywords.slice(0, 3).map(keyword => (
                                  <Badge key={keyword} variant="secondary" className="text-[10px] px-1.5 h-5 font-normal bg-blue-50 text-blue-700 hover:bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300 border-0">
                                    {keyword}
                                  </Badge>
                                ))}
                              </div>
                            )}

                            {memory.tags.length > 0 && (
                              <div className="flex items-center gap-1 ml-2">
                                <span className="text-xs text-muted-foreground">{t("memory.item.tagsLabel")}</span>
                                {memory.tags.slice(0, 3).map(tag => (
                                  <span key={tag} className="text-xs text-muted-foreground bg-muted/50 px-1.5 py-0.5 rounded">
                                    #{tag}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Actions */}
                        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity self-start">
                          <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground" onClick={(e) => { e.stopPropagation(); openEditDialog(memory); }}>
                            <Edit className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive/70 hover:text-destructive" onClick={(e) => { e.stopPropagation(); setDeletingId(memory.id); }}>
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            )}
          </div>
        </main>
      </div>

      {/* Dialogs */}
      <Dialog open={isViewDialogOpen} onOpenChange={setIsViewDialogOpen}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle>{t("memory.viewDialog.title")}</DialogTitle>
            <DialogDescription>{t("memory.viewDialog.description")}</DialogDescription>
          </DialogHeader>
          {viewingMemory && (
            <ScrollArea className="flex-1 overflow-y-auto">
              <div className="space-y-4 p-1">
                <div className="flex items-center gap-2 mb-4">
                  <Badge variant="secondary">{viewingMemory.category}</Badge>
                  <span className="text-sm text-muted-foreground">{formatDate(viewingMemory.timestamp)}</span>
                </div>
                <div className="p-4 bg-muted/50 rounded-lg">
                  <pre className="whitespace-pre-wrap text-sm font-sans text-foreground">{viewingMemory.content}</pre>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {viewingMemory.keywords.length > 0 && (
                    <div>
                      <Label className="text-xs font-medium text-muted-foreground uppercase mb-2 block">{t("memory.viewDialog.labels.keywords")}</Label>
                      <div className="flex flex-wrap gap-2">
                        {viewingMemory.keywords.map((k, i) => <Badge key={i} variant="outline">{k}</Badge>)}
                      </div>
                    </div>
                  )}
                  {viewingMemory.tags.length > 0 && (
                    <div>
                      <Label className="text-xs font-medium text-muted-foreground uppercase mb-2 block">{t("memory.viewDialog.labels.tags")}</Label>
                      <div className="flex flex-wrap gap-2">
                        {viewingMemory.tags.map((t, i) => <Badge key={i} variant="outline">{t}</Badge>)}
                      </div>
                    </div>
                  )}
                </div>

                {Object.keys(viewingMemory.metadata).length > 0 && (
                  <div>
                    <Label className="text-xs font-medium text-muted-foreground uppercase mb-2 block">{t("memory.viewDialog.labels.metadata")}</Label>
                    <div className="p-3 bg-muted/50 rounded-md">
                      <pre className="whitespace-pre-wrap text-xs font-mono">{JSON.stringify(viewingMemory.metadata, null, 2)}</pre>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          )}
          <div className="flex justify-end gap-2 mt-4 pt-4 border-t">
            <Button variant="outline" onClick={() => setIsViewDialogOpen(false)}>{t("memory.viewDialog.close")}</Button>
            <Button onClick={() => { setIsViewDialogOpen(false); openEditDialog(viewingMemory!) }}>
              <Edit className="h-4 w-4 mr-2" />
              {t("memory.viewDialog.edit")}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>{t("memory.editDialog.title")}</DialogTitle>
            <DialogDescription>{t("memory.editDialog.description")}</DialogDescription>
          </DialogHeader>
          <MemoryForm
            formData={formData}
            setFormData={setFormData}
            categories={categories}
            onSubmit={handleUpdateMemory}
            onCancel={() => { setIsEditDialogOpen(false); setEditingMemory(null) }}
          />
        </DialogContent>
      </Dialog>

      <Dialog open={!!deletingId} onOpenChange={() => setDeletingId(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("memory.deleteDialog.title")}</DialogTitle>
            <DialogDescription>{t("memory.deleteDialog.description")}</DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setDeletingId(null)}>{t("common.cancel")}</Button>
            <Button variant="destructive" onClick={() => deletingId && handleDeleteMemory(deletingId)}>{t("memory.deleteDialog.delete")}</Button>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={isBulkDeleting} onOpenChange={setIsBulkDeleting}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("memory.deleteDialog.title")}</DialogTitle>
            <DialogDescription>
              {t("memory.deleteDialog.bulkDescription", { count: selectedIds.size })}
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setIsBulkDeleting(false)}>{t("common.cancel")}</Button>
            <Button variant="destructive" onClick={handleBulkDelete}>{t("memory.deleteDialog.delete")}</Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

interface MemoryFormProps {
  formData: { content: string; keywords: string; tags: string; category: string; metadata: string }
  setFormData: (data: any) => void
  categories: string[]
  onSubmit: () => void
  onCancel: () => void
}

function MemoryForm({ formData, setFormData, categories, onSubmit, onCancel }: MemoryFormProps) {
  const { t } = useI18n()
  return (
    <div className="space-y-4">
      <div>
        <div className="mb-1.5">
          <Label htmlFor="content">{t("memory.form.contentLabel")}</Label>
        </div>
        <Textarea
          id="content"
          placeholder={t("memory.form.contentPlaceholder")}
          value={formData.content}
          onChange={(e) => setFormData((prev: any) => ({ ...prev, content: e.target.value }))}
          rows={4}
        />
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="flex items-center gap-1.5 mb-1.5">
            <Label htmlFor="keywords">{t("memory.form.keywordsLabel")}</Label>
            <InfoTooltip content={t("memory.form.keywordsTooltip")} />
          </div>
          <Input
            id="keywords"
            placeholder={t("memory.form.keywordsPlaceholder")}
            value={formData.keywords}
            onChange={(e) => setFormData((prev: any) => ({ ...prev, keywords: e.target.value }))}
          />
        </div>
        <div>
          <div className="flex items-center gap-1.5 mb-1.5">
            <Label htmlFor="tags">{t("memory.form.tagsLabel")}</Label>
            <InfoTooltip content={t("memory.form.tagsTooltip")} />
          </div>
          <Input
            id="tags"
            placeholder={t("memory.form.tagsPlaceholder")}
            value={formData.tags}
            onChange={(e) => setFormData((prev: any) => ({ ...prev, tags: e.target.value }))}
          />
        </div>
      </div>
      {/* Category field hidden */}
      <div>
        <div className="mb-1.5">
          <Label htmlFor="metadata">{t("memory.form.metadataLabel")}</Label>
        </div>
        <Textarea
          id="metadata"
          placeholder={t("memory.form.metadataPlaceholder")}
          value={formData.metadata}
          onChange={(e) => setFormData((prev: any) => ({ ...prev, metadata: e.target.value }))}
          rows={3}
        />
      </div>
      <div className="flex justify-end gap-2 pt-2">
        <Button variant="outline" onClick={onCancel}>{t("common.cancel")}</Button>
        <Button onClick={onSubmit} disabled={!formData.content.trim()}>
          <Save className="h-4 w-4 mr-2" />
          {t("common.save")}
        </Button>
      </div>
    </div>
  )
}
