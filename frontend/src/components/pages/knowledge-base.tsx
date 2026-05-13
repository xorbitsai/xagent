"use client"

import React, { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { SearchInput } from "@/components/ui/search-input"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { getApiUrl } from "@/lib/utils"
import { useI18n } from "@/contexts/i18n-context"
import { useAuth } from "@/contexts/auth-context"
import { apiRequest } from "@/lib/api-wrapper"
import {
  Plus,
  FileText,
  FolderOpen,
  HardDrive,
  Trash2,
  Settings2,
  X,
  Database,
  UploadCloud,
  Globe,
  Search,
  Plug
} from "lucide-react"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet"
import { KnowledgeBaseDetailContent } from "@/components/kb/knowledge-base-detail"
import { KnowledgeBaseCreationDialog } from "@/components/kb/knowledge-base-creation-dialog"
import { FeatureEmptyState } from "@/components/ui/feature-empty-state"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { toast } from "sonner"

interface Collection {
  name: string
  documents: number
  parses: number
  chunks: number
  embeddings: number
  document_names: string[]
  owners?: number[]
}

interface AdminUser {
  id: number
  username: string
}

interface AdminUserListResponse {
  users?: AdminUser[]
  pages?: number
}

export function KnowledgeBasePage() {
  const { user } = useAuth()
  const { t } = useI18n()
  const [collections, setCollections] = useState<Collection[]>([])
  const [loading, setLoading] = useState(true)
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [filteredCollections, setFilteredCollections] = useState<Collection[]>([])
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null)
  const [isDrawerOpen, setIsDrawerOpen] = useState(false)
  const [collectionToDelete, setCollectionToDelete] = useState<string | null>(null)
  const [isDeletingCollection, setIsDeletingCollection] = useState(false)
  const [adminUsers, setAdminUsers] = useState<Record<number, string>>({})
  const [adminUsersLoadFailed, setAdminUsersLoadFailed] = useState(false)
  const [isManageMode, setIsManageMode] = useState(false)
  const [selectedNames, setSelectedNames] = useState<Set<string>>(new Set())

  useEffect(() => {
    fetchCollections()
  }, [])

  useEffect(() => {
    const fetchAdminUsers = async () => {
      if (!user?.is_admin) return
      try {
        setAdminUsersLoadFailed(false)
        const map: Record<number, string> = {}
        let hasRequestError = false

        // API validation requires size <= 100; fetch all pages to avoid
        // missing owner names when user count grows.
        const pageSize = 100
        const maxPagesToScan = 1000
        let page = 1
        let totalPages: number | null = null

        while (page <= maxPagesToScan && (totalPages === null || page <= totalPages)) {
          const params = new URLSearchParams({
            page: page.toString(),
            size: pageSize.toString(),
          })
          const response = await apiRequest(`${getApiUrl()}/api/admin/users?${params.toString()}`)
          if (!response.ok) {
            hasRequestError = true
            break
          }

          const data: AdminUserListResponse = await response.json()
          const users: AdminUser[] = data.users || []
          for (const u of users) {
            map[u.id] = u.username
          }

          if (typeof data.pages === "number" && data.pages > 0) {
            totalPages = data.pages
          } else if (users.length < pageSize) {
            break
          }

          page += 1
        }

        setAdminUsers(map)
        if (hasRequestError && Object.keys(map).length === 0) {
          setAdminUsersLoadFailed(true)
        }
      } catch (err) {
        setAdminUsersLoadFailed(true)
        console.error("Failed to load admin user list for KB owners display:", err)
      }
    }
    fetchAdminUsers()
  }, [user?.is_admin])

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

  // Keep selection scoped to the current view to avoid hidden-item side effects.
  useEffect(() => {
    if (isManageMode) {
      setSelectedNames(new Set())
    }
  }, [searchQuery, isManageMode])

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

  const handleViewDetail = (collectionName: string) => {
    setSelectedCollection(collectionName)
    setIsDrawerOpen(true)
  }

  const handleDrawerOpenChange = (open: boolean) => {
    setIsDrawerOpen(open)

    if (!open) {
      setSelectedCollection(null)
    }
  }

  const handleDeleteCollection = async () => {
    if (!collectionToDelete) {
      return
    }

    const targetCollection = collectionToDelete
    setIsDeletingCollection(true)

    try {
      const response = await apiRequest(
        `${getApiUrl()}/api/kb/collections/${encodeURIComponent(targetCollection)}`,
        { method: "DELETE" }
      )
      const result = await response.json().catch(() => null)

      if (!response.ok) {
        const errorMessage = typeof result?.detail === "string"
          ? result.detail
          : typeof result?.message === "string"
            ? result.message
            : t("kb.errors.deleteFailed", { name: targetCollection })

        throw new Error(errorMessage)
      }

      const status = typeof result?.status === "string" ? result.status : undefined
      const message = typeof result?.message === "string" ? result.message : ""
      const rawWarnings: unknown[] = Array.isArray(result?.warnings) ? result.warnings : []
      const warnings = rawWarnings.filter(
        (warning: unknown): warning is string => typeof warning === "string" && warning.length > 0
      )

      if (status === "failed") {
        throw new Error(warnings[0] || message || t("kb.errors.deleteFailed", { name: targetCollection }))
      }

      if (status && status !== "success" && status !== "partial_success") {
        throw new Error(warnings[0] || message || t("kb.errors.deleteFailed", { name: targetCollection }))
      }

      toast.success(t("kb.messages.deleteSuccess"))

      setCollectionToDelete(null)
      setSelectedCollection(null)
      setIsDrawerOpen(false)
      await fetchCollections()

      if (status === "partial_success") {
        toast.warning(warnings[0] || message || t("kb.errors.deleteFailedGeneric"))
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : t("kb.errors.deleteFailedGeneric"))
    } finally {
      setIsDeletingCollection(false)
    }
  }

  const toggleSelect = (name: string) => {
    setSelectedNames((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  const visibleNames = filteredCollections.map((c) => c.name)
  const visibleSelectedCount = visibleNames.reduce(
    (count, name) => count + (selectedNames.has(name) ? 1 : 0),
    0
  )
  const allVisibleSelected = visibleNames.length > 0 && visibleSelectedCount === visibleNames.length

  const selectAll = () => {
    setSelectedNames((prev) => {
      const next = new Set(prev)
      if (allVisibleSelected) {
        for (const name of visibleNames) {
          next.delete(name)
        }
      } else {
        for (const name of visibleNames) {
          next.add(name)
        }
      }
      return next
    })
  }

  const handleBatchDelete = async () => {
    const names = Array.from(selectedNames)
    if (names.length === 0) return
    const confirmed = window.confirm(
      t("kb.actions.batchDeleteConfirm", { count: names.length }),
    )
    if (!confirmed) return

    try {
      const response = await apiRequest(`${getApiUrl()}/api/kb/collections/batch-delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ collection_names: names }),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(
          typeof err.detail === "string" ? err.detail : t("kb.errors.batchDeleteFailed"),
        )
      }
      const data = await response.json()
      const deleted = data.deleted?.length ?? 0
      const failed = data.failed?.length ?? 0
      if (deleted > 0) {
        toast.success(t("kb.messages.batchDeleteSuccess", { count: deleted }))
      }
      if (failed > 0) {
        toast.error(t("kb.messages.batchDeleteFailedCount", { count: failed }))
      }
      setSelectedNames(new Set())
      setIsManageMode(false)
      await fetchCollections()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("kb.errors.deleteFailedGeneric"))
    }
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
    <div className="flex flex-col h-full bg-background/50">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center w-full p-4 sm:p-8 gap-4">
        <div className="space-y-1 w-full sm:w-auto">
          <div className="flex items-center gap-3 mb-1 flex-wrap">
            <h1 className="text-2xl sm:text-3xl font-bold">{t("kb.header.title")}</h1>
            <Badge variant="secondary" className="hidden sm:inline-flex font-normal text-xs sm:text-sm">
              {searchQuery
                ? t("kb.header.matchCount", {
                  matched: filteredCollections.length,
                  total: collections.length,
                })
                : t("kb.header.totalCount", { total: collections.length })}
            </Badge>
          </div>
          <p className="text-sm sm:text-base text-muted-foreground">{t("kb.header.description")}</p>
          <Badge variant="secondary" className="sm:hidden inline-flex font-normal text-xs w-fit">
            {searchQuery
              ? t("kb.header.matchCount", {
                matched: filteredCollections.length,
                total: collections.length,
              })
              : t("kb.header.totalCount", { total: collections.length })}
          </Badge>
        </div>

        <div className="flex items-center gap-3 w-full sm:w-auto">
          <SearchInput
            placeholder={t("kb.search.placeholder")}
            value={searchQuery}
            onChange={setSearchQuery}
            containerClassName="flex-1 sm:w-64"
          />
          <Button
            variant={isManageMode ? "secondary" : "outline"}
            onClick={() => {
              setIsManageMode((m) => !m)
              setSelectedNames(new Set())
            }}
            className="flex items-center gap-1 sm:gap-2 shrink-0"
          >
            {isManageMode ? <X size={16} className="sm:mr-2" /> : <Settings2 size={16} className="sm:mr-2" />}
            <span className="hidden sm:inline">{isManageMode ? t("kb.manage.exit") : t("kb.manage.enter")}</span>
          </Button>
          <Button onClick={() => { setIsCreateDialogOpen(true) }} className="flex items-center gap-1 sm:gap-2 shrink-0">
            <Plus size={16} className="sm:mr-2" />
            <span className="hidden sm:inline">{t("kb.header.new")}</span>
            <span className="sm:hidden">{t("common.create") || t("kb.header.new")}</span>
          </Button>
        </div>
      </div>

      {/* Collections Grid */}
      <div className="flex-1 overflow-y-auto w-full px-8 pb-8">
        {collections.length === 0 ? (
          <FeatureEmptyState
            icon={Database}
            title={t("kb.emptyState.title")}
            description={t("kb.emptyState.description")}
            features={[
              {
                icon: UploadCloud,
                title: t("kb.emptyState.features.fileUpload.title"),
                description: t("kb.emptyState.features.fileUpload.description")
              },
              {
                icon: Globe,
                title: t("kb.emptyState.features.websiteImport.title"),
                description: t("kb.emptyState.features.websiteImport.description")
              },
              {
                icon: Search,
                title: t("kb.emptyState.features.semanticSearch.title"),
                description: t("kb.emptyState.features.semanticSearch.description")
              },
              {
                icon: Plug,
                title: t("kb.emptyState.features.connectToAgents.title"),
                description: t("kb.emptyState.features.connectToAgents.description")
              }
            ]}
            actionLabel={t("kb.emptyState.action")}
            onAction={() => setIsCreateDialogOpen(true)}
            className="h-full mt-4"
          />
        ) : (
          <>
            {isManageMode && filteredCollections.length > 0 && (
              <div className="flex items-center gap-4 mb-4 p-3 rounded-lg bg-muted/50 border">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={allVisibleSelected}
                    onChange={selectAll}
                    className="h-4 w-4 rounded border-input"
                  />
                  <span className="text-sm font-medium">
                    {allVisibleSelected ? t("kb.manage.deselectAll") : t("kb.manage.selectAll")}
                  </span>
                </label>
                <Button
                  variant="destructive"
                  size="sm"
                  disabled={selectedNames.size === 0}
                  onClick={handleBatchDelete}
                  className="flex items-center gap-2"
                >
                  <Trash2 className="h-4 w-4" />
                  {t("kb.manage.deleteSelected", { count: selectedNames.size })}
                </Button>
              </div>
            )}
            {filteredCollections.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                {filteredCollections.map((collection) => (
                  <Card
                    key={collection.name}
                    className="py-0 hover:shadow-lg transition-shadow overflow-hidden flex flex-col cursor-pointer"
                    onClick={() => {
                      if (isManageMode) toggleSelect(collection.name)
                      else handleViewDetail(collection.name)
                    }}
                  >
                    <div className="p-6 flex-1">
                      <div className="flex justify-between items-start mb-4">
                        <div className="flex gap-4 min-w-0 flex-1 mr-2">
                          {isManageMode && (
                            <div onClick={(e) => e.stopPropagation()}>
                              <input
                                type="checkbox"
                                checked={selectedNames.has(collection.name)}
                                onChange={() => toggleSelect(collection.name)}
                                className="h-4 w-4 rounded border-input mt-1"
                              />
                            </div>
                          )}
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
                        {!isManageMode && (
                          <div className="flex flex-col items-end gap-1 ml-2 flex-shrink-0">
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-green-600 bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-900 whitespace-nowrap">
                                {t("kb.card.status.active")}
                              </Badge>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-destructive hover:text-destructive"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  setCollectionToDelete(collection.name)
                                }}
                                title={t("kb.card.actions.delete")}
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </div>
                            {user?.is_admin && collection.owners && collection.owners.length > 0 && (
                              <p className="text-xs text-muted-foreground">
                                {adminUsersLoadFailed
                                  ? t("kb.card.ownerFallbackLabel", { owners: collection.owners.join(", ") })
                                  : t("kb.card.ownerLabel", {
                                    owners: collection.owners.map((id) => adminUsers[id] ?? id).join(", "),
                                  })}
                              </p>
                            )}
                          </div>
                        )}
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
          </>
        )}
      </div>

      {/* Create Collection Dialog */}
      <KnowledgeBaseCreationDialog
        open={isCreateDialogOpen}
        onOpenChange={setIsCreateDialogOpen}
        onSuccess={() => {
          fetchCollections()
        }}
      />

      {/* Detail Drawer */}
      <Sheet open={isDrawerOpen} onOpenChange={handleDrawerOpenChange}>
        <SheetContent className="w-[90vw] sm:max-w-[85vw] md:max-w-[1000px] overflow-y-auto">
          <SheetHeader>
            <div className="flex items-start justify-between gap-4 pr-8">
              <div className="space-y-1">
                <SheetTitle>{selectedCollection || ""}</SheetTitle>
                <SheetDescription>
                  {selectedCollection ? t("kb.detail.viewingDetails", { name: selectedCollection }) : ""}
                </SheetDescription>
              </div>
              {selectedCollection && (
                <Button
                  variant="outline"
                  size="sm"
                  className="shrink-0 text-destructive hover:text-destructive"
                  onClick={() => setCollectionToDelete(selectedCollection)}
                >
                  <Trash2 size={16} />
                  {t("common.delete")}
                </Button>
              )}
            </div>
          </SheetHeader>
          <div className="h-full pb-10">
            {selectedCollection && (
              <KnowledgeBaseDetailContent collectionName={selectedCollection} />
            )}
          </div>
        </SheetContent>
      </Sheet>

      <ConfirmDialog
        isOpen={!!collectionToDelete}
        onOpenChange={(open) => !open && setCollectionToDelete(null)}
        onConfirm={handleDeleteCollection}
        isLoading={isDeletingCollection}
        description={t("kb.actions.deleteConfirm", { name: collectionToDelete || "" })}
      />
    </div>
  )
}
