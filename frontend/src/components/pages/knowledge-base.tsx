"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { getApiUrl } from "@/lib/utils"
import { useAuth } from "@/contexts/auth-context"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import {
  Plus,
  Search,
  FileText,
  FolderOpen,
  HardDrive
} from "lucide-react"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet"
import { KnowledgeBaseDetailContent } from "@/components/kb/knowledge-base-detail"
import { KnowledgeBaseCreationDialog } from "@/components/kb/knowledge-base-creation-dialog"
import { toast } from "sonner"

interface Collection {
  name: string
  documents: number
  parses: number
  chunks: number
  embeddings: number
  document_names: string[]
}

export function KnowledgeBasePage() {
  const { token } = useAuth()
  const { t, locale } = useI18n()
  const [collections, setCollections] = useState<Collection[]>([])
  const [loading, setLoading] = useState(true)
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [filteredCollections, setFilteredCollections] = useState<Collection[]>([])
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null)
  const [isDrawerOpen, setIsDrawerOpen] = useState(false)

  useEffect(() => {
    fetchCollections()
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

  const handleViewDetail = (collectionName: string) => {
    setSelectedCollection(collectionName)
    setIsDrawerOpen(true)
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
        <KnowledgeBaseCreationDialog
          open={isCreateDialogOpen}
          onOpenChange={setIsCreateDialogOpen}
          onSuccess={() => {
            fetchCollections()
          }}
        />

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
