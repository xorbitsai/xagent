import React, { useState } from "react"
import { FileIcon, Loader2, Trash2 } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { toast } from "@/components/ui/sonner"

import {
  buildDeleteDocumentUrl,
  CollectionDocumentInfo,
  CollectionDocumentSource,
  DocumentDisplayStatus,
  getCollectionDocuments,
  getDeleteErrorMessage,
  KnowledgeBaseDocumentStatus,
} from "./knowledge-base-detail-helpers"

interface KnowledgeBaseDocumentListProps {
  collectionInfo: CollectionDocumentSource
  collectionName: string
  onRefresh: () => Promise<void>
  t: (key: string, vars?: Record<string, string | number>) => string
  /**
   * Per-document ingestion status rows (including optimistic `uploading` rows).
   * When provided and non-empty these drive the rendered status and delete
   * controls; otherwise the list falls back to `collectionInfo` documents,
   * treating them as already indexed.
   */
  documentStatuses?: KnowledgeBaseDocumentStatus[]
}

const IN_PROGRESS_STATUSES: ReadonlySet<DocumentDisplayStatus> = new Set([
  "uploading",
  "pending",
  "running",
  "chunked",
  "partially_embedded",
])

function getStatusBadgeVariant(
  status: DocumentDisplayStatus
): "default" | "secondary" | "destructive" | "outline" {
  if (status === "failed") return "destructive"
  if (status === "cancelled") return "outline"
  if (status === "success") return "secondary"
  return "default"
}

export function KnowledgeBaseDocumentList({
  collectionInfo,
  collectionName,
  onRefresh,
  t,
  documentStatuses,
}: KnowledgeBaseDocumentListProps) {
  const [documentToDelete, setDocumentToDelete] = useState<CollectionDocumentInfo | null>(null)
  const [isDeletingDocument, setIsDeletingDocument] = useState(false)

  // Prefer live status rows; fall back to collection documents (treated as
  // indexed) so the list still renders if the status endpoint is unavailable.
  const rows: KnowledgeBaseDocumentStatus[] =
    documentStatuses && documentStatuses.length > 0
      ? documentStatuses
      : getCollectionDocuments(collectionInfo).map((document) => ({
          ...document,
          status: "success" as const,
          can_delete: true,
        }))

  const confirmDeleteDocument = async () => {
    if (!documentToDelete) return
    setIsDeletingDocument(true)

    try {
      const response = await apiRequest(buildDeleteDocumentUrl(getApiUrl(), collectionName, documentToDelete), { method: "DELETE" })
      const result = await response.json().catch(() => null)

      if (!response.ok) {
        throw new Error(getDeleteErrorMessage(result, t("kb.detail.errors.deleteFailed")))
      }

      const status = typeof result?.status === "string" ? result.status : undefined
      const message = typeof result?.message === "string" ? result.message : t("kb.detail.errors.deleteFailed")
      const rawErrors: unknown[] = Array.isArray(result?.errors) ? result.errors : []
      const errors = rawErrors.filter((error): error is string => typeof error === "string" && error.length > 0)

      if (status === "partial_success") {
        toast.error(errors[0] || t("kb.detail.errors.partialSuccess", { message }))
      } else if (status === "failed") {
        toast.error(errors[0] || t("kb.detail.errors.deleteFailedWithMessage", { message }))
        return
      } else if (status && status !== "success") {
        toast.error(errors[0] || message)
        return
      }

      setDocumentToDelete(null)
      await new Promise(resolve => setTimeout(resolve, 500))
      await onRefresh()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : t("kb.detail.errors.deleteFailed"))
    } finally {
      setIsDeletingDocument(false)
    }
  }

  if (rows.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
        <FileIcon className="h-16 w-16 mb-4 opacity-30" />
        <p className="text-lg font-medium mb-2">{t("kb.detail.uploaded.emptyTitle")}</p>
        <p className="text-sm text-center">{t("kb.detail.uploaded.emptyHint")}</p>
      </div>
    )
  }

  return (
    <>
      <ScrollArea className="h-96">
        <div className="space-y-3">
          {rows.map((document, index) => {
            const inProgress = IN_PROGRESS_STATUSES.has(document.status)
            const canDelete = document.can_delete !== false && document.status !== "uploading"
            return (
              <div key={`${document.filename}-${document.file_id || document.doc_id || index}`} className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50 transition-colors">
                <div className="flex items-center gap-3">
                  <FileIcon className="h-5 w-5 text-blue-500" />
                  <div className="flex-1">
                    <p className="font-medium text-sm">{document.filename.split('/').pop() || document.filename}</p>
                    <p className="text-xs text-muted-foreground">{document.filename}</p>
                    {document.status === "failed" && document.message && (
                      <p className="text-xs text-red-500 mt-0.5">{document.message}</p>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={getStatusBadgeVariant(document.status)} className="text-xs flex items-center gap-1">
                    {inProgress && <Loader2 className="h-3 w-3 animate-spin" />}
                    {t(`kb.detail.uploaded.status.${document.status}`)}
                  </Badge>
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={!canDelete}
                    onClick={() => setDocumentToDelete(document)}
                    title={canDelete ? (t("kb.detail.uploaded.delete") || "Delete document") : t("kb.detail.uploaded.deleteDisabled")}
                  >
                    <Trash2 size={14} />
                  </Button>
                </div>
              </div>
            )
          })}
        </div>
      </ScrollArea>
      <ConfirmDialog
        isOpen={!!documentToDelete}
        onOpenChange={(open) => !open && setDocumentToDelete(null)}
        onConfirm={confirmDeleteDocument}
        isLoading={isDeletingDocument}
        description={t("kb.detail.uploaded.confirmDelete")}
      />
    </>
  )
}
