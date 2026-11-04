"use client"

import React from "react"
import { FilePreviewActionButtons } from "@/components/file/file-preview-action-buttons"
import { FilePreviewContent } from "@/components/file/file-preview-content"
import { PreviewSheet } from "@/components/preview-sheet"
import { useApp } from "@/contexts/app-context-chat"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

export function BuildFilePreviewSheet() {
  const { state, closeFilePreview, dispatch } = useApp()
  const { filePreview } = state

  const handleDownload = async () => {
    if (!filePreview.fileId) return

    try {
      const response = await apiRequest(
        `${getApiUrl()}/api/files/download/${encodeURIComponent(filePreview.fileId)}`
      )

      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`)
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = filePreview.fileName || "download"
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error("Failed to download file:", error)
    }
  }

  return (
    <PreviewSheet
      open={filePreview.isOpen}
      onOpenChange={(open) => {
        if (!open) closeFilePreview()
      }}
      title={<>{filePreview.fileName}</>}
      actions={
        <FilePreviewActionButtons
          viewMode={filePreview.viewMode}
          onViewModeChange={(mode) => dispatch({ type: "SET_FILE_PREVIEW_MODE", payload: mode })}
          fileName={filePreview.fileName || ""}
          onDownload={handleDownload}
          showText={true}
        />
      }
    >
      <div className="h-full w-full">
        <FilePreviewContent open={filePreview.isOpen} />
      </div>
    </PreviewSheet>
  )
}
