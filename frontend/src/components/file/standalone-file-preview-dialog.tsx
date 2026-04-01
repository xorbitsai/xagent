"use client"

import { useEffect, useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { FileText } from "lucide-react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { FileViewer } from "@/components/file/file-viewer"
import { FilePreviewActionButtons } from "@/components/file/file-preview-action-buttons"

interface StandaloneFilePreviewDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  fileId: string
  fileName: string
}

export function StandaloneFilePreviewDialog({
  open,
  onOpenChange,
  fileId,
  fileName
}: StandaloneFilePreviewDialogProps) {
  const [content, setContent] = useState("")
  const [mimeType, setMimeType] = useState<string | undefined>()
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<'preview' | 'code'>('preview')
  const { t } = useI18n()

  // Load file content when dialog opens
  useEffect(() => {
    if (open && fileId && !content && !error) {
      const loadFileContent = async () => {
        setIsLoading(true)
        setError(null)

        try {
          const isPptxFile = fileName.toLowerCase().endsWith('.pptx') || fileName.toLowerCase().endsWith('.ppt')
          const url = isPptxFile
            ? `${getApiUrl()}/api/files/preview/${encodeURIComponent(fileId)}`
            : `${getApiUrl()}/api/files/download/${encodeURIComponent(fileId)}`

          const response = await apiRequest(url)

          if (response.ok) {
            const contentType = response.headers.get('content-type') || ''
            const mimeType = contentType.split(';')[0].trim().toLowerCase()

            // Determine if the response is binary based on MIME type
            // PPTX preview might return application/pdf or text/html
            const isTextType = mimeType.startsWith('text/') || mimeType === 'application/json' || mimeType === 'application/xml' || mimeType === 'application/javascript'

            let fileContent
            if (!isTextType) {
              const arrayBuffer = await response.arrayBuffer()

              // Convert binary data to base64 using chunks to avoid stack overflow
              const chunkSize = 16384; // 16KB chunks
              const bytes = new Uint8Array(arrayBuffer)
              let binary = ''

              for (let i = 0; i < bytes.length; i += chunkSize) {
                const chunk = bytes.slice(i, i + chunkSize)
                binary += String.fromCharCode.apply(null, Array.from(chunk))
              }

              fileContent = btoa(binary)
            } else {
              // For text files (HTML, etc.), use text() for proper encoding
              fileContent = await response.text()
            }

            setContent(fileContent)
            setMimeType(mimeType)
            setError(null)
          } else {
            setError(t('files.previewDialog.errors.loadFailed'))
          }
        } catch (error) {
          // Check if it's a CORS error
          if ((error as any)?.name === 'TypeError' && (error as any)?.message?.includes('Failed to fetch')) {
            setError(t('files.previewDialog.errors.cors'))
          } else {
            const msg = (error as any)?.message || t('common.errors.unknown')
            setError(t('files.previewDialog.errors.networkErrorWithMsg', { msg }))
          }
        } finally {
          setIsLoading(false)
        }
      }

      loadFileContent()
    }
  }, [open, fileId, content, error, t])

  const handleDownload = async () => {
    if (fileId) {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/files/download/${encodeURIComponent(fileId)}`)

        if (!response.ok) {
          throw new Error(`Download failed: ${response.statusText}`)
        }

        // Create blob from response
        const blob = await response.blob()

        // Create download link
        const url = window.URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = fileName
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)

        // Clean up blob URL
        window.URL.revokeObjectURL(url)
      } catch (error) {
        console.error('Failed to download file:', error)
        // You might want to show an error message to the user here
      }
    }
  }

  // Reset state when dialog closes
  useEffect(() => {
    if (!open) {
      setContent("")
      setMimeType(undefined)
      setError(null)
      setIsLoading(false)
    }
  }, [open])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="fixed inset-0 m-0 p-0 max-w-none max-h-none w-screen h-screen rounded-none border-0 flex flex-col top-0 left-0 translate-x-0 translate-y-0"
        style={{
          width: '100vw',
          height: '100vh',
          maxWidth: 'none',
          maxHeight: 'none',
          top: '0',
          left: '0',
          transform: 'none'
        }}
        showCloseButton={true}
      >
        <DialogHeader className="flex-shrink-0 bg-background/80 backdrop-blur-sm border-b p-4">
          <div className="flex items-center justify-between">
            <DialogTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              {fileName}
            </DialogTitle>
            <div className="mr-8">
              <FilePreviewActionButtons
                viewMode={viewMode}
                onViewModeChange={setViewMode}
                fileName={fileName}
                onDownload={handleDownload}
                showText={true}
              />
            </div>
          </div>
        </DialogHeader>

        <div className="flex-1 overflow-hidden flex flex-col min-h-0">
          <FileViewer
            fileName={fileName}
            fileId={fileId}
            content={content}
            mimeType={mimeType}
            isLoading={isLoading}
            error={error}
            viewMode={viewMode}
          />
        </div>
      </DialogContent>
    </Dialog>
  )
}
