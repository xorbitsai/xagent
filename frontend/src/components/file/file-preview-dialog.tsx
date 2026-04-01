"use client"

import { useEffect, useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { FileText, ChevronLeft, ChevronRight } from "lucide-react"
import { useApp } from "@/contexts/app-context-chat"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { FileViewer } from "@/components/file/file-viewer"
import { FilePreviewActionButtons } from "@/components/file/file-preview-action-buttons"

interface FilePreviewDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function FilePreviewDialog({ open, onOpenChange }: FilePreviewDialogProps) {
  const { state, dispatch, switchFilePreview } = useApp()
  const { filePreview } = state
  const [viewMode, setViewMode] = useState<'preview' | 'code'>('preview')

  // Extract the base filename from filePath if fileName contains path separators
  // This ensures we use just "image.jpeg" not "web_task_235/output/image.jpeg"
  const baseFileName = filePreview.fileName.includes('/')
    ? filePreview.fileName.split('/').pop() || filePreview.fileName
    : filePreview.fileName

  // Load file content when dialog opens
  useEffect(() => {
    if (open && filePreview.fileId && !filePreview.content && !filePreview.error) {
      const loadFileContent = async () => {
        try {
          const apiUrl = getApiUrl()

          // Check if this is a PPTX file that needs preview conversion
          const isPptxFile = filePreview.fileName.toLowerCase().endsWith('.pptx') ||
            filePreview.fileName.toLowerCase().endsWith('.ppt')
          const isDocxFile = filePreview.fileName.toLowerCase().endsWith('.docx')

          let url: string
          if (isPptxFile) {
            url = `${apiUrl}/api/files/preview/${encodeURIComponent(filePreview.fileId)}`
          } else {
            url = `${apiUrl}/api/files/download/${encodeURIComponent(filePreview.fileId)}`
          }

          const response = await apiRequest(url, {
            cache: 'no-cache',
            headers: {
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache'
            }
          })

          if (response.ok) {
            const contentType = response.headers.get('content-type') || ''
            const mimeType = contentType.split(';')[0].trim().toLowerCase()
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

            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: fileContent, mimeType, error: null }
            })
          } else {
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", mimeType: undefined, error: "Failed to load file" }
            })
          }
        } catch (error) {
          console.error('Network error:', error)

          // Check if it's a CORS error
          if ((error as any)?.name === 'TypeError' && (error as any)?.message?.includes('Failed to fetch')) {
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", mimeType: undefined, error: `CORS error: Unable to access file. This might be a browser caching issue. Try refreshing the page.` }
            })
          } else {
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", mimeType: undefined, error: `Network error: ${(error as any)?.message || 'Unknown error'}` }
            })
          }
        }
      }

      loadFileContent()
    }
  }, [open, filePreview.fileId, filePreview.content, filePreview.error, filePreview.fileName, dispatch])

  const handleDownload = async () => {
    if (filePreview.fileId) {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/files/download/${encodeURIComponent(filePreview.fileId)}`)

        if (!response.ok) {
          throw new Error(`Download failed: ${response.statusText}`)
        }

        // Create blob from response
        const blob = await response.blob()

        // Create download link
        const url = window.URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = filePreview.fileName
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

  const handleOpenInNewWindow = () => {
    if (filePreview.fileId) {
      // Check if this is a PPTX file
      const isPptxFile = filePreview.fileName.toLowerCase().endsWith('.pptx') ||
        filePreview.fileName.toLowerCase().endsWith('.ppt')

      let fileUrl: string
      const apiUrl = getApiUrl()
      if (isPptxFile) {
        fileUrl = `${apiUrl}/api/files/preview/${encodeURIComponent(filePreview.fileId)}`
      } else {
        fileUrl = `${apiUrl}/api/files/public/preview/${encodeURIComponent(filePreview.fileId)}`
      }

      // Open in new window/tab
      window.open(fileUrl, '_blank')
    }
  }

  const handlePreviousFile = () => {
    if (filePreview.availableFiles.length > 1 && filePreview.currentIndex > 0) {
      switchFilePreview(filePreview.currentIndex - 1)
    }
  }

  const handleNextFile = () => {
    if (filePreview.availableFiles.length > 1 && filePreview.currentIndex < filePreview.availableFiles.length - 1) {
      switchFilePreview(filePreview.currentIndex + 1)
    }
  }

  const handleFileSelect = (index: number) => {
    switchFilePreview(index)
  }

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
          <div className="flex flex-col gap-2">
            {/* File title and action buttons */}
            <div className="flex items-center justify-between">
              <DialogTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                {filePreview.fileName}
              </DialogTitle>
              <div className="mr-8">
                <FilePreviewActionButtons
                  viewMode={viewMode}
                  onViewModeChange={setViewMode}
                  fileName={filePreview.fileName}
                  onDownload={handleDownload}
                  onOpenInNewWindow={handleOpenInNewWindow}
                  showText={true}
                />
              </div>
            </div>

            {/* File switching UI - only show when multiple files are available */}
            {filePreview.availableFiles.length > 1 && (
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handlePreviousFile}
                  disabled={filePreview.currentIndex === 0}
                  className="h-8 w-8 p-0"
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>

                {/* File tabs */}
                <div className="flex-1 flex gap-1 overflow-x-auto">
                  {filePreview.availableFiles.map((file, index) => (
                    <Button
                      key={index}
                      variant={index === filePreview.currentIndex ? "default" : "ghost"}
                      size="sm"
                      onClick={() => handleFileSelect(index)}
                      className="text-xs h-8 px-3 min-w-fit"
                      title={file.fileName}
                    >
                      <span className="truncate max-w-32">
                        {file.fileName}
                      </span>
                    </Button>
                  ))}
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleNextFile}
                  disabled={filePreview.currentIndex === filePreview.availableFiles.length - 1}
                  className="h-8 w-8 p-0"
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}
          </div>
        </DialogHeader>

        {/* Main content area with scroll */}
        <div className="flex-1 overflow-auto bg-muted/30 p-4">
          <div className="max-w-4xl mx-auto h-full bg-background rounded shadow-sm border overflow-hidden min-h-[600px]">
            <FileViewer
              fileName={filePreview.fileName}
              fileId={filePreview.fileId}
              content={filePreview.content}
              mimeType={filePreview.mimeType}
              isLoading={filePreview.isLoading}
              error={filePreview.error}
              viewMode={viewMode}
            />
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
