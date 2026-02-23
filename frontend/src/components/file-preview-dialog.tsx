"use client"

import { useEffect } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { XIcon, Loader2, FileText, Download, ChevronLeft, ChevronRight, ExternalLink } from "lucide-react"
import { useApp } from "@/contexts/app-context"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { DocxPreviewRenderer } from "@/components/docx-preview-renderer"

interface FilePreviewDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function FilePreviewDialog({ open, onOpenChange }: FilePreviewDialogProps) {
  const { state, dispatch, switchFilePreview } = useApp()
  const { filePreview } = state
  const { t } = useI18n()


  // Load file content when dialog opens
  useEffect(() => {
    if (open && filePreview.filePath && !filePreview.content && !filePreview.error) {
      const loadFileContent = async () => {
        try {
          const apiUrl = getApiUrl()

          // Check if this is a PPTX file that needs preview conversion
          const isPptxFile = filePreview.fileName.toLowerCase().endsWith('.pptx') ||
                           filePreview.fileName.toLowerCase().endsWith('.ppt')
          const isDocxFile = filePreview.fileName.toLowerCase().endsWith('.docx')

          let url: string
          if (isPptxFile) {
            // Extract task ID from file path for preview endpoint
            // Format: web_task_103/output/file.pptx
            const pathMatch = filePreview.filePath.match(/web_task_(\d+)/)
            if (pathMatch && pathMatch[1]) {
              const taskId = pathMatch[1]
              const absolutePath = filePreview.filePath.startsWith('/')
                ? filePreview.filePath.substring(1)
                : filePreview.filePath
              url = `${apiUrl}/api/files/preview/${taskId}/${encodeURIComponent(absolutePath)}`
            } else {
              // Fallback to download endpoint for files without task ID
              url = `${apiUrl}/api/files/download/${encodeURIComponent(filePreview.filePath)}`
            }
          } else {
            url = `${apiUrl}/api/files/download/${encodeURIComponent(filePreview.filePath)}`
          }

          const response = await apiRequest(url, {
            cache: 'no-cache',
            headers: {
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache'
            }
          })

          if (response.ok) {
            // For PPTX files (when preview endpoint returns HTML), use text()
            // For binary files (images, PDFs), use arrayBuffer to get binary data
            // For text files (HTML, etc.), use text() for proper encoding
            let fileContent
            if (isPptxFile) {
              // PPTX preview endpoint returns HTML
              fileContent = await response.text()
            } else if (isDocxFile || filePreview.fileName.match(/\.(jpg|jpeg|png|gif|webp|svg|pdf)$/i)) {
              const arrayBuffer = await response.arrayBuffer()
              console.log('Debug: ArrayBuffer size:', arrayBuffer.byteLength)

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
              payload: { content: fileContent, error: null }
            })
          } else {
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", error: "Failed to load file" }
            })
          }
        } catch (error) {
          console.error('Network error:', error)

          // Check if it's a CORS error
          if ((error as any)?.name === 'TypeError' && (error as any)?.message?.includes('Failed to fetch')) {
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", error: `CORS error: Unable to access file. This might be a browser caching issue. Try refreshing the page.` }
            })
          } else {
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", error: `Network error: ${(error as any)?.message || 'Unknown error'}` }
            })
          }
        }
      }

      loadFileContent()
    }
  }, [open, filePreview.filePath, filePreview.content, filePreview.error, filePreview.fileName, dispatch])

  // Convert relative paths in HTML to absolute paths
  const processHtmlContent = (htmlContent: string, filePath: string) => {
    if (!htmlContent || !filePath) return htmlContent

    // Get the directory path of the HTML file
    const dirPath = filePath.substring(0, filePath.lastIndexOf('/'))
    const apiUrl = getApiUrl()

    // Extract task ID from file path for public preview endpoint
    // Format: web_task_103/output/file.html
    let taskId: string | null = null
    const pathMatch = filePath.match(/web_task_(\d+)/)
    if (pathMatch && pathMatch[1]) {
      taskId = pathMatch[1]
    }

    // Debug log
    console.log('[FilePreviewDialog processHtmlContent] filePath:', filePath)
    console.log('[FilePreviewDialog processHtmlContent] taskId:', taskId)

    // Replace relative paths for images, scripts, links, etc.
    return htmlContent.replace(
      /(src|href)=["']([^"']+)["']/g,
      (match, attr, path) => {
        // Skip if it's already an absolute URL, data URL, or has a protocol
        if (path.match(/^(https?:\/|data:|\/\/|#)/)) {
          return match
        }

        // Convert relative path to absolute path
        const absolutePath = path.startsWith('/')
          ? path.substring(1) // Remove leading slash
          : `${dirPath}/${path}`

        // Use public preview endpoint if task ID is available
        const newUrl = taskId
          ? `${apiUrl}/api/files/public/preview/${taskId}/${encodeURIComponent(absolutePath)}`
          : `${apiUrl}/api/files/download/${encodeURIComponent(absolutePath)}`

        console.log(`[FilePreviewDialog] Replacing ${path} -> ${newUrl}`)

        return `${attr}="${newUrl}"`
      }
    )
  }

  const handleDownload = async () => {
    if (filePreview.filePath) {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/files/download/${encodeURIComponent(filePreview.filePath)}`)

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
    if (filePreview.filePath) {
      // Extract task ID from file path for public preview
      let taskId: string | null = null
      const pathMatch = filePreview.filePath.match(/web_task_(\d+)/)
      if (pathMatch && pathMatch[1]) {
        taskId = pathMatch[1]
      }

      // Check if this is a PPTX file
      const isPptxFile = filePreview.fileName.toLowerCase().endsWith('.pptx') ||
                        filePreview.fileName.toLowerCase().endsWith('.ppt')

      // Construct URL based on file type and task ID availability
      let fileUrl: string
      if (taskId) {
        const apiUrl = getApiUrl()
        const absolutePath = filePreview.filePath.startsWith('/')
          ? filePreview.filePath.substring(1)
          : filePreview.filePath

        if (isPptxFile) {
          // Use preview endpoint for PPTX files (returns HTML)
          fileUrl = `${apiUrl}/api/files/preview/${taskId}/${encodeURIComponent(absolutePath)}`
        } else {
          // Use public preview endpoint for other files
          fileUrl = `${apiUrl}/api/files/public/preview/${taskId}/${encodeURIComponent(absolutePath)}`
        }
      } else {
        // Fallback to download endpoint (requires authentication)
        const apiUrl = getApiUrl()
        fileUrl = `${apiUrl}/api/files/download/${encodeURIComponent(filePreview.filePath)}`
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
              <div className="flex items-center gap-2 mr-8">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleOpenInNewWindow}
                  className="flex items-center gap-2"
                  title={t('files.previewDialog.buttons.openInNewWindow')}
                >
                  <ExternalLink className="h-4 w-4" />
                  {t('files.previewDialog.buttons.openInNewWindow')}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDownload}
                  className="flex items-center gap-2"
                  title={t('files.previewDialog.buttons.download')}
                >
                  <Download className="h-4 w-4" />
                  {t('files.previewDialog.buttons.download')}
                </Button>
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

        <div className="flex-1 overflow-hidden flex flex-col min-h-0">
          {filePreview.isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="flex flex-col items-center gap-2">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
                <span className="text-sm text-muted-foreground">{t('files.previewDialog.loading')}</span>
              </div>
            </div>
          ) : filePreview.error ? (
            <div className="flex items-center justify-center h-full">
              <div className="flex flex-col items-center gap-2 text-center">
                <XIcon className="h-8 w-8 text-destructive" />
                <span className="text-sm text-muted-foreground">{filePreview.error}</span>
              </div>
            </div>
          ) : (
            <div className="flex-1 overflow-auto bg-muted/30 rounded border">
              {/* PPTX files - display as converted HTML */}
              {filePreview.fileName.toLowerCase().endsWith('.pptx') || filePreview.fileName.toLowerCase().endsWith('.ppt') ? (
                <iframe
                  srcDoc={filePreview.content || ''}
                  className="w-full h-full border-0"
                  sandbox="allow-same-origin allow-scripts"
                  title={filePreview.fileName}
                />
              ) : filePreview.fileName.toLowerCase().endsWith('.docx') ? (
                <DocxPreviewRenderer base64Content={filePreview.content || ''} />
              ) : filePreview.fileName.endsWith('.html') || filePreview.fileName.endsWith('.htm') ? (
                <iframe
                  srcDoc={processHtmlContent(filePreview.content, filePreview.filePath)}
                  className="w-full h-full border-0"
                  sandbox="allow-same-origin allow-scripts"
                  title={filePreview.fileName}
                />
              ) : filePreview.fileName.toLowerCase().endsWith('.pdf') ? (
                <div className="flex items-center justify-center h-full p-4">
                  <iframe
                    src={`data:application/pdf;base64,${filePreview.content || ''}`}
                    className="w-full h-full border-0"
                    title={filePreview.fileName}
                  />
                </div>
              ) : filePreview.fileName.match(/\.(jpg|jpeg|png|gif|webp|svg)$/i) ? (
                <div className="flex items-center justify-center h-full p-4">
                  <img
                    src={`data:image/${filePreview.fileName.split('.').pop()};base64,${filePreview.content || ''}`}
                    alt={filePreview.fileName}
                    className="max-w-full max-h-full object-contain"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none'
                      const fallback = e.currentTarget.nextElementSibling as HTMLElement
                      if (fallback) fallback.style.display = 'flex'
                    }}
                  />
                  <div className="hidden flex-col items-center justify-center h-full text-muted-foreground">
                    <span>{t('files.previewDialog.imageError.title')}</span>
                    <span className="text-sm">{t('files.previewDialog.imageError.hint')}</span>
                  </div>
                </div>
              ) : (
                <pre className="p-4 text-sm font-mono whitespace-pre-wrap break-words">
                  {filePreview.content || t('files.previewDialog.emptyContent')}
                </pre>
              )}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
