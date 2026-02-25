"use client"

import { useEffect, useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { XIcon, Loader2, FileText, Download, Eye } from "lucide-react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { DocxPreviewRenderer } from "@/components/docx-preview-renderer"

interface StandaloneFilePreviewDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  filePath: string
  fileName: string
}

export function StandaloneFilePreviewDialog({
  open,
  onOpenChange,
  filePath,
  fileName
}: StandaloneFilePreviewDialogProps) {
  const [content, setContent] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { t } = useI18n()

  // Load file content when dialog opens
  useEffect(() => {
    if (open && filePath && !content && !error) {
      const loadFileContent = async () => {
        setIsLoading(true)
        setError(null)

        try {
          const response = await apiRequest(`${getApiUrl()}/api/files/download/${encodeURIComponent(filePath)}`)

          if (response.ok) {
            // For image files, use arrayBuffer to get binary data
            // For text files (HTML, etc.), use text() for proper encoding
            let fileContent
            if (fileName.match(/\.(docx|jpg|jpeg|png|gif|webp|svg)$/i)) {
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
  }, [open, filePath, content, error, t])

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

        return `${attr}="${newUrl}"`
      }
    )
  }

  const handleDownload = async () => {
    if (filePath) {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/files/download/${encodeURIComponent(filePath)}`)

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
            <div className="flex items-center gap-2 mr-8">
              <Button
                variant="outline"
                size="sm"
                onClick={handleDownload}
                className="flex items-center gap-2"
                title={t('files.previewDialog.buttons.download')}
                aria-label={t('files.previewDialog.buttons.download')}
              >
                <Download className="h-4 w-4" />
                {t('files.previewDialog.buttons.download')}
              </Button>
            </div>
          </div>
        </DialogHeader>

        <div className="flex-1 overflow-hidden flex flex-col min-h-0">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="flex flex-col items-center gap-2">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
                <span className="text-sm text-muted-foreground">{t('files.previewDialog.loading')}</span>
              </div>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-full">
              <div className="flex flex-col items-center gap-2 text-center">
                <XIcon className="h-8 w-8 text-destructive" />
                <span className="text-sm text-muted-foreground">{error}</span>
              </div>
            </div>
          ) : (
            <div className="flex-1 overflow-auto bg-muted/30 rounded border">
              {fileName.toLowerCase().endsWith('.docx') ? (
                <DocxPreviewRenderer base64Content={content || ''} />
              ) : fileName.endsWith('.html') || fileName.endsWith('.htm') ? (
                <iframe
                  srcDoc={processHtmlContent(content, filePath)}
                  className="w-full h-full border-0"
                  sandbox="allow-same-origin allow-scripts"
                  title={fileName}
                />
              ) : fileName.match(/\.(jpg|jpeg|png|gif|webp|svg)$/i) ? (
                <div className="flex items-center justify-center h-full p-4">
                  <img
                    src={`data:image/${fileName.split('.').pop()};base64,${content || ''}`}
                    alt={fileName}
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
                  {content || t('files.previewDialog.emptyContent')}
                </pre>
              )}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
