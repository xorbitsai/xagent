"use client"

import { useEffect } from "react"
import { useApp } from "@/contexts/app-context-chat"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { Loader2, XIcon } from "lucide-react"
import { DocxPreviewRenderer } from "@/components/docx-preview-renderer"

interface FilePreviewContentProps {
  open: boolean
}

export function FilePreviewContent({ open }: FilePreviewContentProps) {
  const { state, dispatch } = useApp()
  const { filePreview } = state
  const { t } = useI18n()

  // Load file content when the preview is open within container
  useEffect(() => {
    if (open && filePreview.filePath && !filePreview.content && !filePreview.error) {
      const loadFileContent = async () => {
        try {
          const apiUrl = getApiUrl()

          // PPTX files are converted to PDF by backend, treat as PDF
          const isPptx = filePreview.fileName.match(/\.pptx$/i)
          const isPdf = isPptx || filePreview.fileName.match(/\.pdf$/i)
          const isDocx = filePreview.fileName.match(/\.docx$/i)

          const url = `${apiUrl}/api/files/download/${encodeURIComponent(filePreview.filePath)}`

          const response = await apiRequest(url, {
            cache: 'no-cache',
            headers: {
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache'
            }
          })

          if (response.ok) {
            let fileContent
            if (isDocx || isPdf || filePreview.fileName.match(/\.(jpg|jpeg|png|gif|webp|svg)$/i)) {
              const arrayBuffer = await response.arrayBuffer()

              const chunkSize = 16384
              const bytes = new Uint8Array(arrayBuffer)
              let binary = ''

              for (let i = 0; i < bytes.length; i += chunkSize) {
                const chunk = bytes.slice(i, i + chunkSize)
                binary += String.fromCharCode.apply(null, Array.from(chunk))
              }

              fileContent = btoa(binary)
            } else {
              fileContent = await response.text()
            }

            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: fileContent, error: null }
            })
          } else {
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", error: t('files.previewDialog.errors.loadFailed') }
            })
          }
        } catch (error) {
          if ((error as any)?.name === 'TypeError' && (error as any)?.message?.includes('Failed to fetch')) {
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", error: t('files.previewDialog.errors.cors') }
            })
          } else {
            const msg = (error as any)?.message || t('common.errors.unknown')
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", error: t('files.previewDialog.errors.networkErrorWithMsg', { msg }) }
            })
          }
        }
      }

      loadFileContent()
    }
  }, [open, filePreview.filePath, filePreview.content, filePreview.error, dispatch, t, filePreview.fileName])

  const processHtmlContent = (htmlContent: string, filePath: string) => {
    if (!htmlContent || !filePath) return htmlContent

    const dirPath = filePath.substring(0, filePath.lastIndexOf('/'))
    const apiUrl = getApiUrl()

    // Extract task_id from filePath (e.g., "web_task_78/output/file.html" -> "78")
    const taskIdMatch = filePath.match(/web_task_(\d+)/)
    const taskId = taskIdMatch ? taskIdMatch[1] : null

    return htmlContent.replace(
      /(src|href)=["']([^"']+)["']/g,
      (match, attr, path) => {
        if (path.match(/^(https?:\/|data:|\/\/|#)/)) return match

        const absolutePath = path.startsWith('/') ? path.substring(1) : `${dirPath}/${path}`

        // Use public preview API if taskId is available, otherwise use download API
        if (taskId) {
          return `${attr}="${apiUrl}/api/files/public/preview/${taskId}/${encodeURIComponent(absolutePath)}"`
        } else {
          return `${attr}="${apiUrl}/api/files/download/${encodeURIComponent(absolutePath)}"`
        }
      }
    )
  }

  return (
    <div className="w-full h-full">
      <div className="flex-1 overflow-hidden flex flex-col min-h-0 h-full">
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
            {filePreview.fileName.toLowerCase().endsWith('.docx') ? (
              <DocxPreviewRenderer base64Content={filePreview.content || ''} />
            ) : filePreview.fileName.endsWith('.html') || filePreview.fileName.endsWith('.htm') ? (
              <iframe
                srcDoc={processHtmlContent(filePreview.content, filePreview.filePath)}
                className="w-full h-full border-0"
                sandbox="allow-same-origin allow-scripts"
                title={filePreview.fileName}
              />
            ) : filePreview.fileName.toLowerCase().endsWith('.pdf') || filePreview.fileName.toLowerCase().endsWith('.pptx') ? (
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
    </div>
  )
}
