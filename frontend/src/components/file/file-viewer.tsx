import { Loader2, XIcon } from "lucide-react"
import { DocxPreviewRenderer } from "@/components/file/docx-preview-renderer"
import { MarkdownRenderer } from "@/components/ui/markdown-renderer"
import { useI18n } from "@/contexts/i18n-context"
import { getApiUrl, isHtmlFile, isMarkdownFile } from "@/lib/utils"

interface FileViewerProps {
  fileName: string
  fileId: string
  content: string | null
  mimeType?: string
  isLoading: boolean
  error: string | null
  viewMode: 'preview' | 'code'
}

export function FileViewer({
  fileName,
  fileId,
  content,
  mimeType,
  isLoading,
  error,
  viewMode
}: FileViewerProps) {
  const { t } = useI18n()

  const processHtmlContent = (htmlContent: string, fileId: string) => {
    if (!htmlContent || !fileId) return htmlContent

    const apiUrl = getApiUrl()

    return htmlContent.replace(
      /(src|href)=["']([^"']+)["']/g,
      (match, attr, path) => {
        if (path.match(/^(https?:\/|data:|\/\/|#)/)) return match

        return `${attr}="${apiUrl}/api/files/public/preview/${encodeURIComponent(fileId)}?relative_path=${encodeURIComponent(path)}"`
      }
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="flex flex-col items-center gap-2">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <span className="text-sm text-muted-foreground">{t('files.previewDialog.loading')}</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="flex flex-col items-center gap-2 text-center">
          <XIcon className="h-8 w-8 text-destructive" />
          <span className="text-sm text-muted-foreground">{error}</span>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-auto bg-muted/30 rounded border h-full">
      {(fileName.toLowerCase().endsWith('.pptx') || fileName.toLowerCase().endsWith('.ppt')) && mimeType !== 'application/pdf' ? (
        <iframe
          srcDoc={content || ''}
          className="w-full h-full border-0"
          sandbox="allow-same-origin allow-scripts"
          title={fileName}
        />
      ) : mimeType?.startsWith('image/') || fileName.match(/\.(jpg|jpeg|png|gif|webp|svg)$/i) ? (
        <div className="flex items-center justify-center h-full p-4">
          <img
            src={`data:${mimeType || 'image/png'};base64,${content || ''}`}
            alt={fileName}
            className="max-w-full max-h-full object-contain"
            onError={(e) => {
              console.error('Image load error:', e)
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
      ) : mimeType === 'application/pdf' || fileName.toLowerCase().endsWith('.pdf') ? (
        <div className="flex items-center justify-center h-full p-4">
          <iframe
            src={`data:application/pdf;base64,${content || ''}`}
            className="w-full h-full border-0"
            title={fileName}
          />
        </div>
      ) : mimeType?.includes('wordprocessingml') || fileName.toLowerCase().endsWith('.docx') ? (
        <DocxPreviewRenderer base64Content={content || ''} />
      ) : isHtmlFile(fileName) ? (
        viewMode === 'code' ? (
          <pre className="p-4 text-sm font-mono whitespace-pre-wrap break-words">
            {content || t('files.previewDialog.emptyContent')}
          </pre>
        ) : (
          <iframe
            srcDoc={processHtmlContent(content || '', fileId)}
            className="w-full h-full border-0"
            sandbox="allow-same-origin allow-scripts"
            title={fileName}
          />
        )
      ) : isMarkdownFile(fileName) ? (
        viewMode === 'code' ? (
          <pre className="p-4 text-sm font-mono whitespace-pre-wrap break-words">
            {content || t('files.previewDialog.emptyContent')}
          </pre>
        ) : (
          <div className="p-6">
            <MarkdownRenderer content={content || ''} />
          </div>
        )
      ) : (
        <pre className="p-4 text-sm font-mono whitespace-pre-wrap break-words">
          {content || t('files.previewDialog.emptyContent')}
        </pre>
      )}
    </div>
  )
}
