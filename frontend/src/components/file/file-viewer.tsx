import React from "react"
import { Loader2, XIcon } from "lucide-react"
import { DocxPreviewRenderer } from "@/components/file/docx-preview-renderer"
import { ExcelPreviewRenderer } from "@/components/file/excel-preview-renderer"
import { PptxPreviewRenderer } from "@/components/file/pptx-preview-renderer"
import { MarkdownRenderer } from "@/components/ui/markdown-renderer"
import { useI18n } from "@/contexts/i18n-context"
import {
  getApiUrl,
  getFilePublicPreviewUrl,
  getFileRelativePreviewUrl,
  isHtmlFile,
  isMarkdownFile,
  isCsvFile,
  withPublicAccessToken,
} from "@/lib/utils"

const VIDEO_EXTENSION_MIME_TYPES: Record<string, string> = {
  mp4: "video/mp4",
  m4v: "video/mp4",
  mov: "video/quicktime",
  webm: "video/webm",
  mpeg: "video/mpeg",
  mpg: "video/mpeg",
}

const AUDIO_EXTENSION_MIME_TYPES: Record<string, string> = {
  mp3: "audio/mpeg",
  wav: "audio/wav",
  ogg: "audio/ogg",
  opus: "audio/ogg",
  flac: "audio/flac",
  m4a: "audio/mp4",
  mp4: "audio/mp4",
  aac: "audio/aac",
  webm: "audio/webm",
}

interface FileViewerProps {
  fileName: string
  fileId: string
  content: string | null
  mimeType?: string
  isLoading: boolean
  error: string | null
  viewMode: 'preview' | 'code'
}

function getFileExtension(fileName: string): string {
  return fileName.split('.').pop()?.toLowerCase() || ''
}

function getVideoMimeType(fileName: string, mimeType: string | undefined): string | null {
  if (mimeType?.startsWith("video/")) {
    return mimeType
  }

  return VIDEO_EXTENSION_MIME_TYPES[getFileExtension(fileName)] || null
}

function getBase64HeaderBytes(content: string | null, byteCount = 16): number[] {
  if (!content) return []

  const prefixMatch = content.match(/^data:[^,]+,/)
  const prefixLength = prefixMatch ? prefixMatch[0].length : 0
  const compact = content
    .slice(prefixLength, prefixLength + byteCount * 8)
    .replace(/\s/g, '')
  if (!compact) return []

  try {
    const binary = atob(compact.slice(0, Math.ceil(byteCount / 3) * 4))
    return Array.from(binary.slice(0, byteCount), char => char.charCodeAt(0))
  } catch {
    return []
  }
}

function getBase64Payload(content: string | null): string {
  if (!content) return ''

  const prefixMatch = content.match(/^data:[^,]+,/)
  return prefixMatch ? content.slice(prefixMatch[0].length) : content
}

function inferAudioMimeTypeFromBase64(content: string | null): string | null {
  const bytes = getBase64HeaderBytes(content)
  if (bytes.length < 4) return null

  if (
    bytes[0] === 0x49
    && bytes[1] === 0x44
    && bytes[2] === 0x33
  ) {
    return "audio/mpeg"
  }

  if (bytes[0] === 0xff && (bytes[1] & 0xe0) === 0xe0) {
    return "audio/mpeg"
  }

  if (
    bytes.length >= 12
    && bytes[0] === 0x52
    && bytes[1] === 0x49
    && bytes[2] === 0x46
    && bytes[3] === 0x46
    && bytes[8] === 0x57
    && bytes[9] === 0x41
    && bytes[10] === 0x56
    && bytes[11] === 0x45
  ) {
    return "audio/wav"
  }

  if (
    bytes[0] === 0x4f
    && bytes[1] === 0x67
    && bytes[2] === 0x67
    && bytes[3] === 0x53
  ) {
    return "audio/ogg"
  }

  if (
    bytes[0] === 0x66
    && bytes[1] === 0x4c
    && bytes[2] === 0x61
    && bytes[3] === 0x43
  ) {
    return "audio/flac"
  }

  if (
    bytes.length >= 8
    && bytes[4] === 0x66
    && bytes[5] === 0x74
    && bytes[6] === 0x79
    && bytes[7] === 0x70
  ) {
    return "audio/mp4"
  }

  return null
}

function getAudioMimeType(fileName: string, mimeType: string | undefined, content: string | null): string | null {
  if (mimeType?.startsWith("audio/")) {
    return mimeType
  }

  const extensionMimeType = AUDIO_EXTENSION_MIME_TYPES[getFileExtension(fileName)]
  if (extensionMimeType) {
    return extensionMimeType
  }

  return inferAudioMimeTypeFromBase64(content)
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
  const videoMimeType = getVideoMimeType(fileName, mimeType)
  const audioMimeType = getAudioMimeType(fileName, mimeType, content)
  const base64Content = getBase64Payload(content)
  const [videoObjectUrl, setVideoObjectUrl] = React.useState<string | null>(null)

  React.useEffect(() => {
    if (!videoMimeType || !base64Content) {
      setVideoObjectUrl(null)
      return
    }

    let active = true
    let objectUrl: string | null = null

    fetch(`data:${videoMimeType};base64,${base64Content.replace(/\s/g, "")}`)
      .then((response) => response.blob())
      .then((blob) => {
        objectUrl = URL.createObjectURL(blob)
        if (!active) {
          URL.revokeObjectURL(objectUrl)
          return
        }
        setVideoObjectUrl(objectUrl)
      })
      .catch((error) => {
        if (!active) return
        console.error("Video preview decode error:", error)
        setVideoObjectUrl(null)
      })

    return () => {
      active = false
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl)
      }
    }
  }, [base64Content, videoMimeType])

  const processHtmlContent = (htmlContent: string, fileId: string) => {
    if (!htmlContent || !fileId) return htmlContent

    const apiUrl = getApiUrl()

    return htmlContent.replace(
      /(src|href)=["']([^"']+)["']/g,
      (match, attr, path) => {
        if (path.match(/^(https?:\/|data:|\/\/|#)/)) return match
        if (path.startsWith("file:")) {
          const fileRef = path.replace(/^file:/, "")
          return `${attr}="${getFilePublicPreviewUrl(fileRef, apiUrl)}"`
        }
        if (path.startsWith("/api/files/public/preview/")) {
          return `${attr}="${withPublicAccessToken(`${apiUrl}${path}`)}"`
        }

        return `${attr}="${getFileRelativePreviewUrl(fileId, path, apiUrl)}"`
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
      {fileName.toLowerCase().endsWith('.pptx') ? (
        <PptxPreviewRenderer base64Content={base64Content} fileId={fileId} />
      ) : mimeType?.startsWith('image/') || fileName.match(/\.(jpg|jpeg|png|gif|webp|svg)$/i) ? (
        <div className="flex items-center justify-center h-full p-4">
          <img
            src={`data:${mimeType || 'image/png'};base64,${base64Content}`}
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
            src={`data:application/pdf;base64,${base64Content}`}
            className="w-full h-full border-0"
            title={fileName}
          />
        </div>
      ) : videoMimeType ? (
        <div className="flex h-full items-center justify-center bg-black/95 p-4">
          <video
            controls
            playsInline
            preload="metadata"
            src={videoObjectUrl || undefined}
            className="max-h-full max-w-full rounded border border-border/30 bg-black shadow-sm"
            aria-label={fileName}
            title={fileName}
          />
        </div>
      ) : audioMimeType ? (
        <div className="flex h-full items-center justify-center p-6">
          <div className="w-full max-w-2xl rounded-lg border bg-background p-6 shadow-sm">
            <div className="mb-4 truncate text-sm font-medium text-foreground" title={fileName}>
              {fileName}
            </div>
            <audio
              controls
              preload="metadata"
              src={`data:${audioMimeType};base64,${base64Content}`}
              className="w-full"
              aria-label={fileName}
              title={fileName}
            />
          </div>
        </div>
      ) : mimeType?.includes('wordprocessingml') || fileName.toLowerCase().endsWith('.docx') ? (
        <DocxPreviewRenderer base64Content={base64Content} />
      ) : mimeType?.includes('spreadsheetml') || fileName.toLowerCase().endsWith('.xlsx') || fileName.toLowerCase().endsWith('.csv') ? (
        viewMode === 'code' && isCsvFile(fileName) ? (
          <pre className="p-4 text-sm font-mono whitespace-pre-wrap break-words">
            {(() => {
              const c = base64Content;
              if (!c) return t('files.previewDialog.emptyContent');
              if (/^[A-Za-z0-9+/=]+$/.test(c.replace(/\s/g, ''))) {
                try {
                  return decodeURIComponent(escape(atob(c)));
                } catch {
                  return c;
                }
              }
              return c;
            })()}
          </pre>
        ) : (
          <ExcelPreviewRenderer base64Content={base64Content} />
        )
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
