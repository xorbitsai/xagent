import React from "react"
import { Loader2, XIcon } from "lucide-react"
import { DocxPreviewRenderer } from "@/components/file/docx-preview-renderer"
import { ExcelPreviewRenderer } from "@/components/file/excel-preview-renderer"
import { IsolatedHtmlPreview } from "@/components/file/isolated-html-preview"
import { PptxPreviewRenderer } from "@/components/file/pptx-preview-renderer"
import { MarkdownRenderer } from "@/components/ui/markdown-renderer"
import { useI18n } from "@/contexts/i18n-context"
import { useFileAccess } from "@/contexts/file-access-context"
import { withPreviewReferrerPolicy } from "@/lib/html-preview-document"
import {
  isHtmlFile,
  isMarkdownFile,
  isCsvFile,
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
  filesDisabled?: boolean
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
  viewMode,
  filesDisabled = false,
}: FileViewerProps) {
  const { t } = useI18n()
  const fileAccess = useFileAccess()
  const videoMimeType = getVideoMimeType(fileName, mimeType)
  const audioMimeType = getAudioMimeType(fileName, mimeType, content)
  const base64Content = getBase64Payload(content)
  const blobMimeType = mimeType === 'application/pdf' || fileName.toLowerCase().endsWith('.pdf')
    ? 'application/pdf' : videoMimeType
  const [mediaObjectUrl, setMediaObjectUrl] = React.useState<string | null>(null)
  const [preparedHtml, setPreparedHtml] = React.useState<{
    content: typeof content; fileId: string; fileName: string; policy: typeof fileAccess; html: string
  } | null>(null)
  const htmlPreviewContent = preparedHtml?.content === content && preparedHtml.fileId === fileId
    && preparedHtml.fileName === fileName && preparedHtml.policy === fileAccess ? preparedHtml.html : ""

  React.useEffect(() => {
    if (!blobMimeType || !base64Content) {
      setMediaObjectUrl(null)
      return
    }

    let active = true
    let objectUrl: string | null = null
    setMediaObjectUrl(null)
    async function decode() {
      try {
        // Decode locally: fetch(data:) is blocked by connect-src. Keep chunks
        // aligned to base64 quanta and yield so large previews don't monopolize
        // the UI thread or allocate one huge intermediate binary string.
        const source = base64Content!.replace(/\s/g, "")
        const parts: BlobPart[] = []
        const chunkSize = 64 * 1024
        for (let offset = 0; offset < source.length; offset += chunkSize) {
          if (!active) return
          const decoded = atob(source.slice(offset, offset + chunkSize))
          const bytes = new Uint8Array(decoded.length)
          for (let i = 0; i < decoded.length; i++) bytes[i] = decoded.charCodeAt(i)
          parts.push(bytes)
          if (offset + chunkSize < source.length) {
            await new Promise<void>((resolve) => setTimeout(resolve, 0))
          }
        }
        if (!active) return
        objectUrl = URL.createObjectURL(new Blob(parts, { type: blobMimeType! }))
        setMediaObjectUrl(objectUrl)
      } catch (error) {
        if (!active) return
        console.error("Media preview decode error:", error)
        setMediaObjectUrl(null)
      }
    }
    void decode()

    return () => {
      active = false
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl)
      }
    }
  }, [base64Content, blobMimeType])

  const processHtmlContent = React.useCallback((
    htmlContent: string,
    fileId: string,
    mapUrl: (url: string) => string = (url) => url,
  ) => {
    const rewritten = htmlContent.replace(
      /(src|href)=["']([^"']+)["']/g,
      (match, attr, path) => {
        if (path.match(/^(https?:\/|data:|\/\/|#)/)) return match
        // Root-relative API URLs must remain on the app/API origin when the
        // HTML is delivered to a separate preview host (default API_URL="").
        const absolute = (url: string) => mapUrl(new URL(url, window.location.origin).href)
        if (path.startsWith("file:")) {
          const fileRef = path.replace(/^file:/, "")
          return `${attr}="${absolute(fileAccess.inlinePreviewUrl(fileRef))}"`
        }
        if (path.startsWith("/api/files/public/preview/")) {
          const requestedFileId = path.slice("/api/files/public/preview/".length)
          return `${attr}="${absolute(fileAccess.inlinePreviewUrl(requestedFileId))}"`
        }

        return `${attr}="${absolute(fileAccess.relativePreviewUrl(fileId, path))}"`
      }
    )
    // srcDoc subresources do not inherit the iframe navigation's
    // referrerPolicy. Set the document policy before any asset is parsed.
    return withPreviewReferrerPolicy(rewritten)
  }, [fileAccess])

  React.useEffect(() => {
    // Keep the initial SSR/client render blank; prepare before assigning srcDoc.
    if (!isHtmlFile(fileName) || viewMode !== "preview") return
    const prepared = (html: string) => ({ content, fileId, fileName, policy: fileAccess, html })
    const authorize = fileAccess.authorizeBrowserUrl
    if (!content || !authorize) {
      setPreparedHtml(prepared(content ? processHtmlContent(content, fileId) : ""))
      return
    }
    // Subresources load without headers, so a policy that needs one signs
    // each distinct asset URL with a short-lived file ticket first. An asset
    // that cannot be signed keeps its plain URL and simply fails to load.
    //
    // Signed once per preparation, deliberately not refreshed: tickets live
    // XAGENT_FILE_STREAM_TICKET_TTL_SECONDS (default 10 min), so assets the
    // document requests only after that -- lazy-loaded images, in-document
    // media range requests, in-preview navigations -- fail until the preview
    // is reopened (which re-signs). Re-signing on a timer would mean
    // re-delivering srcDoc, which reloads the sandboxed cross-origin document
    // and loses its scroll, form and playback state every TTL period; the
    // parent also cannot see subresource failures inside that sandbox to
    // refresh only when needed. Eager assets, the common case, load at first
    // render. See docs/html-preview-security.md.
    let active = true
    const assetUrls = new Set<string>()
    processHtmlContent(content, fileId, (url) => {
      assetUrls.add(url)
      return url
    })
    void Promise.all(Array.from(assetUrls, async (url) => {
      try {
        return [url, await authorize(url)] as const
      } catch {
        return [url, url] as const
      }
    })).then((entries) => {
      if (!active) return
      const signed = new Map<string, string>(entries)
      setPreparedHtml(prepared(processHtmlContent(content, fileId, (url) => signed.get(url) ?? url)))
    })
    return () => {
      active = false
    }
  }, [content, fileId, fileName, viewMode, processHtmlContent, fileAccess])

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
            src={mediaObjectUrl || undefined}
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
            src={mediaObjectUrl || undefined}
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
          <IsolatedHtmlPreview
            html={htmlPreviewContent}
            className="w-full h-full border-0"
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
            <MarkdownRenderer content={content || ''} filesDisabled={filesDisabled} />
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
