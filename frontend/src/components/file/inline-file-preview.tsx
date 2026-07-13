import React, { useEffect, useState } from 'react'
import { FileText, Loader2, Volume2 } from 'lucide-react'

import { DocxPreviewRenderer } from '@/components/file/docx-preview-renderer'
import { ExcelPreviewRenderer } from '@/components/file/excel-preview-renderer'
import { PptxPreviewRenderer } from '@/components/file/pptx-preview-renderer'
import { apiRequest } from '@/lib/api-wrapper'
import { cn, getApiUrl } from '@/lib/utils'
import {
  arrayBufferToBase64,
  getInlineFileDownloadUrl,
  getInlineFilePreviewKind,
  getInlineFilePreviewUrl,
  getPreviewUrlTrust,
  isPreviewableInlineFileKind,
  resolveInlineFileId,
  type InlineFilePreviewSource,
} from './inline-file-preview-utils'

type InlineFilePreviewProps = {
  source: InlineFilePreviewSource
  className?: string
  imageClassName?: string
  onFileClick?: (filePath: string, fileName: string) => void
  openLabel?: string
  loadErrorText?: string
}

const fileNameFromSource = (source: InlineFilePreviewSource) =>
  source.filename || source.fileId?.split('/').pop() || 'artifact'

const DEFAULT_OPEN_LABEL = 'Open'
const DEFAULT_LOAD_ERROR_TEXT = 'Failed to load preview.'

function InlineImagePreview({
  source,
  previewUrl,
  filename,
  imageClassName,
  onFileClick,
}: {
  source: InlineFilePreviewSource
  previewUrl: string
  filename: string
  imageClassName?: string
  onFileClick?: (filePath: string, fileName: string) => void
}) {
  const apiUrl = getApiUrl()
  const shouldFallback = Boolean(source.fileId)
  const [resolvedUrl, setResolvedUrl] = useState(shouldFallback ? '' : previewUrl)

  useEffect(() => {
    let objectUrl: string | null = null
    let isCancelled = false

    setResolvedUrl(shouldFallback ? '' : previewUrl)

    const runFallback = async () => {
      if (!shouldFallback) return
      try {
        const response = await apiRequest(
          `${apiUrl}/api/files/preview/${encodeURIComponent(source.fileId!)}`,
          {
            cache: 'no-cache',
            headers: {
              'Cache-Control': 'no-cache',
              Pragma: 'no-cache',
            },
          }
        )
        if (isCancelled) return
        if (!response.ok) {
          // Last-resort fallback: try the public preview URL when the
          // authenticated route fails. On Agent Builder surfaces that
          // route may also require auth, but a spinner with no recovery
          // path is worse than attempting the anonymous endpoint.
          setResolvedUrl(previewUrl)
          return
        }
        const blob = await response.blob()
        if (isCancelled) return
        objectUrl = URL.createObjectURL(blob)
        setResolvedUrl(objectUrl)
      } catch {
        if (!isCancelled) {
          // See comment above: public preview is best-effort after auth errors.
          setResolvedUrl(previewUrl)
        }
      }
    }

    void runFallback()

    return () => {
      isCancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [apiUrl, previewUrl, shouldFallback, source.fileId])

  const handleClick = (event: React.MouseEvent<HTMLImageElement>) => {
    if (!onFileClick || !source.fileId) return
    event.preventDefault()
    onFileClick(source.fileId, filename)
  }

  if (!resolvedUrl) {
    return (
      <div
        aria-label={filename}
        data-inline-file-preview-wrapper
        className={cn(
          'flex min-h-[8rem] items-center justify-center text-muted-foreground',
          imageClassName || 'max-w-full rounded-lg border border-border/50 bg-muted/20'
        )}
      >
        <Loader2 className="h-4 w-4 animate-spin" />
      </div>
    )
  }

  return (
    <img
      src={resolvedUrl}
      alt={filename}
      title={filename}
      data-file-path={source.fileId}
      className={imageClassName || 'max-w-full rounded-lg border border-border/50 bg-muted/20'}
      onClick={handleClick}
    />
  )
}

function InlineAudioPreview({
  source,
  previewUrl,
  filename,
  openLabel,
  className,
}: {
  source: InlineFilePreviewSource
  previewUrl: string
  filename: string
  openLabel: string
  className?: string
}) {
  const apiUrl = getApiUrl()
  const shouldFallback = Boolean(source.fileId)
  const [resolvedUrl, setResolvedUrl] = useState(shouldFallback ? '' : previewUrl)

  useEffect(() => {
    let objectUrl: string | null = null
    let isCancelled = false

    setResolvedUrl(shouldFallback ? '' : previewUrl)

    const loadAuthenticatedAudio = async () => {
      if (!shouldFallback || !source.fileId) return
      try {
        const response = await apiRequest(
          `${apiUrl}/api/files/preview/${encodeURIComponent(source.fileId)}`,
          {
            cache: 'no-cache',
            headers: {
              'Cache-Control': 'no-cache',
              Pragma: 'no-cache',
            },
          }
        )
        if (isCancelled) return
        if (!response.ok) {
          setResolvedUrl(previewUrl)
          return
        }
        const blob = await response.blob()
        if (isCancelled) return
        objectUrl = URL.createObjectURL(blob)
        setResolvedUrl(objectUrl)
      } catch {
        if (!isCancelled) {
          setResolvedUrl(previewUrl)
        }
      }
    }

    void loadAuthenticatedAudio()

    return () => {
      isCancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [apiUrl, previewUrl, shouldFallback, source.fileId])

  return (
    <div
      className={cn(
        'overflow-hidden rounded-md border border-border/50 bg-background',
        className
      )}
      data-inline-file-preview-wrapper
    >
      <div className="flex items-center gap-2 border-b border-border/50 bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
        <Volume2 className="h-4 w-4 shrink-0" />
        <span className="min-w-0 flex-1 truncate">{filename}</span>
        {resolvedUrl ? (
          <a
            href={resolvedUrl}
            target="_blank"
            rel="noreferrer"
            className="shrink-0 text-foreground hover:underline"
          >
            {openLabel}
          </a>
        ) : null}
      </div>
      <div className="p-3">
        {resolvedUrl ? (
          <audio
            controls
            preload="metadata"
            src={resolvedUrl}
            className="w-full"
            aria-label={filename}
            title={filename}
          />
        ) : (
          <div className="flex h-14 items-center justify-center text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
          </div>
        )}
      </div>
    </div>
  )
}

function InlineOfficeContent({
  kind,
  previewUrl,
  loadErrorText,
  fileId,
}: {
  kind: 'presentation' | 'document' | 'spreadsheet'
  previewUrl: string
  loadErrorText: string
  /** Optional fileId; when set, enables server-side PDF preview for .pptx. */
  fileId?: string
}) {
  const [base64Content, setBase64Content] = useState('')
  const [error, setError] = useState(false)

  useEffect(() => {
    // Presentation + fileId: skip the eager bytes download. Hooks must be
    // called unconditionally, so we guard here rather than relying on the
    // early render return below.  PptxPreviewRenderer lazy-fetches if needed.
    if (kind === 'presentation' && fileId) return
    if (!previewUrl) return

    let isCancelled = false

    const loadPreview = async () => {
      try {
        const response = await apiRequest(previewUrl, {
          cache: 'no-cache',
          headers: {
            'Cache-Control': 'no-cache',
            Pragma: 'no-cache',
          },
        })
        if (!response.ok) {
          throw new Error(`Failed to load file preview: ${response.status}`)
        }
        const buffer = await response.arrayBuffer()
        if (!isCancelled) {
          setBase64Content(arrayBufferToBase64(buffer))
          setError(false)
        }
      } catch {
        if (!isCancelled) {
          setBase64Content('')
          setError(true)
        }
      }
    }

    void loadPreview()

    return () => {
      isCancelled = true
    }
  }, [previewUrl])

  // Fast path: presentation with a managed fileId — skip the eager PPTX
  // download and mount the renderer immediately.  PptxPreviewRenderer will
  // probe the LibreOffice PDF endpoint first; only if that 503s does it
  // lazy-fetch the raw bytes from /api/files/public/preview/{fileId}.  For
  // large decks this means the PDF iframe can appear without ever paying the
  // base64 download + memory cost.  Per PR #542 review (rogercloud).
  if (kind === 'presentation' && fileId) {
    return <PptxPreviewRenderer fileId={fileId} />
  }

  if (error) {
    return <div className="p-3 text-xs text-muted-foreground">{loadErrorText}</div>
  }

  if (!base64Content) {
    return (
      <div className="flex h-32 items-center justify-center text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
      </div>
    )
  }

  // Presentation without a managed fileId (external previewUrl path) falls
  // through here with pre-loaded bytes.
  if (kind === 'presentation') {
    return <PptxPreviewRenderer base64Content={base64Content} />
  }

  if (kind === 'document') {
    return <DocxPreviewRenderer base64Content={base64Content} />
  }

  return <ExcelPreviewRenderer base64Content={base64Content} />
}

function ExternalPreviewPlaceholder({
  className,
  domain,
  filename,
  openLabel,
  previewUrl,
}: {
  className?: string
  domain?: string
  filename: string
  openLabel: string
  previewUrl: string
}) {
  return (
    <a
      href={previewUrl}
      target="_blank"
      rel="noreferrer noopener"
      className={cn(
        'flex items-center gap-2 rounded-md border border-border/50 bg-muted/20 px-3 py-2 text-xs text-foreground hover:bg-muted/40',
        className
      )}
    >
      <FileText className="h-4 w-4 text-muted-foreground" />
      <span className="min-w-0 flex-1 truncate">{filename}</span>
      {domain ? <span className="shrink-0 text-muted-foreground">{domain}</span> : null}
      <span className="shrink-0 text-foreground">{openLabel}</span>
    </a>
  )
}

export function InlineFilePreview({
  source,
  className,
  imageClassName,
  onFileClick,
  openLabel = DEFAULT_OPEN_LABEL,
  loadErrorText = DEFAULT_LOAD_ERROR_TEXT,
}: InlineFilePreviewProps) {
  const apiUrl = getApiUrl()
  const resolvedSource = source.fileId
    ? { ...source, fileId: resolveInlineFileId(source.fileId) }
    : source
  const kind = getInlineFilePreviewKind(resolvedSource)
  const previewUrl = getInlineFilePreviewUrl(resolvedSource, apiUrl)
  const downloadUrl = getInlineFileDownloadUrl(resolvedSource, apiUrl)
  const previewUrlTrust = getPreviewUrlTrust(resolvedSource, apiUrl)
  const filename = fileNameFromSource(resolvedSource)
  const canOpenFilePreview = Boolean(onFileClick && resolvedSource.fileId)

  const handleOpenPreview = (event: React.MouseEvent<HTMLElement>) => {
    if (!onFileClick || !resolvedSource.fileId) return
    event.preventDefault()
    onFileClick(resolvedSource.fileId, filename)
  }

  if (!previewUrl) return null

  if (!previewUrlTrust.isTrusted) {
    return (
      <ExternalPreviewPlaceholder
        className={className}
        domain={previewUrlTrust.domain}
        filename={filename}
        openLabel={openLabel}
        previewUrl={previewUrl}
      />
    )
  }

  if (kind === 'image') {
    return (
      <InlineImagePreview
        source={resolvedSource}
        previewUrl={previewUrl}
        filename={filename}
        imageClassName={imageClassName}
        onFileClick={onFileClick}
      />
    )
  }

  if (kind === 'audio') {
    return (
      <InlineAudioPreview
        source={resolvedSource}
        previewUrl={previewUrl}
        filename={filename}
        openLabel={openLabel}
        className={className}
      />
    )
  }

  if (!isPreviewableInlineFileKind(kind)) {
    return (
      <a
        href={downloadUrl}
        target={canOpenFilePreview ? undefined : '_blank'}
        rel={canOpenFilePreview ? undefined : 'noreferrer'}
        onClick={canOpenFilePreview ? handleOpenPreview : undefined}
        className={cn(
          'flex items-center gap-2 rounded-md border border-border/50 bg-muted/20 px-3 py-2 text-xs text-foreground hover:bg-muted/40',
          className
        )}
      >
        <FileText className="h-4 w-4 text-muted-foreground" />
        <span className="min-w-0 flex-1 truncate">{filename}</span>
      </a>
    )
  }

  return (
    <div
      className={cn('overflow-hidden rounded-md border border-border/50 bg-background', className)}
      data-inline-file-preview-wrapper
    >
      <div className="flex items-center gap-2 border-b border-border/50 bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
        <FileText className="h-4 w-4 shrink-0" />
        <span className="min-w-0 flex-1 truncate">{filename}</span>
        <a
          href={downloadUrl}
          target={canOpenFilePreview ? undefined : '_blank'}
          rel={canOpenFilePreview ? undefined : 'noreferrer'}
          onClick={canOpenFilePreview ? handleOpenPreview : undefined}
          className="shrink-0 text-foreground hover:underline"
        >
          {openLabel}
        </a>
      </div>
      <div className="h-[360px] overflow-auto">
        <InlineOfficeContent
          kind={kind}
          previewUrl={previewUrl}
          loadErrorText={loadErrorText}
          fileId={resolvedSource.fileId}
        />
      </div>
    </div>
  )
}
