import React, { useEffect, useState } from 'react'
import { FileText, Loader2 } from 'lucide-react'

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
  type InlineFilePreviewSource,
  UUID_PATTERN,
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
  const [resolvedUrl, setResolvedUrl] = useState(previewUrl)

  useEffect(() => {
    let objectUrl: string | null = null
    let isCancelled = false

    setResolvedUrl(previewUrl)

    const runFallback = async () => {
      if (!source.fileId || source.previewUrl || UUID_PATTERN.test(source.fileId)) return
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
        if (!response.ok) return
        const blob = await response.blob()
        objectUrl = URL.createObjectURL(blob)
        if (!isCancelled) {
          setResolvedUrl(objectUrl)
        }
      } catch {
        return
      }
    }

    void runFallback()

    return () => {
      isCancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [apiUrl, previewUrl, source.fileId, source.previewUrl])

  const handleClick = (event: React.MouseEvent<HTMLImageElement>) => {
    if (!onFileClick || !source.fileId) return
    event.preventDefault()
    onFileClick(source.fileId, filename)
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

function InlineOfficeContent({
  kind,
  previewUrl,
  loadErrorText,
}: {
  kind: 'presentation' | 'document' | 'spreadsheet'
  previewUrl: string
  loadErrorText: string
}) {
  const [base64Content, setBase64Content] = useState('')
  const [error, setError] = useState(false)

  useEffect(() => {
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

  // Presentation now goes through PptxPreviewRenderer (canvas-based,
  // pptxviewjs) instead of an iframe — browsers can't render raw .pptx
  // bytes in an iframe, and the backend's /api/files/public/preview
  // endpoint now returns those raw bytes (rogercloud review on #465).
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
  const kind = getInlineFilePreviewKind(source)
  const previewUrl = getInlineFilePreviewUrl(source, apiUrl)
  const downloadUrl = getInlineFileDownloadUrl(source, apiUrl)
  const previewUrlTrust = getPreviewUrlTrust(source, apiUrl)
  const filename = fileNameFromSource(source)
  const canOpenFilePreview = Boolean(onFileClick && source.fileId)

  const handleOpenPreview = (event: React.MouseEvent<HTMLElement>) => {
    if (!onFileClick || !source.fileId) return
    event.preventDefault()
    onFileClick(source.fileId, filename)
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
        source={source}
        previewUrl={previewUrl}
        filename={filename}
        imageClassName={imageClassName}
        onFileClick={onFileClick}
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
        />
      </div>
    </div>
  )
}
