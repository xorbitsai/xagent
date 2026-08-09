import React, { useEffect, useState } from 'react'
import { FileText, Loader2, Video, Volume2 } from 'lucide-react'

import { DocxPreviewRenderer } from '@/components/file/docx-preview-renderer'
import { ExcelPreviewRenderer } from '@/components/file/excel-preview-renderer'
import { PptxPreviewRenderer } from '@/components/file/pptx-preview-renderer'
import { cn, getApiUrl } from '@/lib/utils'
import { useFileAccess, type FileAccessPolicy } from '@/contexts/file-access-context'
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

/**
 * Resolve the URL a media element (<img>/<audio>/<video>) can load.
 *
 * The default in-app policy's authenticated preview route needs a Bearer
 * header that media elements cannot send, so managed files are fetched into
 * a blob object URL first. If that fetch fails, the public preview URL is a
 * last-resort fallback: on Agent Builder surfaces it may also require auth,
 * but a spinner with no recovery path is worse than attempting the
 * anonymous endpoint.
 *
 * A policy that declares ``requiresBlobFetch: false`` (the public
 * widget/share policy carries its guest token in the query string) hands
 * the URL to the media element directly instead: a blob fetch would add no
 * authorization there, and skipping it preserves HTTP range requests for
 * progressive audio/video playback. Policies that do not declare the
 * capability get the conservative blob path, which works everywhere.
 */
function useResolvedMediaUrl(
  source: InlineFilePreviewSource,
  previewUrl: string,
  fileAccess: FileAccessPolicy
): string {
  const fileId = source.fileId
  const needsBlobFetch = Boolean(fileId) && (fileAccess.requiresBlobFetch ?? true)
  const [resolvedUrl, setResolvedUrl] = useState(needsBlobFetch ? '' : previewUrl)

  useEffect(() => {
    let objectUrl: string | null = null
    let isCancelled = false

    setResolvedUrl(needsBlobFetch ? '' : previewUrl)

    const loadAuthenticatedMedia = async () => {
      if (!needsBlobFetch || !fileId) return
      try {
        const response = await fileAccess.request(fileAccess.previewUrl(fileId), {
          cache: 'no-cache',
          headers: {
            'Cache-Control': 'no-cache',
            Pragma: 'no-cache',
          },
        })
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

    void loadAuthenticatedMedia()

    return () => {
      isCancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [fileAccess, fileId, needsBlobFetch, previewUrl])

  return resolvedUrl
}

function InlineImagePreview({
  source,
  previewUrl,
  filename,
  imageClassName,
  onFileClick,
  fileAccess,
}: {
  source: InlineFilePreviewSource
  previewUrl: string
  filename: string
  imageClassName?: string
  onFileClick?: (filePath: string, fileName: string) => void
  fileAccess: FileAccessPolicy
}) {
  const resolvedUrl = useResolvedMediaUrl(source, previewUrl, fileAccess)

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

function InlineMediaPreview({
  source,
  previewUrl,
  filename,
  openLabel,
  loadErrorText,
  className,
  fileAccess,
  icon: Icon,
  bodyClassName,
  spinnerClassName,
  renderMedia,
}: {
  source: InlineFilePreviewSource
  previewUrl: string
  filename: string
  openLabel: string
  loadErrorText: string
  className?: string
  fileAccess: FileAccessPolicy
  icon: React.ComponentType<{ className?: string }>
  bodyClassName: string
  spinnerClassName: string
  renderMedia: (
    resolvedUrl: string,
    media: { onError: () => void; onLoaded: () => void }
  ) => React.ReactNode
}) {
  const resolvedUrl = useResolvedMediaUrl(source, previewUrl, fileAccess)
  const [failedUrl, setFailedUrl] = useState('')
  const [loadedUrl, setLoadedUrl] = useState('')
  // Terminal load failure only: useResolvedMediaUrl already falls back from
  // the authenticated fetch to the public preview URL, so an error event
  // from a media element that never loaded data means both paths are
  // exhausted. Errors after data has loaded (e.g. a mid-playback decode
  // hiccup) keep the player mounted — the element surfaces those itself.
  // Keyed by URL so a later re-resolve clears the failure.
  const failed = Boolean(resolvedUrl) && failedUrl === resolvedUrl

  return (
    <div
      className={cn(
        'overflow-hidden rounded-md border border-border/50 bg-background',
        className
      )}
      data-inline-file-preview-wrapper
    >
      <div className="flex items-center gap-2 border-b border-border/50 bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
        <Icon className="h-4 w-4 shrink-0" />
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
      {failed ? (
        <div className="p-3 text-xs text-muted-foreground">{loadErrorText}</div>
      ) : (
        <div className={bodyClassName}>
          {resolvedUrl ? (
            renderMedia(resolvedUrl, {
              onError: () => {
                if (loadedUrl !== resolvedUrl) setFailedUrl(resolvedUrl)
              },
              onLoaded: () => setLoadedUrl(resolvedUrl),
            })
          ) : (
            <div
              className={cn(
                'flex items-center justify-center text-muted-foreground',
                spinnerClassName
              )}
            >
              <Loader2 className="h-4 w-4 animate-spin" />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

type MediaWrapperProps = {
  source: InlineFilePreviewSource
  previewUrl: string
  filename: string
  openLabel: string
  loadErrorText: string
  className?: string
  fileAccess: FileAccessPolicy
}

function InlineAudioPreview(props: MediaWrapperProps) {
  const { filename } = props
  return (
    <InlineMediaPreview
      {...props}
      icon={Volume2}
      bodyClassName="p-3"
      spinnerClassName="h-14"
      renderMedia={(resolvedUrl, { onError, onLoaded }) => (
        <audio
          controls
          preload="metadata"
          src={resolvedUrl}
          className="w-full"
          aria-label={filename}
          title={filename}
          onError={onError}
          onLoadedData={onLoaded}
        />
      )}
    />
  )
}

function InlineVideoPreview(props: MediaWrapperProps) {
  const { filename } = props
  return (
    <InlineMediaPreview
      {...props}
      icon={Video}
      bodyClassName="flex items-center justify-center bg-black/95 p-2"
      spinnerClassName="h-40 w-full"
      renderMedia={(resolvedUrl, { onError, onLoaded }) => (
        <video
          controls
          playsInline
          preload="metadata"
          src={resolvedUrl}
          className="max-h-[360px] w-full max-w-full rounded bg-black"
          aria-label={filename}
          title={filename}
          onError={onError}
          onLoadedData={onLoaded}
        />
      )}
    />
  )
}

function InlineOfficeContent({
  kind,
  previewUrl,
  loadErrorText,
  fileId,
  fileAccess,
}: {
  kind: 'presentation' | 'document' | 'spreadsheet'
  previewUrl: string
  loadErrorText: string
  /** Optional fileId; when set, enables server-side PDF preview for .pptx. */
  fileId?: string
  fileAccess: FileAccessPolicy
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
        const response = await fileAccess.request(previewUrl, {
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
  }, [fileAccess, previewUrl])

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
  const fileAccess = useFileAccess()
  const resolvedSource = source.fileId
    ? { ...source, fileId: resolveInlineFileId(source.fileId) }
    : source
  const kind = getInlineFilePreviewKind(resolvedSource)
  const previewUrl = resolvedSource.fileId
    ? fileAccess.inlinePreviewUrl(resolvedSource.fileId)
    : getInlineFilePreviewUrl(resolvedSource, apiUrl)
  const downloadUrl = resolvedSource.fileId
    ? fileAccess.inlineDownloadUrl(resolvedSource.fileId)
    : getInlineFileDownloadUrl(resolvedSource, apiUrl)
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
        fileAccess={fileAccess}
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
        loadErrorText={loadErrorText}
        className={className}
        fileAccess={fileAccess}
      />
    )
  }

  if (kind === 'video') {
    return (
      <InlineVideoPreview
        source={resolvedSource}
        previewUrl={previewUrl}
        filename={filename}
        openLabel={openLabel}
        loadErrorText={loadErrorText}
        className={className}
        fileAccess={fileAccess}
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
          fileAccess={fileAccess}
        />
      </div>
    </div>
  )
}
