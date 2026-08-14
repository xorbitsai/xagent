import React, { useCallback, useEffect, useRef, useState } from 'react'
import { FileText, Loader2, Video, Volume2 } from 'lucide-react'

import { DocxPreviewRenderer } from '@/components/file/docx-preview-renderer'
import { ExcelPreviewRenderer } from '@/components/file/excel-preview-renderer'
import { PptxPreviewRenderer } from '@/components/file/pptx-preview-renderer'
import { toast } from '@/components/ui/sonner'
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

// A minted streaming URL is cached briefly per fileId so remounts/rerenders
// of the same attachment (common in a chat transcript) don't each pay an
// extra mint round trip. Deliberately much shorter than the server's own
// operator-configurable ticket TTL (XAGENT_FILE_STREAM_TICKET_TTL_SECONDS,
// default 600s / 10min, validated only as >0) rather than trying to track
// the real expiry -- this is a de-dup window for near-simultaneous mounts,
// not an attempt to use a ticket right up to the edge of its validity. A
// deployment that lowers the server TTL below this window can have a
// client serve an already-expired ticket from cache; that self-heals via
// reportLoadFailure (one dead load, then a fresh mint), so it costs one
// failed request rather than a stuck player.
const STREAMING_URL_CACHE_TTL_MS = 4 * 60 * 1000
// Stores the in-flight mint promise, not just the settled result: two
// mounts of the same fileId within one render burst (e.g. the same
// attachment shown on two messages) would otherwise each pay a separate
// mint round trip before either resolves.
const streamingUrlCache = new Map<string, Promise<{ url: string; expiresAt: number }>>()

async function mintStreamingUrl(
  fileAccess: FileAccessPolicy,
  fileId: string
): Promise<string> {
  const cached = streamingUrlCache.get(fileId)
  if (cached) {
    const entry = await cached.catch(() => null)
    if (entry && entry.expiresAt > Date.now()) return entry.url
    // Expired, or the in-flight mint this promise represented failed --
    // either way this exact entry is now stale. Clear it before re-minting
    // rather than leaving a dead/expired promise for the next caller to
    // hit this same branch again, but only if nothing else already
    // replaced it (e.g. a concurrent caller's own retry).
    if (streamingUrlCache.get(fileId) === cached) {
      streamingUrlCache.delete(fileId)
    } else {
      // A concurrent caller already replaced this stale/failed entry with
      // its own fresh mint while this call was awaiting it above -- reuse
      // that instead of minting a second ticket for the same fileId. Every
      // caller in a burst hits this same await, one at a time, in the
      // order they were scheduled: whichever runs first deletes and mints;
      // every caller after it sees a mismatch here and recurses onto that
      // fresh entry rather than deleting-and-minting again itself.
      return mintStreamingUrl(fileAccess, fileId)
    }
  }
  const mintPromise = fileAccess.getStreamingUrl!(fileId).then((url) => ({
    url,
    expiresAt: Date.now() + STREAMING_URL_CACHE_TTL_MS,
  }))
  streamingUrlCache.set(fileId, mintPromise)
  try {
    return (await mintPromise).url
  } catch (error) {
    // Don't let a failed mint poison the cache for the next caller --
    // without this, every subsequent mount of this fileId would
    // immediately re-throw this same rejection instead of retrying.
    if (streamingUrlCache.get(fileId) === mintPromise) streamingUrlCache.delete(fileId)
    throw error
  }
}

/**
 * Test-only escape hatch: exercises the module-level mint de-dup/retry
 * logic directly, with exact control over concurrency (several calls for
 * the same fileId racing on one mint), which is impractical to pin down
 * reliably through React's own effect-scheduling timing.
 */
export function __mintStreamingUrlForTests(
  fileAccess: FileAccessPolicy,
  fileId: string
): Promise<string> {
  return mintStreamingUrl(fileAccess, fileId)
}

/**
 * Test-only escape hatch: this module-level cache persists across test
 * cases within the same test file (no module reset between them), so a
 * fixed test fileId reused across cases would otherwise silently return an
 * earlier case's mocked ticket instead of exercising the current case's
 * mock. Call from `beforeEach` in any test that mocks `getStreamingUrl`.
 */
export function __resetStreamingUrlCacheForTests(): void {
  streamingUrlCache.clear()
}

/**
 * Resolve the URL a media element (<img>/<audio>/<video>) can load.
 *
 * Three strategies, tried in this order:
 *
 * 1. ``fileAccess.getStreamingUrl`` (only the default in-app policy
 *    implements this, and only when ``allowStreaming`` is true): mints a
 *    short-lived, file-scoped ticket and hands the media element a direct
 *    URL, preserving HTTP range requests for progressive playback. Falls
 *    through to (2) if minting fails (e.g. offline) rather than leaving
 *    the player on a permanent spinner. If the media element itself then
 *    fails to load that URL after the ticket expires mid-session, the
 *    caller's ``onError`` should invoke the returned ``remintStreamingUrl``
 *    first -- minting is cheap and doesn't re-check ownership, so it's the
 *    right recovery for the common case of "the ticket simply expired
 *    while a long video was still playing" -- and only fall back to
 *    ``reportLoadFailure`` (which retries via (2)/(3) instead of leaving a
 *    dead ``src``) if the re-mint itself fails: this is not hypothetical,
 *    minting never checks file ownership by design (only redemption does),
 *    so a ticket can mint successfully for a file the caller can't
 *    actually read.
 * 2. Blob fetch: the default policy's authenticated preview route needs a
 *    Bearer header that media elements cannot send, so managed files are
 *    fetched into a blob object URL. If that fetch fails, the public
 *    preview URL is a last-resort fallback: on Agent Builder surfaces it
 *    may also require auth, but a spinner with no recovery path is worse
 *    than attempting the anonymous endpoint.
 * 3. Direct src: a policy that declares ``requiresBlobFetch: false`` (the
 *    public widget/share policy carries its guest token in the query
 *    string) hands the URL to the media element directly — a blob fetch
 *    would add no authorization there, and skipping it also preserves
 *    range requests. Policies that declare neither capability get the
 *    conservative blob path, which works everywhere.
 *
 * ``allowStreaming`` restricts strategy (1) to audio/video: images have no
 * seek-before-download or progressive-loading need, so paying an extra
 * network round trip to mint a ticket before every image load would be
 * pure overhead with no benefit.
 *
 * Returns ``openUrl`` alongside ``resolvedUrl`` for the "Open in new tab"
 * affordance when no in-app file-preview dialog is available (see
 * ``resolveOpenUrl`` below for when one isn't): the ticketed URL from
 * strategy (1) is a replayable credential (unlike the blob path's
 * session-scoped ``blob:`` URL, or the direct path's already-anonymous
 * public URL), so it must never be handed to something a user can put in
 * the address bar, browser history, or a copied link -- ``openUrl`` falls
 * back to the credential-free ``previewUrl`` the blob path itself uses on
 * failure, which is safe to use directly except in one case: for the
 * default policy while actively streaming, ``previewUrl`` is the tokenless
 * public preview route, which 403s once a task has access control
 * configured. ``resolveOpenUrl`` covers that case on demand.
 *
 * Callers with an in-app file-preview dialog available should route "Open"
 * through it instead of either of these (see ``InlineMediaPreview`` below,
 * which decides between the two using ``canOpenFilePreview``/
 * ``onOpenPreview`` passed down from ``InlineFilePreview``) -- this hook's
 * ``openUrl``/``resolveOpenUrl`` exist for surfaces with no such dialog
 * (e.g. the public widget, or a read-only transcript/log viewer).
 */
function useResolvedMediaUrl(
  source: InlineFilePreviewSource,
  previewUrl: string,
  fileAccess: FileAccessPolicy,
  allowStreaming: boolean
): {
  resolvedUrl: string
  openUrl: string
  isStreamingUrl: boolean
  resolveOpenUrl: () => Promise<string>
  remintStreamingUrl: () => Promise<boolean>
  reportLoadFailure: () => boolean
} {
  const fileId = source.fileId
  const canStream =
    allowStreaming && Boolean(fileId) && Boolean(fileAccess.getStreamingUrl)
  const needsBlobFetch = Boolean(fileId) && (fileAccess.requiresBlobFetch ?? true)
  const [resolvedUrl, setResolvedUrl] = useState(
    canStream || needsBlobFetch ? '' : previewUrl
  )
  const [isStreamingUrl, setIsStreamingUrl] = useState(false)
  // Set once the *media element itself* (not just minting) has failed to
  // load a streaming URL for this fileId, so this hook stops retrying
  // strategy (1) and settles on (2)/(3) for the remainder of this mount.
  const [streamingFailedFor, setStreamingFailedFor] = useState<string | null>(null)
  const streamingDisabled = canStream && streamingFailedFor === fileId

  useEffect(() => {
    let objectUrl: string | null = null
    let isCancelled = false
    const effectiveCanStream = canStream && !streamingDisabled

    setResolvedUrl(effectiveCanStream || needsBlobFetch ? '' : previewUrl)
    setIsStreamingUrl(false)

    const loadBlobOrDirect = async () => {
      if (!needsBlobFetch || !fileId) {
        setResolvedUrl(previewUrl)
        return
      }
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
      } catch (error) {
        if (!isCancelled) {
          console.warn(
            'InlineFilePreview: authenticated preview fetch failed, falling back to the public preview URL.',
            error
          )
          setResolvedUrl(previewUrl)
        }
      }
    }

    const loadMedia = async () => {
      // No separate "no fileId" early-return here: canStream (and so
      // effectiveCanStream) is already false whenever fileId is falsy, so
      // this always falls through to loadBlobOrDirect, whose own !fileId
      // guard produces the identical setResolvedUrl(previewUrl) outcome.
      // The "&& fileId" below is for TypeScript's narrowing, not runtime
      // behavior -- effectiveCanStream already guarantees it's truthy.
      if (effectiveCanStream && fileAccess.getStreamingUrl && fileId) {
        try {
          const streamingUrl = await mintStreamingUrl(fileAccess, fileId)
          if (isCancelled) return
          setResolvedUrl(streamingUrl)
          setIsStreamingUrl(true)
          return
        } catch (error) {
          console.warn(
            'InlineFilePreview: failed to mint a media streaming ticket, falling back.',
            error
          )
        }
      }
      if (isCancelled) return
      await loadBlobOrDirect()
    }

    void loadMedia()

    return () => {
      isCancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [fileAccess, fileId, canStream, streamingDisabled, needsBlobFetch, previewUrl])

  const reportLoadFailure = useCallback((): boolean => {
    if (isStreamingUrl && fileId) {
      // The ticket minted but the media element couldn't actually load it.
      // Evict the cache entry too, so a remount of this same fileId mints
      // fresh rather than immediately reusing the ticket that just failed.
      streamingUrlCache.delete(fileId)
      setStreamingFailedFor(fileId)
      return false // not terminal -- a (2)/(3) fallback attempt is starting
    }
    return true // every strategy has been exhausted
  }, [isStreamingUrl, fileId])

  // setResolvedUrl below fires from a callback whose completion the
  // component's own lifecycle doesn't otherwise gate (unlike the main
  // useEffect's isCancelled, which only covers *that* effect's run) -- this
  // is the guard for the one async state update outside it. Re-armed in the
  // effect body, not just initialized at declaration: StrictMode's dev-only
  // mount->cleanup->mount double-invoke would otherwise leave it stuck
  // false after the first cleanup, silently disabling the remint recovery.
  const isMountedRef = useRef(true)
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
    }
  }, [])

  // A ticket that expired mid-session is cheap to replace: minting doesn't
  // re-check file ownership (redemption does, and this fileId already
  // loaded successfully once under it), so a fresh ticket is the right
  // recovery -- not the full-file blob download reportLoadFailure below
  // falls back to, which is exactly the download-before-playback symptom
  // this streaming path exists to avoid. Returns false (falls through to
  // reportLoadFailure) only if re-minting itself fails, e.g. the file was
  // deleted or access was revoked since the last successful load.
  const remintStreamingUrl = useCallback(async (): Promise<boolean> => {
    if (!fileId) return false
    streamingUrlCache.delete(fileId) // the cached ticket just failed to load
    try {
      const freshUrl = await mintStreamingUrl(fileAccess, fileId)
      // JWT iat/exp have whole-second precision, so a mint within the same
      // second as the failing one (a transient network error, not expiry)
      // can be byte-identical -- setResolvedUrl would then be a no-op, the
      // element's src never changes, and nothing ever reloads it. Treat
      // that as a failed recovery so the caller falls back to the blob
      // path, which does recover.
      if (freshUrl === resolvedUrl) return false
      if (!isMountedRef.current) return true
      setResolvedUrl(freshUrl)
      return true
    } catch (error) {
      console.warn(
        'InlineFilePreview: failed to re-mint a media streaming ticket after mid-playback expiry, falling back.',
        error
      )
      return false
    }
  }, [fileId, fileAccess, resolvedUrl])

  // Every blob URL minted on demand for a "Open" click, kept alive until
  // unmount rather than revoked on the *next* click: a still-open first
  // tab's media keeps making Range requests against its blob URL as the
  // user seeks/reloads it, and revoking it out from under that tab would
  // break playback there the moment a second "Open" click happens on a
  // different tab. Independent of the playback src's own blob (which
  // useEffect above already manages on its own lifecycle).
  const openBlobUrlsRef = useRef<string[]>([])
  useEffect(() => () => {
    for (const url of openBlobUrlsRef.current) URL.revokeObjectURL(url)
  }, [])

  const resolveOpenUrl = useCallback(async (): Promise<string> => {
    // Not currently on the ticketed path: resolvedUrl is already either a
    // blob: URL or a direct src that's safe to hand to a new tab as-is.
    if (!isStreamingUrl) return resolvedUrl
    if (!needsBlobFetch || !fileId) return previewUrl
    try {
      const response = await fileAccess.request(fileAccess.previewUrl(fileId), {
        cache: 'no-cache',
        headers: {
          'Cache-Control': 'no-cache',
          Pragma: 'no-cache',
        },
      })
      if (!response.ok) return previewUrl
      const blob = await response.blob()
      const objectUrl = URL.createObjectURL(blob)
      openBlobUrlsRef.current.push(objectUrl)
      return objectUrl
    } catch {
      return previewUrl
    }
  }, [isStreamingUrl, needsBlobFetch, fileId, fileAccess, previewUrl, resolvedUrl])

  return {
    resolvedUrl,
    openUrl: isStreamingUrl ? previewUrl : resolvedUrl,
    isStreamingUrl,
    resolveOpenUrl,
    remintStreamingUrl,
    reportLoadFailure,
  }
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
  const { resolvedUrl } = useResolvedMediaUrl(source, previewUrl, fileAccess, false)

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
  canOpenFilePreview,
  onOpenPreview,
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
  // Computed once by the parent InlineFilePreview from onFileClick/fileId,
  // the same way its own image/generic-file branches do -- not re-derived
  // here, so there's exactly one place that decides whether a dialog is
  // available.
  canOpenFilePreview: boolean
  onOpenPreview: (event: React.MouseEvent<HTMLElement>) => void
  icon: React.ComponentType<{ className?: string }>
  bodyClassName: string
  spinnerClassName: string
  renderMedia: (
    resolvedUrl: string,
    media: {
      onError: (event: React.SyntheticEvent<HTMLMediaElement>) => void
      onLoaded: (event: React.SyntheticEvent<HTMLMediaElement>) => void
    }
  ) => React.ReactNode
}) {
  const {
    resolvedUrl,
    openUrl,
    isStreamingUrl,
    resolveOpenUrl,
    remintStreamingUrl,
    reportLoadFailure,
  } = useResolvedMediaUrl(source, previewUrl, fileAccess, true)
  // No dialog available (e.g. a read-only transcript/log viewer, or the
  // public widget): openUrl's static href is the safe default, but while
  // actively streaming under the default policy it's the same 403-prone
  // public URL as above. Open a blank tab synchronously inside this click
  // handler (required so the later async navigation isn't blocked as a
  // popup) and redirect it once resolveOpenUrl settles on a URL that will
  // actually load. Also wired to onAuxClick: middle-click/ctrl-click fire a
  // native 'auxclick' event that bypasses onClick entirely, so without this
  // the static href below (never the ticketed URL, by design) is what a
  // middle-click would navigate to -- a 403 on any access-controlled task.
  const handleOpenFallbackClick = (event: React.MouseEvent<HTMLAnchorElement>) => {
    // auxclick fires for ANY non-primary button, right-click (button 2)
    // included -- there the user wants the context menu, and hijacking it
    // into a surprise tab (plus preventDefault) would be a regression on a
    // gesture this handler was never meant for. Only the middle button
    // (button 1) is the open-in-new-tab gesture being recovered here.
    if (event.type === 'auxclick' && event.button !== 1) return
    event.preventDefault()
    const tab = window.open('about:blank', '_blank')
    if (!tab) {
      // Popup blocked (or a webview that refuses window.open): there is no
      // tab left to redirect once resolveOpenUrl settles, so say so rather
      // than resolving a URL nothing will ever navigate to.
      toast.error('Unable to open in a new tab. Check your browser’s pop-up blocker.')
      return
    }
    // Severs window.opener (the reverse-tabnabbing risk rel="noreferrer"
    // normally guards against on a static <a target="_blank">) while still
    // keeping our own reference to redirect once resolveOpenUrl settles --
    // window.open's own "noopener" feature string would null out that
    // reference too, which this handler needs.
    tab.opener = null
    void resolveOpenUrl().then((url) => {
      if (!tab.closed) tab.location.href = url
    })
  }
  const [failedUrl, setFailedUrl] = useState('')
  const [loadedUrl, setLoadedUrl] = useState('')
  // Where to seek to after a post-load recovery reload (see the onError
  // handler below) -- a plain ref, not state, since setting it must never
  // itself trigger a render; it's only read once, from the *next* onLoaded.
  const resumeAtRef = useRef<number | null>(null)
  // Terminal load failure only: an error event from a media element that
  // never loaded data means every fallback useResolvedMediaUrl has (blob
  // fetch, direct src, and -- via reportLoadFailure below -- a streaming
  // ticket that minted but wouldn't actually load) is exhausted. Errors
  // after data has loaded keep the player mounted by default -- most are
  // mid-playback decode hiccups the element surfaces and recovers from
  // itself -- except a network/src error while still on the ticketed path,
  // which the player can't recover from on its own: a ticket expiring
  // mid-session makes the browser's next Range request 401/403, and without
  // this special case that error would be silently swallowed by the
  // loadedUrl-already-set guard below, leaving playback stalled for the
  // rest of the ticket's (short, by design) TTL. Recovered in place first --
  // remintStreamingUrl swaps in a fresh ticket for the same fileId, which is
  // cheap because minting doesn't re-check ownership -- and only falls back
  // to reportLoadFailure's full blob download if the re-mint itself fails
  // (e.g. access was actually revoked, not just the ticket expired).
  // currentTime is saved first so playback can resume close to where it
  // stopped instead of restarting from 0, whichever recovery path is taken.
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
            href={openUrl}
            onClick={canOpenFilePreview ? onOpenPreview : handleOpenFallbackClick}
            onAuxClick={canOpenFilePreview ? undefined : handleOpenFallbackClick}
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
              onError: (event) => {
                if (loadedUrl === resolvedUrl) {
                  const element = event.currentTarget
                  // 2 === MediaError.MEDIA_ERR_NETWORK, 4 ===
                  // MEDIA_ERR_SRC_NOT_SUPPORTED -- numeric literals rather
                  // than the MediaError global are deliberate: these codes
                  // are a stable, unchanging part of the spec, and the
                  // constructor itself isn't implemented in every test
                  // environment (jsdom has no MediaError global), so a
                  // ReferenceError there must not be how this finds out.
                  // Both codes are treated as recoverable: browsers are
                  // inconsistent about which one an expired ticket's
                  // mid-stream 401/403 surfaces as -- some report NETWORK,
                  // others SRC_NOT_SUPPORTED because the failed fetch never
                  // yields a decodable resource. MEDIA_ERR_DECODE (3) is
                  // deliberately excluded: that's a genuine mid-playback
                  // decode hiccup the element already retries on its own,
                  // and forcing a re-mint there would interrupt playback
                  // that didn't need recovering.
                  const isRecoverableWhileStreaming =
                    isStreamingUrl && (element.error?.code === 2 || element.error?.code === 4)
                  if (!isRecoverableWhileStreaming) return // a hiccup the element handles itself
                  resumeAtRef.current = element.currentTime || null
                  setLoadedUrl('') // this url is dead; a future load must not be swallowed by this guard
                  void remintStreamingUrl().then((reminted) => {
                    // Re-mint failed too (e.g. access was actually revoked,
                    // not just the ticket expired) -- fall back to the full
                    // blob/direct strategy rather than leaving a dead src.
                    if (!reminted) reportLoadFailure()
                  })
                  return
                }
                if (reportLoadFailure()) setFailedUrl(resolvedUrl)
                // else: reportLoadFailure disabled the streaming ticket for
                // this fileId, which reruns useResolvedMediaUrl's effect
                // and falls back to the blob/direct strategy -- resolvedUrl
                // changes on the next render, so this failedUrl is stale
                // and `failed` above naturally goes back to false.
              },
              onLoaded: (event) => {
                setLoadedUrl(resolvedUrl)
                if (resumeAtRef.current !== null) {
                  event.currentTarget.currentTime = resumeAtRef.current
                  resumeAtRef.current = null
                }
              },
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
  canOpenFilePreview: boolean
  onOpenPreview: (event: React.MouseEvent<HTMLElement>) => void
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
        canOpenFilePreview={canOpenFilePreview}
        onOpenPreview={handleOpenPreview}
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
        canOpenFilePreview={canOpenFilePreview}
        onOpenPreview={handleOpenPreview}
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
