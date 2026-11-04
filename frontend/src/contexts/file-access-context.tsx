"use client"

import React, { createContext, useContext, useMemo } from "react"

import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

export type FileAccessRequest = (
  url: string,
  options?: RequestInit,
) => Promise<Response>

export interface FileAccessPolicy {
  previewUrl: (fileId: string) => string
  downloadUrl: (fileId: string) => string
  inlinePreviewUrl: (fileId: string) => string
  inlineDownloadUrl: (fileId: string) => string
  relativePreviewUrl: (fileId: string, relativePath: string) => string
  pdfPreviewUrl?: (fileId: string) => string
  validationUrl?: (fileId: string) => string
  /**
   * Execute under this policy's credential boundary. The built-in default
   * attaches Bearer authorization. The public policy forces same-origin
   * credentials and attaches its guest token as a Bearer header, only for
   * the public file routes. Custom policies must define an equivalent
   * boundary. No policy puts a login or guest token in a URL.
   */
  request: FileAccessRequest
  listFiles?: (query: string) => Promise<Response>
  /**
   * Whether media elements must load managed files through an
   * authenticated fetch into a blob URL. True when previewUrl requires
   * headers a media element cannot send (both built-in policies); false
   * only when the URL is directly loadable as is. Policies that do not
   * declare the capability get the conservative blob path.
   */
  requiresBlobFetch?: boolean
  /**
   * Mint a short-lived, file-scoped ticket and return a URL a media
   * element can load directly, preserving HTTP range requests even under
   * a policy whose previewUrl otherwise requires an Authorization header.
   * When present, this takes priority over requiresBlobFetch; failure
   * (e.g. offline) falls back to that capability instead.
   */
  getStreamingUrl?: (fileId: string) => Promise<string>
  /**
   * Turn a URL from this policy's own builders into one the browser can load
   * without headers (for example an HTML preview's subresources), by adding a
   * short-lived, file-bound ticket. Absent when the builders' URLs already
   * load as is. Rejects for URLs this policy did not build.
   */
  authorizeBrowserUrl?: (url: string) => Promise<string>
}

const encodeFileId = (fileId: string) => encodeURIComponent(fileId)

const buildUrl = (path: string): string => `${getApiUrl()}${path}`

const STREAM_TICKET_MINT_TIMEOUT_MS = 10_000

// Single source of truth for the files router's mount prefix, shared between
// every URL builder below and the mint-response shape check -- otherwise the
// two could drift the way files.py's own prefix derivation was written to
// avoid on the server side.
const FILES_API_PREFIX = "/api/files"

// Exported so tests unrelated to streaming-ticket mechanics can spread this
// and override just `getStreamingUrl` (e.g. to undefined), rather than
// depending on an incidental mock-shape quirk (a mocked response missing
// `.json()`) to keep a ticket mint attempt from participating in the test.
export const defaultFileAccessPolicy: FileAccessPolicy = {
  previewUrl: (fileId) => buildUrl(`${FILES_API_PREFIX}/preview/${encodeFileId(fileId)}`),
  downloadUrl: (fileId) => buildUrl(`${FILES_API_PREFIX}/download/${encodeFileId(fileId)}`),
  inlinePreviewUrl: (fileId) =>
    buildUrl(`${FILES_API_PREFIX}/public/preview/${encodeFileId(fileId)}`),
  inlineDownloadUrl: (fileId) =>
    buildUrl(`${FILES_API_PREFIX}/public/download/${encodeFileId(fileId)}`),
  relativePreviewUrl: (fileId, relativePath) => {
    const url = new URL(
      buildUrl(`${FILES_API_PREFIX}/public/preview/${encodeFileId(fileId)}`),
      typeof window === "undefined" ? "http://localhost" : window.location.origin,
    )
    url.searchParams.set("relative_path", relativePath)
    return getApiUrl() ? url.toString() : `${url.pathname}${url.search}${url.hash}`
  },
  pdfPreviewUrl: (fileId) => buildUrl(`${FILES_API_PREFIX}/preview-pdf/${encodeFileId(fileId)}`),
  validationUrl: (fileId) => buildUrl(`${FILES_API_PREFIX}/preview/${encodeFileId(fileId)}?validation_only=true`),
  request: apiRequest,
  listFiles: (query) => {
    const params = new URLSearchParams({ page: "1", size: "20" })
    if (query.trim()) params.set("search", query.trim())
    return apiRequest(buildUrl(`${FILES_API_PREFIX}/list?${params.toString()}`))
  },
  // previewUrl needs the Bearer header apiRequest attaches; media elements
  // cannot send it, so they must go through the blob fetch unless a
  // streaming ticket is available (see getStreamingUrl below).
  requiresBlobFetch: true,
  getStreamingUrl: async (fileId) => {
    // A hung mint request would otherwise leave a media element stuck
    // loading indefinitely, with nothing to time it out -- the caller's own
    // unmount handling (isCancelled checks in useResolvedMediaUrl) ignores
    // a late response but doesn't abort the underlying request.
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), STREAM_TICKET_MINT_TIMEOUT_MS)
    let response: Response
    try {
      response = await apiRequest(
        buildUrl(`${FILES_API_PREFIX}/stream-tickets/${encodeFileId(fileId)}`),
        { signal: controller.signal },
      )
    } finally {
      clearTimeout(timeout)
    }
    if (!response.ok) {
      throw new Error(`Failed to mint stream ticket: ${response.status}`)
    }
    // The mint endpoint's response shape is a server contract this client
    // doesn't otherwise validate; a malformed/unexpected payload must
    // surface as a caught rejection (so the caller falls back to the blob
    // path) rather than silently becoming a broken `undefined`-based URL.
    // Bound to this exact fileId, not just the generic preview prefix: a
    // prefix-only check would pass a path with extra segments after it
    // (e.g. a dot-segment escaping to a different route) as long as it
    // still started with "/api/files/preview/" -- exploitability is ~nil
    // today (this server is the sole minter, and the ticket's own file_id
    // claim is re-checked at redemption regardless of what path string got
    // us there), but there's no reason to accept a shape looser than what
    // this server actually returns. Decoded and compared, not re-encoded
    // and string-matched: the server encodes file_id with Python's
    // urllib.parse.quote(safe=""), which escapes a different character set
    // than encodeURIComponent does (e.g. "!*'()" round-trip unescaped
    // through encodeURIComponent but ARE escaped by quote) -- re-encoding
    // fileId here and comparing strings would reject a genuinely valid
    // response for any fileId containing one of those characters.
    // decodeURIComponent has no such mismatch: it decodes any valid
    // percent-encoded UTF-8 sequence regardless of which safe-set the
    // encoder chose, so it round-trips correctly against either encoder.
    const previewPrefix = `${FILES_API_PREFIX}/preview/`
    const data: unknown = await response.json()
    const path =
      data && typeof data === "object" ? (data as { path?: unknown }).path : undefined
    if (typeof path !== "string" || !path.startsWith(previewPrefix)) {
      throw new Error("Stream ticket response has an unexpected shape")
    }
    const encodedFileIdAndQuery = path.slice(previewPrefix.length)
    const encodedFileId = encodedFileIdAndQuery.split("?")[0]
    let decodedFileId: string
    try {
      decodedFileId = decodeURIComponent(encodedFileId)
    } catch {
      throw new Error("Stream ticket response has an unexpected shape")
    }
    if (decodedFileId !== fileId) {
      throw new Error("Stream ticket response has an unexpected shape")
    }
    return buildUrl(path)
  },
}

const FileAccessContext = createContext<FileAccessPolicy>(defaultFileAccessPolicy)

export function FileAccessProvider({
  children,
  policy,
}: {
  children: React.ReactNode
  policy?: FileAccessPolicy
}) {
  return (
    <FileAccessContext.Provider value={policy ?? defaultFileAccessPolicy}>
      {children}
    </FileAccessContext.Provider>
  )
}

export function useFileAccess(): FileAccessPolicy {
  return useContext(FileAccessContext)
}

const PUBLIC_FILES_PREFIX = `${FILES_API_PREFIX}/public/`
const PUBLIC_FILE_ROUTE = new RegExp(`^${FILES_API_PREFIX}/public/(?:preview|download)/([^/?#]+)$`)
const TICKET_SHAPE = /^[A-Za-z0-9._~-]{1,8192}$/
// Re-mint this long before the server-side expiry so a URL handed to the
// browser does not expire while its first request is still in flight.
// Clamped to half the ticket's lifetime (see mintTicket), so a deployment
// with a short XAGENT_FILE_STREAM_TICKET_TTL_SECONDS still caches a ticket
// for a while instead of treating every fresh one as already stale.
const TICKET_REFRESH_MARGIN_MS = 30_000

const resolveAgainstPage = (url: string): URL =>
  new URL(url, typeof window === "undefined" ? "http://localhost" : window.location.origin)

/** Whether a URL targets this API's public file routes (the only place the
 * guest token is ever sent). */
const isPublicFileRoute = (url: string): boolean => {
  try {
    const resolved = resolveAgainstPage(url)
    const apiOrigin = resolveAgainstPage(buildUrl("/")).origin
    return resolved.origin === apiOrigin && resolved.pathname.startsWith(PUBLIC_FILES_PREFIX)
  } catch {
    return false
  }
}

/**
 * The public widget/share provider owns this capability. Its access token is
 * captured by the provider instance, never read from browser storage, and is
 * only sent as an Authorization header to the public file routes. Loads that
 * cannot send headers (media streaming, HTML preview subresources) use a
 * short-lived ticket bound to one file, minted with that header; the guest
 * token itself never appears in a URL.
 */
export function createPublicFileAccessPolicy(accessToken: string): FileAccessPolicy {
  const inlinePreviewUrl = (fileId: string) =>
    buildUrl(`${FILES_API_PREFIX}/public/preview/${encodeFileId(fileId)}`)
  const inlineDownloadUrl = (fileId: string) =>
    buildUrl(`${FILES_API_PREFIX}/public/download/${encodeFileId(fileId)}`)

  const request: FileAccessRequest = (url, options = {}) => {
    const headers = new Headers(options.headers)
    headers.delete("Authorization")
    if (accessToken && isPublicFileRoute(url)) {
      headers.set("Authorization", `Bearer ${accessToken}`)
    }
    // Same-origin requests carry ambient cookies, including the edge's
    // selected_region routing cookie. Cookies are not an authorization input.
    // Task-bound files validate the guest token header (or a file ticket);
    // taskless and legacy paths use their file ID or path capability.
    return fetch(url, { ...options, headers, credentials: "same-origin" })
  }

  type TicketEntry = { ticket: string; refreshAt: number }
  // One cache per policy instance (so per guest token): pending and settled
  // mints by fileId. A failed or timed-out mint is evicted, so the next load
  // retries instead of re-throwing a cached rejection.
  const tickets = new Map<string, Promise<TicketEntry>>()
  // When each settled mint stops being reused, keyed by its cache promise.
  // Pending mints have no entry, so pruning never drops a per-file
  // single-flight promise.
  const settledRefreshAt = new WeakMap<Promise<TicketEntry>, number>()
  // Without this, a file that is never requested again would keep its stale
  // entry for the life of the policy, so the cache would grow with every
  // file visited.
  const pruneStaleTickets = () => {
    const now = Date.now()
    for (const [fileId, cached] of tickets) {
      const refreshAt = settledRefreshAt.get(cached)
      if (refreshAt !== undefined && refreshAt <= now) tickets.delete(fileId)
    }
  }

  const readTicket = async (fileId: string, signal: AbortSignal): Promise<TicketEntry> => {
    const response = await request(
      buildUrl(`${FILES_API_PREFIX}/public/tickets/${encodeFileId(fileId)}`),
      { method: "POST", signal },
    )
    if (!response.ok) {
      throw new Error(`Failed to mint file ticket: ${response.status}`)
    }
    const data: unknown = await response.json()
    const record = data && typeof data === "object" ? data as Record<string, unknown> : {}
    const ticket = record.ticket
    const expiresIn = record.expires_in
    if (typeof ticket !== "string" || !TICKET_SHAPE.test(ticket)
      || typeof expiresIn !== "number" || !Number.isFinite(expiresIn) || expiresIn <= 0) {
      throw new Error("File ticket response has an unexpected shape")
    }
    const lifetimeMs = expiresIn * 1000
    const marginMs = Math.min(TICKET_REFRESH_MARGIN_MS, lifetimeMs / 2)
    return { ticket, refreshAt: Date.now() + lifetimeMs - marginMs }
  }

  const fetchTicket = async (fileId: string): Promise<TicketEntry> => {
    // A hung mint would otherwise stall every media/HTML-preview load for
    // this file behind the cached pending promise. The deadline covers the
    // whole mint -- headers, body and validation -- not just the request:
    // aborting cancels a real fetch's body read too, and the race settles
    // even when the body stream ignores the signal.
    const controller = new AbortController()
    let timeout: ReturnType<typeof setTimeout> | undefined
    const deadline = new Promise<never>((_, reject) => {
      timeout = setTimeout(() => {
        reject(new Error("File ticket mint timed out"))
        controller.abort()
      }, STREAM_TICKET_MINT_TIMEOUT_MS)
    })
    try {
      return await Promise.race([readTicket(fileId, controller.signal), deadline])
    } finally {
      clearTimeout(timeout)
    }
  }

  const mintTicket = async (fileId: string): Promise<string> => {
    pruneStaleTickets()
    const cached = tickets.get(fileId)
    if (cached) {
      const entry = await cached.catch(() => null)
      if (entry && entry.refreshAt > Date.now()) return entry.ticket
      const current = tickets.get(fileId)
      if (current === cached) {
        tickets.delete(fileId)
      } else if (current) {
        // A concurrent caller already replaced this stale entry with its own
        // mint: share that one (without re-checking it) instead of minting
        // a second ticket.
        return (await current).ticket
      }
    }
    const pending = fetchTicket(fileId)
    tickets.set(fileId, pending)
    try {
      // A just-minted ticket is returned as is, never re-checked against
      // refreshAt: that check is only for reuse, so no TTL can make this loop.
      const entry = await pending
      settledRefreshAt.set(pending, entry.refreshAt)
      return entry.ticket
    } catch (error) {
      if (tickets.get(fileId) === pending) tickets.delete(fileId)
      throw error
    }
  }

  const authorizeBrowserUrl = async (url: string): Promise<string> => {
    const resolved = resolveAgainstPage(url)
    const match = isPublicFileRoute(url) ? PUBLIC_FILE_ROUTE.exec(resolved.pathname) : null
    if (!match) throw new Error("Not a public file URL")
    const ticket = await mintTicket(decodeURIComponent(match[1]))
    resolved.searchParams.set("ticket", ticket)
    return /^[A-Za-z][A-Za-z0-9+.-]*:/.test(url)
      ? resolved.toString()
      : `${resolved.pathname}${resolved.search}${resolved.hash}`
  }

  return {
    previewUrl: inlinePreviewUrl,
    downloadUrl: inlineDownloadUrl,
    inlinePreviewUrl,
    inlineDownloadUrl,
    validationUrl: (fileId) => `${inlinePreviewUrl(fileId)}?validation_only=true`,
    relativePreviewUrl: (fileId, relativePath) => {
      const url = resolveAgainstPage(inlinePreviewUrl(fileId))
      url.searchParams.set("relative_path", relativePath)
      return getApiUrl() ? url.toString() : `${url.pathname}${url.search}${url.hash}`
    },
    // Public file routes deliberately do not expose the authenticated PDF
    // conversion endpoint. PPTX falls back to the scoped public byte route.
    request,
    // previewUrl needs the guest Bearer header, which media elements cannot
    // send: they load through a blob fetch, or a file ticket when streaming.
    requiresBlobFetch: true,
    getStreamingUrl: (fileId) => authorizeBrowserUrl(inlinePreviewUrl(fileId)),
    authorizeBrowserUrl,
  }
}

/** Keep a stable policy instance when a caller derives it from a public token. */
export function usePublicFileAccessPolicy(accessToken: string): FileAccessPolicy {
  return useMemo(() => createPublicFileAccessPolicy(accessToken), [accessToken])
}
