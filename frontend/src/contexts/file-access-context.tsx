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
  /**
   * Execute under this policy's credential boundary. The built-in default
   * attaches Bearer authorization. The public policy forces same-origin
   * credentials, strips caller Authorization, and leaves its scoped query
   * token on the URL. Custom policies must define an equivalent boundary.
   */
  request: FileAccessRequest
  listFiles?: (query: string) => Promise<Response>
  /**
   * Whether media elements must load managed files through an
   * authenticated fetch into a blob URL. True when previewUrl requires
   * headers a media element cannot send (the default Bearer policy);
   * false when the URL is directly loadable (the public policy carries
   * its guest token in the query string), which preserves HTTP range
   * requests for progressive playback. Policies that do not declare the
   * capability get the conservative blob path.
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

const appendPublicToken = (url: string, accessToken: string): string => {
  const resolved = new URL(
    url,
    typeof window === "undefined" ? "http://localhost" : window.location.origin,
  )
  resolved.searchParams.set("token", accessToken)
  return getApiUrl() ? resolved.toString() : `${resolved.pathname}${resolved.search}${resolved.hash}`
}

/**
 * The public widget/share provider owns this capability. Its access token is
 * captured by the provider instance, never read from browser storage, and is
 * only attached to the file routes that need public guest authorization.
 * Browser media loads cannot attach an Authorization header, so the query
 * token stays scoped to those public routes rather than becoming ambient auth.
 */
export function createPublicFileAccessPolicy(accessToken: string): FileAccessPolicy {
  const publicUrl = (path: string) => appendPublicToken(buildUrl(path), accessToken)
  const inlinePreviewUrl = (fileId: string) =>
    publicUrl(`${FILES_API_PREFIX}/public/preview/${encodeFileId(fileId)}`)
  const inlineDownloadUrl = (fileId: string) =>
    publicUrl(`${FILES_API_PREFIX}/public/download/${encodeFileId(fileId)}`)

  return {
    previewUrl: inlinePreviewUrl,
    downloadUrl: inlineDownloadUrl,
    inlinePreviewUrl,
    inlineDownloadUrl,
    relativePreviewUrl: (fileId, relativePath) => {
      const url = new URL(
        inlinePreviewUrl(fileId),
        typeof window === "undefined" ? "http://localhost" : window.location.origin,
      )
      url.searchParams.set("relative_path", relativePath)
      return getApiUrl() ? url.toString() : `${url.pathname}${url.search}${url.hash}`
    },
    // Public file routes deliberately do not expose the authenticated PDF
    // conversion endpoint. PPTX falls back to the scoped public byte route.
    request: (url, options = {}) => {
      const headers = new Headers(options.headers)
      headers.delete("Authorization")
      // Same-origin requests carry ambient cookies, including the edge's
      // selected_region routing cookie. Cookies are not an authorization input.
      // Task-bound configured files validate the scoped query token; taskless
      // and legacy paths use their file ID or path capability.
      return fetch(url, { ...options, headers, credentials: "same-origin" })
    },
    // The guest token rides the query string, so media elements can load
    // previewUrl directly — no headers needed, and range requests work.
    requiresBlobFetch: false,
  }
}

/** Keep a stable policy instance when a caller derives it from a public token. */
export function usePublicFileAccessPolicy(accessToken: string): FileAccessPolicy {
  return useMemo(() => createPublicFileAccessPolicy(accessToken), [accessToken])
}
