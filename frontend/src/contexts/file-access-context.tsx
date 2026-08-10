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

const defaultFileAccessPolicy: FileAccessPolicy = {
  previewUrl: (fileId) => buildUrl(`/api/files/preview/${encodeFileId(fileId)}`),
  downloadUrl: (fileId) => buildUrl(`/api/files/download/${encodeFileId(fileId)}`),
  inlinePreviewUrl: (fileId) => buildUrl(`/api/files/public/preview/${encodeFileId(fileId)}`),
  inlineDownloadUrl: (fileId) => buildUrl(`/api/files/public/download/${encodeFileId(fileId)}`),
  relativePreviewUrl: (fileId, relativePath) => {
    const url = new URL(
      buildUrl(`/api/files/public/preview/${encodeFileId(fileId)}`),
      typeof window === "undefined" ? "http://localhost" : window.location.origin,
    )
    url.searchParams.set("relative_path", relativePath)
    return getApiUrl() ? url.toString() : `${url.pathname}${url.search}${url.hash}`
  },
  pdfPreviewUrl: (fileId) => buildUrl(`/api/files/preview-pdf/${encodeFileId(fileId)}`),
  request: apiRequest,
  listFiles: (query) => {
    const params = new URLSearchParams({ page: "1", size: "20" })
    if (query.trim()) params.set("search", query.trim())
    return apiRequest(buildUrl(`/api/files/list?${params.toString()}`))
  },
  // previewUrl needs the Bearer header apiRequest attaches; media elements
  // cannot send it, so they must go through the blob fetch unless a
  // streaming ticket is available (see getStreamingUrl below).
  requiresBlobFetch: true,
  getStreamingUrl: async (fileId) => {
    const response = await apiRequest(
      buildUrl(`/api/files/stream-tickets/${encodeFileId(fileId)}`),
    )
    if (!response.ok) {
      throw new Error(`Failed to mint stream ticket: ${response.status}`)
    }
    const data: { path: string } = await response.json()
    return buildUrl(data.path)
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
    publicUrl(`/api/files/public/preview/${encodeFileId(fileId)}`)
  const inlineDownloadUrl = (fileId: string) =>
    publicUrl(`/api/files/public/download/${encodeFileId(fileId)}`)

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
      return fetch(url, { ...options, headers, credentials: "omit" })
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
