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
  }
}

/** Keep a stable policy instance when a caller derives it from a public token. */
export function usePublicFileAccessPolicy(accessToken: string): FileAccessPolicy {
  return useMemo(() => createPublicFileAccessPolicy(accessToken), [accessToken])
}
