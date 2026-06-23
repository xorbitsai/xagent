export interface ApiSnippetTarget {
  baseUrl: string
}

export function normalizeApiSnippetBaseUrl(rawBaseUrl: string): string {
  const baseUrl = rawBaseUrl.trim().replace(/\/+$/, "")
  if (!baseUrl) {
    throw new Error("Unable to determine API snippet base URL")
  }
  return baseUrl
}
