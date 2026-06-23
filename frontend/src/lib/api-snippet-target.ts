export interface ApiSnippetTarget {
  baseUrl: string
}

export function normalizeApiSnippetBaseUrl(rawBaseUrl: string): string {
  return rawBaseUrl.trim().replace(/\/+$/, "")
}
