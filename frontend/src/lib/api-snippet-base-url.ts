import { getApiUrl } from "@/lib/utils"
import { normalizeApiSnippetBaseUrl, type ApiSnippetTarget } from "@/lib/api-snippet-target"

export function getApiSnippetTarget(): ApiSnippetTarget {
  const candidate =
    getApiUrl() || (typeof window !== "undefined" ? window.location.origin : "")
  return { baseUrl: normalizeApiSnippetBaseUrl(candidate) }
}
