import { getApiUrl } from "@/lib/utils"
import { resolveApiSnippetBaseUrl, type ApiSnippetTarget } from "@/lib/api-snippet-target"

export function getApiSnippetTarget(): ApiSnippetTarget {
  const browserOrigin =
    typeof window !== "undefined" ? window.location?.origin ?? "" : ""
  const candidate = getApiUrl() || browserOrigin
  return { baseUrl: resolveApiSnippetBaseUrl(candidate, browserOrigin) }
}
