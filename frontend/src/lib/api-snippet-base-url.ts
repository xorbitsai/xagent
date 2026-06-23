import { getBrowserLocationOrigin } from "@/lib/browser-location"
import { getApiUrl } from "@/lib/utils"
import { resolveApiSnippetBaseUrl, type ApiSnippetTarget } from "@/lib/api-snippet-target"

export function getApiSnippetTarget(): ApiSnippetTarget {
  const browserOrigin = getBrowserLocationOrigin()
  const candidate = getApiUrl() || browserOrigin
  return { baseUrl: resolveApiSnippetBaseUrl(candidate, browserOrigin) }
}
