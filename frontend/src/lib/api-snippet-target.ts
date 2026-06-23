export interface ApiSnippetTarget {
  baseUrl: string
}

export function normalizeApiSnippetBaseUrl(rawBaseUrl: string): string {
  return rawBaseUrl.trim().replace(/\/+$/, "")
}

export function resolveApiSnippetBaseUrl(
  rawBaseUrl: string,
  browserOrigin = "",
): string {
  const candidate = normalizeApiSnippetBaseUrl(rawBaseUrl)
  if (!candidate) {
    return ""
  }
  if (isHttpUrl(candidate)) {
    return candidate
  }

  const origin = normalizeApiSnippetBaseUrl(browserOrigin)
  if (!isHttpUrl(origin)) {
    return ""
  }

  try {
    return normalizeApiSnippetBaseUrl(new URL(candidate, `${origin}/`).toString())
  } catch {
    return ""
  }
}

function isHttpUrl(value: string): boolean {
  try {
    const url = new URL(value)
    return url.protocol === "http:" || url.protocol === "https:"
  } catch {
    return false
  }
}
