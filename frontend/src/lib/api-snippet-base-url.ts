import { getApiUrl } from "@/lib/utils"

export interface ApiSnippetTarget {
  baseUrl: string
}

export function getApiSnippetTarget(): ApiSnippetTarget {
  const baseUrl =
    getApiUrl() || (typeof window !== "undefined" ? window.location.origin : "")
  return { baseUrl }
}
