import type { AppIntegration } from "@/components/mcp/types"

export interface UnsharedConnector {
  type: string
  id?: number | string
  name: string
  reason?: "unresolved"
}

export interface UnsharedKnowledgeBase {
  name: string
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)

export const sanitizeUnsharedConnectors = (value: unknown): UnsharedConnector[] => {
  if (!Array.isArray(value)) return []
  return value.filter((item): item is UnsharedConnector => {
    if (!isRecord(item)) return false
    const validId =
      (typeof item.id === "number" && Number.isInteger(item.id)) ||
      (typeof item.id === "string" && item.id.length > 0)
    const unresolved = item.reason === "unresolved"
    return (
      typeof item.type === "string" &&
      item.type.length > 0 &&
      typeof item.name === "string" &&
      item.name.length > 0 &&
      (validId || unresolved) &&
      (item.reason === undefined || unresolved)
    )
  })
}

export const sanitizeUnsharedKnowledgeBases = (value: unknown): UnsharedKnowledgeBase[] => {
  if (!Array.isArray(value)) return []
  return value.filter(
    (item): item is UnsharedKnowledgeBase =>
      isRecord(item) && typeof item.name === "string" && item.name.length > 0,
  )
}

export const sanitizeAppIntegrations = (value: unknown): AppIntegration[] => {
  if (!Array.isArray(value)) return []
  return value.filter(
    (item): item is AppIntegration =>
      isRecord(item) &&
      typeof item.id === "string" &&
      typeof item.name === "string" &&
      typeof item.description === "string" &&
      typeof item.icon === "string",
  )
}

export interface ConnectorStatusEntry {
  shared: boolean
  is_owner: boolean
  needs_config: boolean
}

export const sanitizeConnectorStatusEntry = (value: unknown): ConnectorStatusEntry | null => {
  if (!isRecord(value)) return null
  if (
    typeof value.shared !== "boolean" ||
    typeof value.is_owner !== "boolean" ||
    typeof value.needs_config !== "boolean"
  ) {
    return null
  }
  return { shared: value.shared, is_owner: value.is_owner, needs_config: value.needs_config }
}

export const sanitizeConnectorStatus = (value: unknown): Record<string, ConnectorStatusEntry> => {
  if (!isRecord(value)) return {}
  const result: Record<string, ConnectorStatusEntry> = {}
  for (const [key, entry] of Object.entries(value)) {
    const sanitized = sanitizeConnectorStatusEntry(entry)
    if (sanitized) result[key] = sanitized
  }
  return result
}
