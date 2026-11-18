export interface ConnectorLogoData {
  mark: string
  bg: string
  fg: string
  icon_url?: string | null
}

export type ConnectorRoute = "native" | "mcp" | "api"

export interface ConnectorField {
  key: string
  label: string
  placeholder?: string
  secret?: boolean
}

/** The `connector` object of a `connect_connector` interaction (display data only). */
export interface ConnectorInfo {
  /** The interaction item's `interaction_id`, which the host endpoints key on. */
  interaction_id: string
  app_id: string
  name: string
  logo: ConnectorLogoData
  route: ConnectorRoute
  what: string
  key_label: string
  how: string[]
  fields: ConnectorField[]
  probe?: { checks: string[]; found: ConnectorRoute }
  agent_name?: string | null
}

/** `metadata.result_card` of an assistant message with `display: "result_card"`. */
export interface ResultCardData {
  title: string
  source: { name: string; logo: ConnectorLogoData }
  items: string[]
  footnote?: string | null
  created_at?: string | null
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v)
const text = (v: unknown): string => (typeof v === "string" ? v : "")
const strings = (v: unknown): string[] =>
  Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []

function logo(v: unknown): ConnectorLogoData {
  const l = isRecord(v) ? v : {}
  return {
    mark: text(l.mark) || "•",
    bg: text(l.bg) || "#3D3D46",
    fg: text(l.fg) || "#fff",
    icon_url: typeof l.icon_url === "string" ? l.icon_url : null,
  }
}

/**
 * Defensive parse of untrusted wire data; null when unusable. The interaction
 * id is the `interaction_id` of the interaction item itself (`interactionId`),
 * not a field of `connector`. A card with no id or no fields cannot be used.
 */
export function parseConnectorInfo(raw: unknown, interactionId: string | undefined): ConnectorInfo | null {
  const id = interactionId
  if (!isRecord(raw) || !id || !text(raw.name)) return null
  const route: ConnectorRoute = raw.route === "native" || raw.route === "api" ? raw.route : "mcp"
  const probe = isRecord(raw.probe) ? raw.probe : null
  const fields: ConnectorField[] = Array.isArray(raw.fields)
    ? raw.fields.flatMap((f) =>
        isRecord(f) && text(f.key)
          ? [{ key: text(f.key), label: text(f.label) || text(f.key), placeholder: text(f.placeholder), secret: f.secret === true }]
          : [],
      )
    : []
  if (fields.length === 0) return null
  return {
    interaction_id: id,
    app_id: text(raw.app_id),
    name: text(raw.name),
    logo: logo(raw.logo),
    route,
    what: text(raw.what),
    key_label: text(raw.key_label),
    how: strings(raw.how),
    fields,
    probe: probe
      ? { checks: strings(probe.checks), found: probe.found === "native" || probe.found === "api" ? probe.found : "mcp" }
      : undefined,
    agent_name: typeof raw.agent_name === "string" ? raw.agent_name : null,
  }
}

export function parseResultCard(raw: unknown): ResultCardData | null {
  if (!isRecord(raw) || !text(raw.title)) return null
  const source = isRecord(raw.source) ? raw.source : {}
  return {
    title: text(raw.title),
    source: { name: text(source.name), logo: logo(source.logo) },
    items: strings(raw.items),
    footnote: typeof raw.footnote === "string" && raw.footnote ? raw.footnote : null,
    created_at: typeof raw.created_at === "string" ? raw.created_at : null,
  }
}
