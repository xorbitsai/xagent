import type { ComponentType, ReactNode } from "react"

// Embedding distributions may replace default implementation modules during frontend composition.
// For the Home extension module (`home-page-extension.tsx`), `HomePageExtension` is the
// required export; `homeGetStartedDestinationOverrides` is OPTIONAL — a replacement module
// that omits it still builds and falls back to all canonical defaults (see page.tsx, which
// reads it through the module namespace rather than a static named import). A replacement
// module that omits the optional export will produce a benign webpack "Attempted import
// error" WARNING at compile time (not an error), so distributions with warnings-as-errors
// gates should expect it.
export type HomePageExtensionComponent = ComponentType

/**
 * Per-key override for the "Get started" card destinations on the home page.
 * Each key is resolved independently with three-state semantics:
 * - `undefined` (key omitted, or the whole `homeGetStartedDestinationOverrides`
 *   export is missing from a replacement module): the canonical OSS default
 *   destination is used. For `video`, the canonical OSS default is itself
 *   `null` — the card plays its inline tutorial video without linking out.
 * - `null` (or a non-string / blank value): the card renders as a
 *   non-interactive card — no anchor, no tab stop, no pointer cursor, and no
 *   link-only hover styling. A `video` card still autoplays its inline video
 *   in this state; only the link wrapper is affected.
 * - a string that is non-empty after trimming: used as the destination
 *   verbatim, including any surrounding whitespace, which is deliberately
 *   not trimmed from the emitted href.
 */
export interface HomeGetStartedDestinationOverrides {
  video?: string | null
  docs?: string | null
  guides?: string | null
  whatsNew?: string | null
}

// The page guarantees a stable Provider lifetime and agentId join key.
// The paired replacement implementation owns data loading, sharing, and invalidation.
export interface BuildAgentCardExtensionProps {
  // Stable key for joining Provider-owned page data to an Agent card.
  agentId: number
}

export interface BuildPageExtensionProviderProps {
  children: ReactNode
}

export type BuildPageExtensionProviderComponent =
  ComponentType<BuildPageExtensionProviderProps>

export type BuildAgentCardExtensionComponent =
  ComponentType<BuildAgentCardExtensionProps>

/** What an avatar is drawn from. `agentId` names the Agent a persona stands
 * for, when it stands for one: a distribution that lets people style an
 * Agent's avatar looks the style up by it. The OSS avatar does not read it. */
export interface AvatarPersona {
  name: string
  avatar?: string | null
  agentId?: number | null
}

/** An avatar's own color as an [r, g, b] triple, each 0-255. */
export type AvatarAccentColor = readonly [number, number, number]

export interface AgentBuilderSettingsExtensionOptions {
  /** Undefined when creating an Agent. */
  agentId?: number
  /** True when the Builder can render editable settings. */
  enabled: boolean
  /** True while an Agent save is in progress. */
  saving: boolean
  /** The name as currently typed; an avatar drawn from it changes with it. */
  agentName: string
  /** The Agent's logo image, which is the one thing its avatar can be besides a drawn one. */
  logo: AgentBuilderLogoControls
}

export interface AgentBuilderLogoControls {
  /** Where the current logo can be shown (a chosen file's preview or the saved image), or null. */
  src: string | null
  /** Use this image as the logo; it is uploaded when the Agent is saved. */
  select: (file: File) => void
  /** Clear the logo, saved or just chosen. */
  remove: () => void
}

export interface AgentBuilderSettingsExtension {
  panel?: ReactNode
  /** Replaces the built-in logo upload at the top of the form in every state,
   * read-only and loading included: while `enabled` is false return it with
   * its control disabled instead of leaving it out, or the built-in logo
   * upload would show in its place. A distribution that draws avatars shows
   * the logo and its avatar as one control through it, driving the logo with
   * `logo` from the options. */
  avatarPanel?: ReactNode
  /** Immutable JSON settings to submit. Omit or return an empty object when clean. */
  changes?: Record<string, unknown>
  /** Acknowledge the submitted settings after a successful save. Keep newer edits dirty. */
  markSaved?: (submitted: Record<string, unknown>) => void
}

// Wraps the Deploy dialog's key-based embed snippet. A replacement may render
// `children` alongside alternative embed instructions; the OSS default renders only `children`.
export interface DeployAgentEmbedExtensionProps {
  agentId: number
  // Origin the generated widget script loads from; empty until deployment config loads.
  widgetOrigin: string
  children: ReactNode
}

export type DeployAgentEmbedExtensionComponent =
  ComponentType<DeployAgentEmbedExtensionProps>

// Agent the channel extension acts on; a subset of the chat header's agent.
// The header only renders its action once the agent details have loaded, and
// the Deploy dialog receives the caller's agent, so these are real values.
export interface AgentChannelAgent {
  id: number
  name: string
  description: string
  logo_url: string | null
  status: string
}

// Extra deployment channel (for example a chat integration) offered by the
// agent chat header and the Deploy dialog. The OSS default renders nothing.
export interface AgentChannelExtensionProps {
  agent: AgentChannelAgent
}

// Header action rendered next to Routines; `onOpen` opens the Deploy dialog
// directly on the channel view.
export interface AgentChannelHeaderActionProps extends AgentChannelExtensionProps {
  onOpen: () => void
}
export type AgentChannelHeaderActionComponent =
  ComponentType<AgentChannelHeaderActionProps>

// Option card rendered first in the Deploy dialog's options grid.
export interface AgentChannelOptionProps extends AgentChannelExtensionProps {
  onOpen: () => void
}
export type AgentChannelOptionComponent = ComponentType<AgentChannelOptionProps>

// Full replacement for the Deploy dialog's `DialogContent` while the channel
// view is active. Contract:
// - render its own `DialogContent` (or portal/overlay/content equivalent);
// - include a `DialogTitle` and `DialogDescription`, so Radix does not warn and
//   screen readers get a dialog name;
// - own the back (`onBack`) and close (`onClose`) controls;
// - own any deployment-config error UI it needs: the dialog's default
//   `DeploymentConfigErrorAlert` is not rendered in this view.
export interface AgentChannelViewProps extends AgentChannelExtensionProps {
  onBack: () => void
  onClose: () => void
}
export type AgentChannelViewComponent = ComponentType<AgentChannelViewProps>

// ---------------------------------------------------------------------------
// Connecting an app from chat (`connect_connector` cards)
//
// The `connector` payload of the interaction and the types below are a
// contract between a host and this frontend; the backend forwards the payload
// as-is and defines no type for it.
//
// The module `lib/connector-connect.ts` is the extension point. It must export
// the four functions of `ConnectorConnectExtension` as named exports. The OSS
// default has no host service behind it and answers every call with
// `state: "unavailable"` without calling any endpoint; a distribution that
// provides the service replaces the module. `parseConnectorConnectResult`
// (lib/connector-connect-response.ts) is available to a replacement for
// turning a response body into a `ConnectorConnectResult`.
//
// Saving or skipping makes the host settle the interaction, which continues the
// task through the normal event stream; the card never sends a chat message.
// `taskResumed` says whether the host already queued the task to continue.
// `resumeConnector` queues it again and must be idempotent on the host.
// ---------------------------------------------------------------------------

export type ConnectorConnectState = "pending" | "connected" | "failed" | "skipped" | "unavailable"

export type ConnectorErrorCode =
  | "auth_failed"
  | "unreachable"
  | "timeout"
  | "no_tools"
  | "invalid_field"

export interface ConnectorToolSummary {
  name: string
  label: string
}

export interface ConnectorConnectResult {
  state: ConnectorConnectState
  errorCode: ConnectorErrorCode | null
  /** Key of the field the error is about; the card ignores one that matches no field. */
  errorField: string | null
  /** Masked tail of the saved secret, e.g. "•••• •••• fdsf". Never a field value. */
  secretMasked: string | null
  scope: string | null
  agentName: string | null
  tools: ConnectorToolSummary[]
  /** Whether the host has queued the task to continue; null when it does not say. */
  taskResumed: boolean | null
}

export interface ConnectorConnectExtension {
  /** Field values go only in the request body, and are never echoed back. */
  saveConnector(interactionId: string, appId: string, values: Record<string, string>): Promise<ConnectorConnectResult>
  skipConnector(interactionId: string): Promise<ConnectorConnectResult>
  /** The stored state of the interaction, to restore a card after a reload. */
  getConnectorState(interactionId: string): Promise<ConnectorConnectResult>
  resumeConnector(interactionId: string): Promise<ConnectorConnectResult>
}
