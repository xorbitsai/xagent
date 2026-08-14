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
