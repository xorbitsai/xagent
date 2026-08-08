"use client"

import React from "react"

// AppProvider owns the transport-derived value. Standalone renderers retain
// the legacy enabled behavior through this default.
export const AgentCardPresentationCapability = React.createContext(true)

export function resolveAgentCardPresentationCapability(
  filesDisabled: boolean,
  requestedAgentCards: boolean | undefined,
  inheritedAgentCards = true,
): boolean {
  return !filesDisabled && (requestedAgentCards ?? inheritedAgentCards)
}

// Off by default so every existing MarkdownRenderer consumer (skill-hub docs,
// the file viewer, conversation logs, ...) keeps opening links in the same
// tab. AppProvider turns this on only for the embedded Chat Widget, where an
// in-tab navigation would carry the visitor's iframe away from the page.
export const LinksOpenInNewTabCapability = React.createContext(false)

export function resolveLinksOpenInNewTabCapability(
  requestedLinksOpenInNewTab: boolean | undefined,
  inheritedLinksOpenInNewTab = false,
): boolean {
  return requestedLinksOpenInNewTab ?? inheritedLinksOpenInNewTab
}
