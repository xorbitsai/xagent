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
