"use client"

import React from "react"
import { useI18n } from "@/contexts/i18n-context"

// Shown wherever a builtin OAuth connector row is listed but its grant needs
// a reconnect (lib/mcp-connection-status.ts). Styled like the Tools card's
// "Needs config" badge, so the two warnings read as one family.
export function NeedsReconnectBadge() {
  const { t } = useI18n()
  return (
    <span className="whitespace-nowrap rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-semibold text-amber-700">
      {t("tools.mcp.needsReconnect")}
    </span>
  )
}
