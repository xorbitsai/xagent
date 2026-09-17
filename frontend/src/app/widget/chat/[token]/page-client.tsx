"use client"

import React, { Suspense } from "react"
import { useParams, useSearchParams } from "next/navigation"
import { PublicAgentChatPage } from "@/components/widget/public-agent-chat-page"
import { SessionAgentChatPage } from "@/components/widget/session-agent-chat-page"

function LegacyWidgetChat({ routeToken }: { routeToken: string }) {
  const searchParams = useSearchParams()

  return (
    <PublicAgentChatPage
      authMode="widget"
      routeToken={routeToken}
      guestId={searchParams.get("guest_id")}
      searchAgentId={searchParams.get("agent_id") ? parseInt(searchParams.get("agent_id") as string, 10) : null}
      embedTicket={searchParams.get("embed_ticket")}
      widgetKey={searchParams.get("widget_key")}
    />
  )
}

function WidgetChatInner() {
  const params = useParams()
  const routeToken = params.token as string

  if (routeToken === "session") {
    return <SessionAgentChatPage />
  }

  return <LegacyWidgetChat routeToken={routeToken} />
}

export default function WidgetChatPage() {
  // useSearchParams must be inside a Suspense boundary for static export.
  return (
    <Suspense fallback={null}>
      <WidgetChatInner />
    </Suspense>
  )
}
