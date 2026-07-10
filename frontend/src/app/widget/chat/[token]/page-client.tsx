"use client"

import { Suspense } from "react"
import { useParams, useSearchParams } from "next/navigation"
import { PublicAgentChatPage } from "@/components/widget/public-agent-chat-page"

function WidgetChatInner() {
  const params = useParams()
  const searchParams = useSearchParams()
  const token = params.token as string

  return (
    <PublicAgentChatPage
      authMode="widget"
      routeToken={token}
      guestId={searchParams.get("guest_id")}
      endUserId={searchParams.get("end_user_id")}
      endUserSignature={searchParams.get("end_user_signature")}
      searchAgentId={searchParams.get("agent_id") ? parseInt(searchParams.get("agent_id") as string, 10) : null}
      embedTicket={searchParams.get("embed_ticket")}
      widgetKey={searchParams.get("widget_key")}
    />
  )
}

export default function WidgetChatPage() {
  // useSearchParams must be inside a Suspense boundary for static export.
  return (
    <Suspense fallback={null}>
      <WidgetChatInner />
    </Suspense>
  )
}
