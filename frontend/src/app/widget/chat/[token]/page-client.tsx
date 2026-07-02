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
      searchAgentId={searchParams.get("agent_id") ? parseInt(searchParams.get("agent_id") as string, 10) : null}
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
