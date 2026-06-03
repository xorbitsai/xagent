"use client"

import { useParams, useSearchParams } from "next/navigation"
import { PublicAgentChatPage } from "@/components/widget/public-agent-chat-page"

export default function ShareChatPage() {
  const params = useParams()
  const searchParams = useSearchParams()
  const token = params.token as string

  return (
    <PublicAgentChatPage
      authMode="share"
      routeToken={token}
      guestId={searchParams.get("guest_id")}
    />
  )
}
