"use client"

import { useParams } from "next/navigation"
import { PublicAgentChatPage } from "@/components/widget/public-agent-chat-page"

export default function ShareChatPage() {
  const params = useParams()
  const token = params.token as string

  return (
    <PublicAgentChatPage
      authMode="share"
      routeToken={token}
    />
  )
}
