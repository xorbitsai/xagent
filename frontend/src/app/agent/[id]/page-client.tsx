"use client"

import React, { useEffect } from "react"
import { Loader2 } from "lucide-react"
import { useParams, useRouter } from "next/navigation"

import { getAgentTaskHref } from "@/lib/agent-ui-access"

/**
 * Compatibility-only route for old bookmarks and links. Agent chat no longer
 * has a separate page; every published agent is opened through the shared
 * task composer with its id in the query string.
 */
export default function AgentChatRedirectPage() {
  const params = useParams()
  const router = useRouter()
  const rawAgentId = Array.isArray(params.id) ? params.id[0] : params.id

  useEffect(() => {
    router.replace(rawAgentId && rawAgentId !== "__shell__" ? getAgentTaskHref(rawAgentId) : "/task")
  }, [rawAgentId, router])

  return (
    <div className="grid h-full place-items-center bg-background" aria-label="Opening teammate chat">
      <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
    </div>
  )
}
