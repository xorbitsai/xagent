"use client"

import { AgentBuilder } from "@/components/build/agent-builder"
import { useParams } from "next/navigation"

export default function BuildDetailPage() {
  const params = useParams()
  const id = Array.isArray(params.id) ? params.id[0] : params.id

  return <AgentBuilder agentId={id} />
}
