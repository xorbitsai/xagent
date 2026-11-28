"use client"

import React, { useEffect, useState } from "react"
import { MCP_OAUTH_STATE_ALREADY_CONSUMED, MCP_OAUTH_SUCCESS_PARAM } from "@/lib/mcp-utils"

const headings = {
  success: "Authorization completed",
  alreadyUsed: "Authorization link already used",
  failed: "Authorization not completed",
}

/** No credentials, opener messaging, or navigation back to a logged-in page. */
export default function McpCompletePage() {
  const [outcome, setOutcome] = useState<keyof typeof headings | null>(null)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    if (params.get("mcp_oauth_error") === MCP_OAUTH_STATE_ALREADY_CONSUMED) {
      setOutcome("alreadyUsed")
    } else if (params.get(MCP_OAUTH_SUCCESS_PARAM) === "1" && !params.has("mcp_oauth_error")) {
      setOutcome("success")
    } else {
      setOutcome("failed")
    }
    window.close()
  }, [])
  return (
    <main>
      {outcome && <h1>{headings[outcome]}</h1>}
      <p>You can close this window. Check the connection status in the original window.</p>
    </main>
  )
}
