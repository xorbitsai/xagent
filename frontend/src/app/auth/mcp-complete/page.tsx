"use client"

import React, { useEffect, useState } from "react"

/** No credentials, opener messaging, or navigation back to a logged-in page. */
export default function McpCompletePage() {
  const [success, setSuccess] = useState(false)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    setSuccess(params.get("mcp_oauth_success") === "1" && !params.has("mcp_oauth_error"))
    window.close()
  }, [])
  return <main><h1>{success ? "Authorization completed" : "Authorization not completed"}</h1>
    <p>You can close this window. Check the connection status in the original window.</p></main>
}
