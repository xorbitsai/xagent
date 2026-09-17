"use client"

import React, { useEffect, useRef, useState } from "react"
import { getApiUrl } from "@/lib/utils"
import { createAuthSession, takeOidcAuthLoginIntent } from "@/lib/auth-cache"
import { useI18n } from "@/contexts/i18n-context"

export function OidcCallbackPage() {
  const { t } = useI18n()
  const [message, setMessage] = useState(t("login.google.completing"))
  const started = useRef(false)

  useEffect(() => {
    if (started.current) return
    started.current = true

    const completeLogin = async () => {
      const params = new URLSearchParams(window.location.search)
      const provider = params.get("provider")
      const code = params.get("code")
      const intentResult = takeOidcAuthLoginIntent()

      if (intentResult.status === "unavailable") {
        window.location.href = `/login?auth_unavailable=${intentResult.reason}`
        return
      }
      if (provider !== "google" || !code || intentResult.status !== "present") {
        window.location.href = "/login?oidc_error=exchange_failed"
        return
      }

      try {
        const response = await fetch(`${getApiUrl()}/api/auth/oidc/google/exchange`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ code }),
        })

        if (!response.ok) {
          throw new Error("OIDC exchange failed")
        }

        const data = await response.json()
        if (!data.success || !data.access_token || !data.user) {
          throw new Error("OIDC exchange response was incomplete")
        }

        const created = await createAuthSession(data, intentResult.intent)
        if (created.status !== "created") {
          if (created.status === "unavailable") {
            window.location.href = `/login?auth_unavailable=${created.reason}`
            return
          }
          throw new Error("OIDC session storage failed")
        }
        window.location.href = "/"
      } catch {
        // An OIDC callback can arrive after another login completed in this
        // browser. A failed exchange must not clear that unrelated lineage.
        setMessage(t("login.alerts.google_failed"))
        window.location.href = "/login?oidc_error=exchange_failed"
      }
    }

    void completeLogin()
  }, [t])

  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-muted-foreground border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-muted-foreground">{message}</p>
      </div>
    </div>
  )
}
