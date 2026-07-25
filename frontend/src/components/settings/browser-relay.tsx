"use client"

import { useCallback, useEffect, useState } from "react"
import { Copy, MonitorSmartphone, RefreshCw, Unplug } from "lucide-react"

import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { copyToClipboard } from "@/lib/clipboard"
import { getApiUrl } from "@/lib/utils"

interface BrowserRelayStatus {
  connected: boolean
  attached: boolean
  client_name?: string | null
  title?: string | null
  url?: string | null
}

interface BrowserRelayPairing {
  pairing_token: string
  expires_at: string
  websocket_url: string
  protocol_version: number
}

export function BrowserRelaySettings() {
  const { t } = useI18n()
  const [status, setStatus] = useState<BrowserRelayStatus>({
    connected: false,
    attached: false,
  })
  const [pairing, setPairing] = useState<BrowserRelayPairing | null>(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  const refreshStatus = useCallback(async () => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/browser-relay/status`)
      if (response.ok) {
        setStatus(await response.json())
      }
    } catch {
      setStatus({ connected: false, attached: false })
    }
  }, [])

  useEffect(() => {
    void refreshStatus()
    const timer = window.setInterval(() => void refreshStatus(), 3_000)
    return () => window.clearInterval(timer)
  }, [refreshStatus])

  const createPairing = async () => {
    setLoading(true)
    setMessage(null)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/browser-relay/pairings`, {
        method: "POST",
      })
      if (!response.ok) throw new Error(t("settings.browserRelay.errors.create"))
      setPairing(await response.json())
    } catch (error) {
      setMessage(error instanceof Error ? error.message : t("settings.browserRelay.errors.create"))
    } finally {
      setLoading(false)
    }
  }

  const revoke = async () => {
    setLoading(true)
    setMessage(null)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/browser-relay/session`, {
        method: "DELETE",
      })
      if (!response.ok) throw new Error(t("settings.browserRelay.errors.revoke"))
      setPairing(null)
      await refreshStatus()
      setMessage(t("settings.browserRelay.revoked"))
    } catch (error) {
      setMessage(error instanceof Error ? error.message : t("settings.browserRelay.errors.revoke"))
    } finally {
      setLoading(false)
    }
  }

  const copy = async (value: string) => {
    setMessage(
      await copyToClipboard(value)
        ? t("settings.browserRelay.copied")
        : t("settings.browserRelay.errors.copy"),
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MonitorSmartphone className="h-5 w-5" />
          {t("settings.browserRelay.title")}
        </CardTitle>
        <CardDescription>{t("settings.browserRelay.description")}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-md border p-3 text-sm">
          <div className="font-medium">
            {status.connected
              ? t("settings.browserRelay.status.connected")
              : t("settings.browserRelay.status.disconnected")}
          </div>
          <div className="mt-1 text-muted-foreground">
            {status.attached
              ? t("settings.browserRelay.status.attached", {
                  tab: status.title || status.url || status.client_name || "Chrome",
                })
              : t("settings.browserRelay.status.notAttached")}
          </div>
        </div>

        {pairing && (
          <div className="space-y-3 rounded-md border p-3">
            <p className="text-sm text-muted-foreground">
              {t("settings.browserRelay.pairingHint")}
            </p>
            <div className="space-y-2">
              <Label htmlFor="browser-relay-url">{t("settings.browserRelay.relayUrl")}</Label>
              <div className="flex gap-2">
                <Input id="browser-relay-url" value={pairing.websocket_url} readOnly />
                <Button variant="outline" size="icon" onClick={() => void copy(pairing.websocket_url)}>
                  <Copy className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="browser-pairing-token">{t("settings.browserRelay.pairingToken")}</Label>
              <div className="flex gap-2">
                <Input id="browser-pairing-token" value={pairing.pairing_token} readOnly />
                <Button variant="outline" size="icon" onClick={() => void copy(pairing.pairing_token)}>
                  <Copy className="h-4 w-4" />
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                {t("settings.browserRelay.expires", {
                  time: new Date(pairing.expires_at).toLocaleTimeString(),
                })}
              </p>
            </div>
          </div>
        )}

        {message && (
          <Alert>
            <AlertDescription>{message}</AlertDescription>
          </Alert>
        )}

        <div className="flex flex-wrap gap-2">
          <Button onClick={() => void createPairing()} disabled={loading}>
            <RefreshCw className="mr-2 h-4 w-4" />
            {t("settings.browserRelay.createPairing")}
          </Button>
          <Button variant="outline" onClick={() => void revoke()} disabled={loading}>
            <Unplug className="mr-2 h-4 w-4" />
            {t("settings.browserRelay.revoke")}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
