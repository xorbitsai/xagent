"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import {
  CheckCircle2,
  Copy,
  ExternalLink,
  Monitor,
  RefreshCw,
  Unplug,
} from "lucide-react"

import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { copyToClipboard } from "@/lib/clipboard"
import { getApiUrl } from "@/lib/utils"

interface DesktopRelayStatus {
  connected: boolean
  attached: boolean
  client_name?: string | null
  title?: string | null
  application?: string | null
  permissions?: Record<string, boolean>
  paused?: boolean
  emergency_stopped?: boolean
}

interface DesktopRelayPairing {
  pairing_token: string
  expires_at: string
  websocket_url: string
  protocol_version: number
}

export function DesktopRelaySettings() {
  const { t } = useI18n()
  const [status, setStatus] = useState<DesktopRelayStatus>({
    connected: false,
    attached: false,
  })
  const [pairing, setPairing] = useState<DesktopRelayPairing | null>(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const pairingSetup = useMemo(
    () =>
      pairing
        ? JSON.stringify(
            {
              websocket_url: pairing.websocket_url,
              pairing_token: pairing.pairing_token,
            },
            null,
            2,
          )
        : "",
    [pairing],
  )

  const refreshStatus = useCallback(async () => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/desktop-relay/status`)
      if (!response.ok) return
      const nextStatus = (await response.json()) as DesktopRelayStatus
      setStatus(nextStatus)
      if (nextStatus.connected) setPairing(null)
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
      const response = await apiRequest(
        `${getApiUrl()}/api/desktop-relay/pairings`,
        { method: "POST" },
      )
      if (!response.ok) throw new Error(t("settings.desktopRelay.errors.create"))
      setPairing(await response.json())
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : t("settings.desktopRelay.errors.create"),
      )
    } finally {
      setLoading(false)
    }
  }

  const revoke = async () => {
    setLoading(true)
    setMessage(null)
    try {
      const response = await apiRequest(
        `${getApiUrl()}/api/desktop-relay/session`,
        { method: "DELETE" },
      )
      if (!response.ok) throw new Error(t("settings.desktopRelay.errors.revoke"))
      setPairing(null)
      await refreshStatus()
      setMessage(t("settings.desktopRelay.revoked"))
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : t("settings.desktopRelay.errors.revoke"),
      )
    } finally {
      setLoading(false)
    }
  }

  const copy = async () => {
    setMessage(
      await copyToClipboard(pairingSetup)
        ? t("settings.desktopRelay.copied")
        : t("settings.desktopRelay.errors.copy"),
    )
  }

  const screenAllowed = status.permissions?.screen_recording === true
  const accessibilityAllowed = status.permissions?.accessibility === true

  return (
    <Card id="desktop-relay" className="scroll-mt-6">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Monitor className="h-5 w-5" />
          {t("settings.desktopRelay.title")}
        </CardTitle>
        <CardDescription>{t("settings.desktopRelay.description")}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2 rounded-md border p-3 text-sm">
          <div className="flex items-center gap-2 font-medium">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                status.connected ? "bg-emerald-500" : "bg-muted-foreground/40"
              }`}
            />
            {status.connected
              ? t("settings.desktopRelay.status.connected")
              : t("settings.desktopRelay.status.disconnected")}
          </div>
          <div className="text-muted-foreground">
            {status.attached
              ? t("settings.desktopRelay.status.attached", {
                  window:
                    [status.application, status.title].filter(Boolean).join(" — ") ||
                    status.client_name ||
                    "macOS",
                })
              : t("settings.desktopRelay.status.notAttached")}
          </div>
          {status.connected && (
            <div className="flex flex-wrap gap-2 text-xs">
              <PermissionBadge
                allowed={screenAllowed}
                label={t("settings.desktopRelay.permissions.screen")}
              />
              <PermissionBadge
                allowed={accessibilityAllowed}
                label={t("settings.desktopRelay.permissions.accessibility")}
              />
            </div>
          )}
        </div>

        {(status.paused || status.emergency_stopped) && (
          <Alert>
            <AlertDescription>
              {status.emergency_stopped
                ? t("settings.desktopRelay.status.emergencyStopped")
                : t("settings.desktopRelay.status.paused")}
            </AlertDescription>
          </Alert>
        )}

        <ol className="grid gap-3 rounded-md border p-3 text-sm">
          {[
            t("settings.desktopRelay.steps.build"),
            t("settings.desktopRelay.steps.pair"),
            t("settings.desktopRelay.steps.authorize"),
          ].map((step, index) => (
            <li key={step} className="flex items-start gap-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-semibold">
                {status.connected && index < 2 ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                ) : status.attached && index === 2 ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                ) : (
                  index + 1
                )}
              </span>
              <span className="pt-0.5">{step}</span>
            </li>
          ))}
        </ol>

        <a
          className="inline-flex items-center gap-1 text-sm text-primary underline-offset-4 hover:underline"
          href="https://github.com/xorbitsai/xagent/tree/main/desktop-relay"
          target="_blank"
          rel="noreferrer"
        >
          {t("settings.desktopRelay.installHelp")}
          <ExternalLink className="h-3.5 w-3.5" />
        </a>

        {pairing && (
          <div className="space-y-3 rounded-md border p-3">
            <p className="text-sm text-muted-foreground">
              {t("settings.desktopRelay.pairingHint")}
            </p>
            <Textarea
              aria-label={t("settings.desktopRelay.pairingSetup")}
              value={pairingSetup}
              readOnly
              rows={5}
              className="font-mono text-xs"
            />
            <Button variant="outline" className="w-full" onClick={() => void copy()}>
              <Copy className="mr-2 h-4 w-4" />
              {t("settings.desktopRelay.copySetup")}
            </Button>
            <p className="text-xs text-muted-foreground">
              {t("settings.desktopRelay.expires", {
                time: new Date(pairing.expires_at).toLocaleTimeString(),
              })}
            </p>
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
            {t("settings.desktopRelay.createPairing")}
          </Button>
          <Button variant="outline" onClick={() => void revoke()} disabled={loading}>
            <Unplug className="mr-2 h-4 w-4" />
            {t("settings.desktopRelay.revoke")}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function PermissionBadge({
  allowed,
  label,
}: {
  allowed: boolean
  label: string
}) {
  return (
    <span
      className={`rounded-full px-2 py-1 ${
        allowed
          ? "bg-emerald-500/10 text-emerald-700"
          : "bg-amber-500/10 text-amber-700"
      }`}
    >
      {label}: {allowed ? "✓" : "—"}
    </span>
  )
}
