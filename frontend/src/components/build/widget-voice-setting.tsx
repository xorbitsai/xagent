"use client"

import React, { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { toast } from "@/components/ui/sonner"
import { useI18n } from "@/contexts/i18n-context"
import type { WidgetVoiceSetting as WidgetVoiceSettingValue } from "@/lib/agent-widget-config"

interface WidgetVoiceSettingProps {
  load: () => Promise<WidgetVoiceSettingValue>
  save: (enabled: boolean) => Promise<WidgetVoiceSettingValue>
}

type LoadState =
  | { status: "loading" }
  | { status: "failed" }
  | { status: "ready"; setting: WidgetVoiceSettingValue }

/**
 * The owner's switch for voice calls in the embedded chat widget.
 *
 * The switch is only usable once the stored value has been read: while it is
 * loading, or when reading failed, it stays disabled instead of showing "off".
 * After a save the server's response is the new state, so a concurrent change
 * made elsewhere is shown rather than overwritten by the click.
 */
export function WidgetVoiceSetting({ load, save }: WidgetVoiceSettingProps) {
  const { t } = useI18n()
  const [state, setState] = useState<LoadState>({ status: "loading" })
  const [isSaving, setIsSaving] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const loadRef = useRef(load)
  const mountedRef = useRef(true)

  useEffect(() => {
    loadRef.current = load
  }, [load])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    setState({ status: "loading" })
    void (async () => {
      try {
        const setting = await loadRef.current()
        if (!cancelled) setState({ status: "ready", setting })
      } catch (error) {
        if (cancelled) return
        console.error(error)
        setState({ status: "failed" })
      }
    })()
    return () => {
      cancelled = true
    }
  }, [attempt])

  const handleChange = async (enabled: boolean) => {
    setIsSaving(true)
    try {
      const setting = await save(enabled)
      if (!mountedRef.current) return
      setState({ status: "ready", setting })
      toast.success(t("widget_voice_setting.saved"))
    } catch (error) {
      if (!mountedRef.current) return
      console.error(error)
      toast.error(t("widget_voice_setting.save_failed"))
    } finally {
      if (mountedRef.current) setIsSaving(false)
    }
  }

  const setting = state.status === "ready" ? state.setting : null
  const unavailableMessage =
    setting?.enabled && setting.available === false
      ? setting.reason === "voice_models_not_configured"
        ? t("widget_voice_setting.unavailable_not_configured")
        : setting.reason === "voice_models_unavailable"
          ? t("widget_voice_setting.unavailable_models")
          : null
      : null

  return (
    <div className="space-y-2 rounded-lg border p-4">
      <div className="flex items-center justify-between gap-4">
        <div className="space-y-0.5">
          <Label className="text-base">{t("widget_voice_setting.title")}</Label>
          <div className="text-sm text-muted-foreground">
            {t("widget_voice_setting.description")}
          </div>
        </div>
        <Switch
          checked={setting?.enabled ?? false}
          onCheckedChange={(checked) => void handleChange(checked)}
          disabled={setting === null || isSaving}
          aria-label={t("widget_voice_setting.toggle_label")}
        />
      </div>
      {state.status === "loading" && (
        <div className="text-sm text-muted-foreground">
          {t("widget_voice_setting.loading")}
        </div>
      )}
      {state.status === "failed" && (
        <div className="flex items-center gap-2 text-sm text-destructive">
          <span>{t("widget_voice_setting.load_failed")}</span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAttempt((value) => value + 1)}
          >
            {t("widget_voice_setting.retry")}
          </Button>
        </div>
      )}
      {unavailableMessage && (
        <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
          {unavailableMessage}
        </div>
      )}
      <div className="text-xs text-muted-foreground">
        {t("widget_voice_setting.requirements_note")}
      </div>
      <div className="text-xs text-muted-foreground">
        {t("widget_voice_setting.refresh_note")}
      </div>
    </div>
  )
}
