"use client"

import React, { useEffect, useState } from "react"
import { Check, Copy, LayoutGrid } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "@/components/ui/sonner"
import { copyToClipboard } from "@/lib/clipboard"
import { getBrowserLocationOrigin } from "@/lib/browser-location"
import { buildWidgetSnippet, isValidAllowedDomain, normalizeAllowedDomain } from "@/lib/agent-widget-config"
import {
  getWorkforceWidgetConfig,
  rotateWorkforceWidgetKey,
  updateWorkforceWidgetConfig,
} from "@/lib/workforces-api"
import type { WorkforceDetail, WorkforceWidgetConfig } from "@/types/workforce"

interface WorkforceWidgetDialogProps {
  workforce: WorkforceDetail | null
  open: boolean
  onClose: () => void
}

export function WorkforceWidgetDialog({ workforce, open, onClose }: WorkforceWidgetDialogProps) {
  const { t } = useI18n()
  const [config, setConfig] = useState<WorkforceWidgetConfig | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isUpdating, setIsUpdating] = useState(false)
  const [isRotating, setIsRotating] = useState(false)
  const [newDomain, setNewDomain] = useState("")
  const [copied, setCopied] = useState(false)
  const [appOrigin, setAppOrigin] = useState("")

  const isActive = workforce?.status === "active"
  const widgetEnabled = config?.widget_enabled ?? false
  const allowedDomains = config?.allowed_domains ?? []
  const widgetKey = config?.widget_key ?? null

  useEffect(() => {
    setAppOrigin(getBrowserLocationOrigin())
  }, [])

  useEffect(() => {
    if (!open || !workforce) {
      return
    }
    let cancelled = false
    const load = async () => {
      try {
        setIsLoading(true)
        const state = await getWorkforceWidgetConfig(workforce.id)
        if (!cancelled) setConfig(state)
      } catch (err) {
        if (!cancelled) {
          console.error(err)
          toast.error(t("workforces.widget.messages.failed") || "Widget action failed")
        }
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }
    void load()
    return () => {
      cancelled = true
    }
    // `t` excluded on purpose: only used for the error toast; depending on it
    // would refetch on every render where the i18n function identity changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, workforce?.id])

  const applyUpdate = async (
    updates: { widget_enabled?: boolean; allowed_domains?: string[] },
    successMessage: string,
  ): Promise<boolean> => {
    if (!workforce) return false
    try {
      setIsUpdating(true)
      const state = await updateWorkforceWidgetConfig(workforce.id, updates)
      setConfig(state)
      toast.success(successMessage)
      return true
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("workforces.widget.messages.failed") || "Widget action failed")
      return false
    } finally {
      setIsUpdating(false)
    }
  }

  const handleToggle = (checked: boolean) => {
    void applyUpdate(
      { widget_enabled: checked },
      checked
        ? t("workforces.widget.messages.enabled") || "Widget enabled"
        : t("workforces.widget.messages.disabled") || "Widget disabled",
    )
  }

  const handleAddDomain = async () => {
    const domain = normalizeAllowedDomain(newDomain)
    if (!domain) return
    if (!isValidAllowedDomain(domain)) {
      toast.error(t("workforces.widget.invalid_domain") || "Invalid domain")
      return
    }
    if (allowedDomains.some((item) => item.toLowerCase() === domain)) {
      setNewDomain("")
      return
    }
    const ok = await applyUpdate(
      { allowed_domains: [...allowedDomains, domain] },
      t("workforces.widget.messages.updated") || "Widget configuration updated",
    )
    if (ok) setNewDomain("")
  }

  const handleRemoveDomain = (domain: string) => {
    void applyUpdate(
      { allowed_domains: allowedDomains.filter((d) => d !== domain) },
      t("workforces.widget.messages.updated") || "Widget configuration updated",
    )
  }

  const handleRotate = async () => {
    if (!workforce) return
    // Rotating immediately invalidates the key on every site that has it
    // embedded — require explicit confirmation before calling the API.
    if (!window.confirm(t("workforces.widget.rotate_confirm") || "Rotating the widget key will immediately break all existing embeds. Re-copy and redeploy the snippet after rotation. Continue?")) {
      return
    }
    try {
      setIsRotating(true)
      const state = await rotateWorkforceWidgetKey(workforce.id)
      setConfig(state)
      toast.success(t("workforces.widget.messages.rotated") || "Widget key rotated")
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("workforces.widget.messages.failed") || "Widget action failed")
    } finally {
      setIsRotating(false)
    }
  }

  const handleCopySnippet = async () => {
    const snippet = buildWidgetSnippet(widgetKey ?? "", appOrigin)
    if (!snippet) return
    if (await copyToClipboard(snippet)) {
      setCopied(true)
      toast.success(t("workforces.widget.messages.copied") || "Copied to clipboard")
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(next) => { if (!next) onClose() }}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <LayoutGrid className="h-5 w-5" />
            {t("workforces.widget.title") || "Embed Widget"}
          </DialogTitle>
          <DialogDescription>{workforce?.name}</DialogDescription>
        </DialogHeader>

        {!isActive ? (
          <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
            {t("workforces.widget.publish_required") || "Please publish this workforce before enabling the widget."}
          </div>
        ) : isLoading ? (
          <div className="pt-2 text-sm text-muted-foreground">
            {t("common.loading") || "Loading..."}
          </div>
        ) : (
          <div className="space-y-6">
            <div className="space-y-4 border rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label className="text-base">{t("workforces.widget.widget_enabled") || "Widget Enabled"}</Label>
                  <div className="text-sm text-muted-foreground">
                    {t("workforces.widget.widget_enabled_desc") || "Allow this widget to be embedded on external sites."}
                  </div>
                </div>
                <Switch checked={widgetEnabled} onCheckedChange={handleToggle} disabled={isUpdating} />
              </div>

              {widgetEnabled && (
                <div className="space-y-3 pt-4 border-t">
                  <div className="space-y-0.5">
                    <Label className="text-base">{t("workforces.widget.allowed_domains") || "Allowed Domains"}</Label>
                    <div className="text-sm text-muted-foreground">
                      {t("workforces.widget.allowed_domains_desc") || "Restrict widget embedding to specific domains. Use * for any domain."}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Input
                      placeholder={t("workforces.widget.domain_placeholder") || "e.g. example.com"}
                      value={newDomain}
                      onChange={(e) => setNewDomain(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && void handleAddDomain()}
                      disabled={isUpdating}
                      className="flex-1"
                    />
                    <Button onClick={() => void handleAddDomain()} disabled={isUpdating || !newDomain.trim()}>
                      {t("workforces.widget.add_btn") || "Add"}
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {allowedDomains.map((domain) => (
                      <Badge key={domain} variant="secondary" className="flex items-center gap-1 px-3 py-1 text-sm">
                        {domain}
                        <button
                          onClick={() => handleRemoveDomain(domain)}
                          disabled={isUpdating}
                          className="text-muted-foreground hover:text-foreground"
                        >
                          ×
                        </button>
                      </Badge>
                    ))}
                    {allowedDomains.length === 0 && (
                      <span className="text-sm text-muted-foreground italic">
                        {t("workforces.widget.no_domains") || "No domains configured. The widget will block all embeds unless * is added."}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>

            {widgetEnabled && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="font-medium">{t("workforces.widget.snippet_title") || "Embed Snippet"}</div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => void handleRotate()}
                    disabled={isRotating || !widgetKey}
                    title={t("workforces.widget.rotate_btn") || "Rotate Key"}
                  >
                    {isRotating
                      ? t("workforces.widget.rotating") || "Rotating…"
                      : t("workforces.widget.rotate_btn") || "Rotate Key"}
                  </Button>
                </div>
                <div className="text-sm text-muted-foreground">
                  {t("workforces.widget.snippet_desc") || "Copy and paste this script tag into the <body> of your website."}
                </div>
                <div className="bg-muted p-4 rounded-md text-xs font-mono relative overflow-hidden group mt-4">
                  <pre className="whitespace-pre-wrap break-all text-muted-foreground">
                    {widgetKey ? buildWidgetSnippet(widgetKey, appOrigin) : "…"}
                  </pre>
                  <Button
                    variant="secondary"
                    size="icon"
                    className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={() => void handleCopySnippet()}
                    title={t("workforces.widget.copy_btn") || "Copy Snippet"}
                  >
                    {copied ? <Check className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
