"use client"

import React, { useEffect, useState } from "react"
import { Check, Copy, Share } from "lucide-react"
import { DeploymentConfigErrorAlert } from "@/components/deployment/deployment-config-error-alert"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "@/components/ui/sonner"
import { copyToClipboard } from "@/lib/clipboard"
import { getBrowserLocationOrigin } from "@/lib/browser-location"
import {
  DEPLOYMENT_CONFIG_LOAD_FAILED_FALLBACK,
  buildDeploymentShareUrl,
  fetchDeploymentConfig,
  type DeploymentConfig,
} from "@/lib/deployment-config"
import {
  disableWorkforceShareLink,
  enableWorkforceShareLink,
  getWorkforceShareLink,
  rotateWorkforceShareLink,
} from "@/lib/workforces-api"
import type { WorkforceSummary, WorkforceShareLink } from "@/types/workforce"

interface WorkforceShareDialogProps {
  workforce: WorkforceSummary | null
  open: boolean
  onClose: () => void
}

export function WorkforceShareDialog({ workforce, open, onClose }: WorkforceShareDialogProps) {
  const { t } = useI18n()
  const [shareLink, setShareLink] = useState<WorkforceShareLink | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isUpdating, setIsUpdating] = useState(false)
  const [copied, setCopied] = useState(false)
  const [appOrigin, setAppOrigin] = useState("")
  const [deploymentConfig, setDeploymentConfig] =
    useState<DeploymentConfig | null>(null)
  const [deploymentConfigFailed, setDeploymentConfigFailed] = useState(false)

  const isActive = workforce?.status === "active"
  const shareEnabled = shareLink?.share_enabled ?? false
  const shareUrl = shareLink?.share_token
    ? buildDeploymentShareUrl(
        shareLink.share_token,
        deploymentConfig,
        appOrigin,
      )
    : ""

  useEffect(() => {
    setAppOrigin(getBrowserLocationOrigin())
  }, [])

  useEffect(() => {
    if (!open || !workforce || !isActive) {
      return
    }
    let cancelled = false
    const load = async () => {
      try {
        setIsLoading(true)
        const [stateResult, targetsResult] = await Promise.allSettled([
          getWorkforceShareLink(workforce.id),
          fetchDeploymentConfig(),
        ])
        if (cancelled) return

        if (targetsResult.status === "fulfilled") {
          setDeploymentConfig(targetsResult.value)
          setDeploymentConfigFailed(false)
        } else {
          console.error(targetsResult.reason)
          setDeploymentConfig(null)
          setDeploymentConfigFailed(true)
          toast.error(
            t("deployment_config.messages.load_failed")
            || DEPLOYMENT_CONFIG_LOAD_FAILED_FALLBACK,
          )
        }

        if (stateResult.status === "rejected") throw stateResult.reason
        setShareLink(stateResult.value)
      } catch (err) {
        if (!cancelled) {
          console.error(err)
          toast.error(t("workforces.share_link.messages.failed") || "Share link action failed")
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
  }, [open, workforce?.id, isActive])

  const retryDeploymentConfig = async () => {
    try {
      setDeploymentConfig(await fetchDeploymentConfig())
      setDeploymentConfigFailed(false)
    } catch (error) {
      console.error(error)
      toast.error(
        t("deployment_config.messages.load_failed")
        || DEPLOYMENT_CONFIG_LOAD_FAILED_FALLBACK,
      )
    }
  }

  const runShareAction = async (
    action: (id: number) => Promise<WorkforceShareLink>,
    successMessage: string,
  ) => {
    if (!workforce) return
    try {
      setIsUpdating(true)
      const state = await action(workforce.id)
      setShareLink(state)
      toast.success(successMessage)
    } catch (err) {
      console.error(err)
      toast.error(t("workforces.share_link.messages.failed") || "Share link action failed")
    } finally {
      setIsUpdating(false)
    }
  }

  const handleCopy = async () => {
    if (!shareUrl) return
    if (await copyToClipboard(shareUrl)) {
      setCopied(true)
      toast.success(t("workforces.share_link.messages.link_copied") || "Link copied to clipboard")
      setTimeout(() => setCopied(false), 2000)
    } else {
      toast.error(
        t("workforces.share_link.messages.copy_failed")
        || "Failed to copy to clipboard",
      )
    }
  }

  return (
    <Dialog open={open} onOpenChange={(next) => { if (!next) onClose() }}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Share className="h-5 w-5" />
            {t("workforces.share_link.title") || "Share Workforce"}
          </DialogTitle>
          <DialogDescription>{workforce?.name}</DialogDescription>
        </DialogHeader>

        {deploymentConfigFailed && (
          <DeploymentConfigErrorAlert onRetry={retryDeploymentConfig} />
        )}

        <div className="space-y-4 border rounded-lg p-4">
          <div className="space-y-1">
            <div className="text-base font-medium">
              {t("workforces.share_link.section_title") || "Public Share Link"}
            </div>
            <div className="text-sm text-muted-foreground">
              {t("workforces.share_link.desc") || "Generate a public page anyone can open to chat with this workforce."}
            </div>
          </div>

          {!isActive ? (
            <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
              {t("workforces.share_link.publish_required") || "Please publish this workforce before generating a share link."}
            </div>
          ) : isLoading ? (
            <div className="pt-2 text-sm text-muted-foreground">
              {t("common.loading") || "Loading..."}
            </div>
          ) : shareEnabled ? (
            <div className="space-y-4 pt-2">
              <div className="space-y-2">
                <Label className="text-sm">{t("workforces.share_link.public_url") || "Public URL"}</Label>
                <div className="flex gap-2">
                  <Input readOnly value={shareUrl} className="flex-1" />
                  <Button variant="secondary" onClick={() => void handleCopy()} disabled={isUpdating || !shareUrl}>
                    {copied ? <Check className="h-4 w-4 mr-1 text-green-500" /> : <Copy className="h-4 w-4 mr-1" />}
                    {t("common.copy") || "Copy"}
                  </Button>
                </div>
              </div>
              <div className="text-xs text-muted-foreground">
                {t("workforces.share_link.anyone_access") || "Anyone with this link can start a public chat with this workforce."}
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={() => void runShareAction(rotateWorkforceShareLink, t("workforces.share_link.messages.rotated") || "Share link rotated")}
                  disabled={isUpdating}
                >
                  {t("workforces.share_link.rotate_btn") || "Reset Link"}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => void runShareAction(disableWorkforceShareLink, t("workforces.share_link.messages.disabled") || "Share link disabled")}
                  disabled={isUpdating}
                >
                  {t("workforces.share_link.disable_btn") || "Disable Link"}
                </Button>
              </div>
            </div>
          ) : (
            <div className="pt-2">
              <Button
                onClick={() => void runShareAction(enableWorkforceShareLink, t("workforces.share_link.messages.enabled") || "Share link generated")}
                disabled={isUpdating}
              >
                {t("workforces.share_link.generate_btn") || "Generate Link"}
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
