"use client"

import React from "react"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Rocket, LayoutGrid, Code2, Share, Webhook, ArrowRight } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"

interface WorkforceDeployHubDialogProps {
  open: boolean
  onClose: () => void
  workforceName: string
  onSelectEmbed: () => void
  onSelectApi: () => void
  onSelectShare: () => void
  onSelectWebhook: () => void
}

/**
 * Entry-point "options" grid for deploying a workforce — mirrors
 * DeployAgentDialog's options view, but delegates each option to the
 * existing standalone dialogs (DeployWorkforceDialog, WorkforceShareDialog,
 * WorkforceWidgetDialog, AgentTriggersDialog) instead of reimplementing
 * their logic inline.
 */
export function WorkforceDeployHubDialog({
  open,
  onClose,
  workforceName,
  onSelectEmbed,
  onSelectApi,
  onSelectShare,
  onSelectWebhook,
}: WorkforceDeployHubDialogProps) {
  const { t } = useI18n()

  const options = [
    {
      id: "embed",
      icon: LayoutGrid,
      iconColor: "text-blue-600",
      iconBg: "bg-blue-100",
      title: t("workforces.deployHub.options.embed.title"),
      desc: t("workforces.deployHub.options.embed.desc"),
      actionText: t("workforces.deployHub.options.embed.action"),
      actionColor: "text-blue-600",
      onClick: onSelectEmbed,
    },
    {
      id: "rest_api",
      icon: Code2,
      iconColor: "text-purple-600",
      iconBg: "bg-purple-100",
      title: t("workforces.deployHub.options.api.title"),
      desc: t("workforces.deployHub.options.api.desc"),
      actionText: t("workforces.deployHub.options.api.action"),
      actionColor: "text-purple-600",
      onClick: onSelectApi,
    },
    {
      id: "shareable_link",
      icon: Share,
      iconColor: "text-indigo-600",
      iconBg: "bg-indigo-100",
      title: t("workforces.deployHub.options.share.title"),
      desc: t("workforces.deployHub.options.share.desc"),
      actionText: t("workforces.deployHub.options.share.action"),
      actionColor: "text-indigo-600",
      onClick: onSelectShare,
    },
    {
      id: "webhook",
      icon: Webhook,
      iconColor: "text-emerald-600",
      iconBg: "bg-emerald-100",
      title: t("workforces.deployHub.options.webhook.title"),
      desc: t("workforces.deployHub.options.webhook.desc"),
      actionText: t("workforces.deployHub.options.webhook.action"),
      actionColor: "text-emerald-600",
      onClick: onSelectWebhook,
    },
  ]

  return (
    <Dialog open={open} onOpenChange={(next) => { if (!next) onClose() }}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Rocket className="h-5 w-5" />
            {t("workforces.deployHub.title")}
          </DialogTitle>
          <DialogDescription>{workforceName}</DialogDescription>
        </DialogHeader>

        <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">
          {options.map((option) => (
            <Card
              key={option.id}
              role="button"
              tabIndex={0}
              onClick={option.onClick}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault()
                  option.onClick()
                }
              }}
              className="cursor-pointer shadow-sm transition-colors hover:border-primary focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <CardHeader>
                <div className={`mb-2 flex h-10 w-10 items-center justify-center rounded-lg ${option.iconBg}`}>
                  <option.icon className={`h-5 w-5 ${option.iconColor}`} />
                </div>
                <CardTitle className="text-base font-semibold">{option.title}</CardTitle>
                <CardDescription className="mt-1 text-xs">{option.desc}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className={`flex items-center text-sm font-medium ${option.actionColor}`}>
                  {option.actionText} <ArrowRight className="ml-1 h-4 w-4" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}
