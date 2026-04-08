"use client"

import React, { useState } from "react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Rocket, LayoutGrid, Code2, Share, Webhook, ArrowRight, Copy, Check, MoreVertical } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "sonner"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"

export interface Agent {
  id: number
  name: string
  description: string
  logo_url: string | null
  status: string
  created_at: string
  updated_at: string
  widget_enabled: boolean
  allowed_domains: string[]
}

interface DeployAgentDialogProps {
  deployAgent: Agent | null
  onClose: () => void
  onUpdate: (updatedAgent: Agent) => void
}

export function DeployAgentDialog({ deployAgent, onClose, onUpdate }: DeployAgentDialogProps) {
  const { t } = useI18n()
  const [showSnippet, setShowSnippet] = useState(false)
  const [copied, setCopied] = useState(false)
  const [isUpdatingWidget, setIsUpdatingWidget] = useState(false)
  const [newDomain, setNewDomain] = useState("")

  const handleUpdateWidgetConfig = async (updates: { widget_enabled?: boolean, allowed_domains?: string[] }) => {
    if (!deployAgent) return
    try {
      setIsUpdatingWidget(true)
      const res = await apiRequest(`${getApiUrl()}/api/agents/${deployAgent.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates)
      })
      if (!res.ok) throw new Error("Failed to update widget config")
      const updatedAgent = await res.json()
      onUpdate(updatedAgent)
      toast.success(t("deploy_agent.messages.update_success") || "Widget configuration updated")
    } catch (err) {
      console.error(err)
      toast.error(t("deploy_agent.messages.update_failed") || "Failed to update widget configuration")
    } finally {
      setIsUpdatingWidget(false)
    }
  }

  const handleAddDomain = () => {
    if (!newDomain.trim() || !deployAgent) return
    const domain = newDomain.trim()
    const currentDomains = deployAgent.allowed_domains || []
    if (currentDomains.includes(domain)) {
      setNewDomain("")
      return
    }
    handleUpdateWidgetConfig({ allowed_domains: [...currentDomains, domain] })
    setNewDomain("")
  }

  const handleRemoveDomain = (domain: string) => {
    if (!deployAgent) return
    const currentDomains = deployAgent.allowed_domains || []
    handleUpdateWidgetConfig({ allowed_domains: currentDomains.filter(d => d !== domain) })
  }

  const handleCopySnippet = () => {
    if (!deployAgent) return
    const origin = typeof window !== 'undefined' ? window.location.origin : getApiUrl()
    const snippet = `<script
  src="${origin}/widget.js"
  data-agent-id="${deployAgent.id}"
  data-button-size="60px"
  data-button-color="#000"
  data-icon-color="#fff"
  data-panel-bg-color="#fff">
</script>`
    navigator.clipboard.writeText(snippet)
    setCopied(true)
    toast.success(t("deploy_agent.messages.copied") || "Copied to clipboard")
    setTimeout(() => setCopied(false), 2000)
  }

  const handleOpenChange = (open: boolean) => {
    if (!open) {
      onClose()
      // Reset state when closing
      setTimeout(() => setShowSnippet(false), 300)
    }
  }

  const deploymentOptions = [
    {
      id: "embed",
      icon: LayoutGrid,
      iconColor: "text-blue-600",
      iconBg: "bg-blue-100",
      title: t("deploy_agent.options.embed.title") || "Embed Widget",
      desc: t("deploy_agent.options.embed.desc") || "Add a chat widget to any website with a single script tag",
      actionText: t("deploy_agent.options.embed.action") || "Get snippet",
      actionColor: "text-blue-600",
      className: "cursor-pointer hover:border-primary transition-colors shadow-sm",
      onClick: () => setShowSnippet(true),
    },
    {
      id: "rest_api",
      icon: Code2,
      iconColor: "text-purple-600",
      iconBg: "bg-purple-100",
      title: t("deploy_agent.options.rest_api.title") || "REST API",
      desc: t("deploy_agent.options.rest_api.desc") || "Call the agent programmatically from your backend or app",
      actionText: t("deploy_agent.options.rest_api.action") || "View endpoints",
      actionColor: "text-purple-600",
      className: "opacity-50 cursor-not-allowed shadow-sm",
    },
    {
      id: "shareable_link",
      icon: Share,
      iconColor: "text-indigo-600",
      iconBg: "bg-indigo-100",
      title: t("deploy_agent.options.shareable_link.title") || "Shareable Link",
      desc: t("deploy_agent.options.shareable_link.desc") || "Generate a public URL anyone can open to chat with this agent",
      actionText: t("deploy_agent.options.shareable_link.action") || "Generate link",
      actionColor: "text-indigo-600",
      className: "opacity-50 cursor-not-allowed shadow-sm",
    },
    {
      id: "webhook",
      icon: Webhook,
      iconColor: "text-emerald-600",
      iconBg: "bg-emerald-100",
      title: t("deploy_agent.options.webhook.title") || "Webhook",
      desc: t("deploy_agent.options.webhook.desc") || "Trigger agent runs via webhook events from external systems",
      actionText: t("deploy_agent.options.webhook.action") || "Configure",
      actionColor: "text-emerald-600",
      className: "opacity-50 cursor-not-allowed shadow-sm",
    },
  ]

  return (
    <Dialog open={deployAgent !== null} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Rocket className="h-5 w-5" />
            {t("deploy_agent.title") || "Deploy Agent"}
          </DialogTitle>
          <DialogDescription>{deployAgent?.name}</DialogDescription>
        </DialogHeader>

        {!showSnippet ? (
          <div className="mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {deploymentOptions.map((option) => (
                <Card
                  key={option.id}
                  className={option.className}
                  onClick={option.onClick}
                >
                  <CardHeader>
                    <div className={`h-10 w-10 rounded-lg ${option.iconBg} flex items-center justify-center mb-2`}>
                      <option.icon className={`h-5 w-5 ${option.iconColor}`} />
                    </div>
                    <CardTitle className="text-base font-semibold">{option.title}</CardTitle>
                    <CardDescription className="text-xs mt-1">
                      {option.desc}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className={`text-sm ${option.actionColor} font-medium flex items-center`}>
                      {option.actionText} <ArrowRight className="h-4 w-4 ml-1" />
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        ) : (
          <div className="mt-4 space-y-6">
            <div className="flex items-center text-sm text-muted-foreground cursor-pointer hover:text-foreground" onClick={() => setShowSnippet(false)}>
              <ArrowRight className="h-4 w-4 mr-1 rotate-180" /> {t("deploy_agent.back_to_options") || "Back to Deploy Options"}
            </div>

            {/* Access Control Section */}
            <div className="space-y-4 border rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label className="text-base">{t("deploy_agent.access_control.widget_enabled") || "Widget Enabled"}</Label>
                  <div className="text-sm text-muted-foreground">
                    {t("deploy_agent.access_control.widget_enabled_desc") || "Allow this widget to be accessed externally."}
                  </div>
                </div>
                <Switch
                  checked={deployAgent?.widget_enabled}
                  onCheckedChange={(checked) => handleUpdateWidgetConfig({ widget_enabled: checked })}
                  disabled={isUpdatingWidget}
                />
              </div>

              {deployAgent?.widget_enabled && (
                <div className="space-y-3 pt-4 border-t">
                  <div className="space-y-0.5">
                    <Label className="text-base">{t("deploy_agent.access_control.allowed_domains") || "Allowed Domains"}</Label>
                    <div className="text-sm text-muted-foreground">
                      {t("deploy_agent.access_control.allowed_domains_desc") || "Restrict widget access to specific domains. Use * for any domain."}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Input
                      placeholder={t("deploy_agent.access_control.domain_placeholder") || "e.g. example.com"}
                      value={newDomain}
                      onChange={(e) => setNewDomain(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleAddDomain()}
                      disabled={isUpdatingWidget}
                      className="flex-1"
                    />
                    <Button onClick={handleAddDomain} disabled={isUpdatingWidget || !newDomain.trim()}>
                      {t("deploy_agent.access_control.add_btn") || "Add"}
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {(deployAgent?.allowed_domains || []).map((domain) => (
                      <Badge key={domain} variant="secondary" className="flex items-center gap-1 px-3 py-1 text-sm">
                        {domain}
                        <button
                          onClick={() => handleRemoveDomain(domain)}
                          disabled={isUpdatingWidget}
                          className="text-muted-foreground hover:text-foreground"
                        >
                          ×
                        </button>
                      </Badge>
                    ))}
                    {(deployAgent?.allowed_domains || []).length === 0 && (
                      <span className="text-sm text-muted-foreground italic">
                        {t("deploy_agent.access_control.no_domains") || "No domains configured. Widget will block all requests unless * is added."}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-2">
              <div className="font-medium">{t("deploy_agent.embed_snippet.title") || "Embed Snippet"}</div>
              <div className="text-sm text-muted-foreground">
                {t("deploy_agent.embed_snippet.desc") || "Copy and paste this script tag into the <body> of your website."}
              </div>
              <div className="bg-muted p-4 rounded-md text-xs font-mono relative overflow-hidden group mt-4">
                <pre className="whitespace-pre-wrap break-all text-muted-foreground">
                  {`<script
  src="${typeof window !== 'undefined' ? window.location.origin : getApiUrl()}/widget.js"
  data-agent-id="${deployAgent?.id}"
  data-button-size="60px"
  data-button-color="#000"
  data-icon-color="#fff"
  data-panel-bg-color="#fff">
</script>`}
                </pre>
                <Button
                  variant="secondary"
                  size="icon"
                  className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                  onClick={handleCopySnippet}
                  title={t("deploy_agent.embed_snippet.copy_btn") || "Copy Snippet"}
                >
                  {copied ? <Check className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog >
  )
}
