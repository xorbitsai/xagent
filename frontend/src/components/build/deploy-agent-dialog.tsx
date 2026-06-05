"use client"

import React, { useEffect, useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Rocket, LayoutGrid, Code2, Share, Webhook, ArrowRight, Copy, Check } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "@/components/ui/sonner"
import { getApiUrl } from "@/lib/utils"
import { copyToClipboard } from "@/lib/clipboard"
import { apiRequest } from "@/lib/api-wrapper"
type ApiSnippetTab = "curl" | "python"

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
  access?: string
  readonly?: boolean
  can_edit?: boolean
  can_publish?: boolean
  can_delete?: boolean
  share_enabled?: boolean
  share_updated_at?: string | null
}

interface ShareLinkResponse {
  agent_id: number
  share_enabled: boolean
  share_token: string | null
  share_updated_at: string | null
}

interface DeployAgentDialogProps {
  deployAgent: Agent | null
  onClose: () => void
  onUpdate: (updatedAgent: Agent) => void
  // Opens the shared API-key dialog (a single instance lives in the parent),
  // so this dialog never nests its own Radix Dialog.
  onManageApiKey?: () => void
}

export function DeployAgentDialog({ deployAgent, onClose, onUpdate, onManageApiKey }: DeployAgentDialogProps) {
  const { t } = useI18n()
  const [activeView, setActiveView] = useState<"options" | "embed" | "api" | "share">("options")
  const [apiTab, setApiTab] = useState<ApiSnippetTab>("curl")
  const [copiedSnippet, setCopiedSnippet] = useState(false)
  const [copiedShareLink, setCopiedShareLink] = useState(false)
  const [isUpdatingWidget, setIsUpdatingWidget] = useState(false)
  const [isUpdatingShare, setIsUpdatingShare] = useState(false)
  const [isLoadingShareLink, setIsLoadingShareLink] = useState(false)
  const [shareLink, setShareLink] = useState<ShareLinkResponse | null>(null)
  const [newDomain, setNewDomain] = useState("")
  const appOrigin = typeof window !== "undefined" ? window.location.origin : getApiUrl()
  const isPublished = deployAgent?.status === "published"
  const shareEnabled = shareLink?.share_enabled ?? deployAgent?.share_enabled ?? false
  const shareUrl = shareLink?.share_token ? `${appOrigin}/share/${shareLink.share_token}` : ""

  const agentId = deployAgent?.id ?? 0
  const apiSnippets: Record<ApiSnippetTab, string> = useMemo(() => {
    const apiBase =
      getApiUrl() || (typeof window !== "undefined" ? window.location.origin : "")
    return {
      curl: `curl -X POST ${apiBase}/v1/chat/tasks \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "agent_id": ${agentId},
    "message": { "role": "user", "content": "Hello" }
  }'`,
      python: `# pip install "xagent-sdk @ git+https://github.com/xorbitsai/xagent-sdk@v0.3.0#subdirectory=python"
from xagent_sdk import AgentClient

with AgentClient(api_key="YOUR_API_KEY", base_url="${apiBase}") as agent:
    result = agent.tasks.run(agent_id=${agentId}, message="Hello")
    print(result.output)`,
    }
  }, [agentId])

  useEffect(() => {
    setShareLink(null)
    setCopiedShareLink(false)
  }, [deployAgent?.id])

  useEffect(() => {
    if (activeView !== "share" || !deployAgent || !isPublished) {
      return
    }

    if (!deployAgent.share_enabled) {
      setShareLink({
        agent_id: deployAgent.id,
        share_enabled: false,
        share_token: null,
        share_updated_at: deployAgent.share_updated_at ?? null,
      })
      return
    }

    if (shareLink?.agent_id === deployAgent.id && shareLink.share_token) {
      return
    }

    let cancelled = false

    const loadShareLink = async () => {
      try {
        setIsLoadingShareLink(true)
        const res = await apiRequest(`${getApiUrl()}/api/agents/${deployAgent.id}/share-link`)
        if (!res.ok) {
          const errorData = await res.json().catch(() => null)
          throw new Error(errorData?.detail || "Failed to load share link")
        }
        const shareData = await res.json() as ShareLinkResponse
        if (!cancelled) {
          setShareLink(shareData)
        }
      } catch (err) {
        if (!cancelled) {
          console.error(err)
          toast.error(t("deploy_agent.messages.share_failed") || "Share link action failed")
        }
      } finally {
        if (!cancelled) {
          setIsLoadingShareLink(false)
        }
      }
    }

    void loadShareLink()

    return () => {
      cancelled = true
    }
  }, [activeView, deployAgent, isPublished, shareLink?.agent_id, shareLink?.share_token, t])

  const syncShareState = (response: ShareLinkResponse) => {
    setShareLink(response)
    if (!deployAgent) return
    onUpdate({
      ...deployAgent,
      share_enabled: response.share_enabled,
      share_updated_at: response.share_updated_at,
    })
  }

  const handleCopyApiSnippet = async () => {
    if (await copyToClipboard(apiSnippets[apiTab])) {
      setCopiedSnippet(true)
      toast.success(t("deploy_agent.messages.copied") || "Copied to clipboard")
      setTimeout(() => setCopiedSnippet(false), 2000)
    } else {
      toast.error(t("deploy_agent.messages.copy_failed") || "Failed to copy to clipboard")
    }
  }

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
    const snippet = `<script
  src="${appOrigin}/widget.js"
  data-agent-id="${deployAgent.id}"
  data-button-size="60px"
  data-button-color="#000"
  data-icon-color="#fff"
  data-panel-bg-color="#fff">
</script>`
    navigator.clipboard.writeText(snippet)
    setCopiedSnippet(true)
    toast.success(t("deploy_agent.messages.copied") || "Copied to clipboard")
    setTimeout(() => setCopiedSnippet(false), 2000)
  }

  const handleCopyShareLink = () => {
    if (!shareUrl) return
    navigator.clipboard.writeText(shareUrl)
    setCopiedShareLink(true)
    toast.success(t("deploy_agent.messages.link_copied") || "Link copied to clipboard")
    setTimeout(() => setCopiedShareLink(false), 2000)
  }

  const handleEnableShare = async () => {
    if (!deployAgent) return
    try {
      setIsUpdatingShare(true)
      const res = await apiRequest(`${getApiUrl()}/api/agents/${deployAgent.id}/share-link`, {
        method: "POST",
      })
      if (!res.ok) {
        const errorData = await res.json().catch(() => null)
        throw new Error(errorData?.detail || "Failed to generate share link")
      }
      const shareData = await res.json() as ShareLinkResponse
      syncShareState(shareData)
      toast.success(t("deploy_agent.messages.share_enabled") || "Share link generated")
    } catch (err) {
      console.error(err)
      toast.error(t("deploy_agent.messages.share_failed") || "Failed to generate share link")
    } finally {
      setIsUpdatingShare(false)
    }
  }

  const handleRotateShare = async () => {
    if (!deployAgent) return
    try {
      setIsUpdatingShare(true)
      const res = await apiRequest(`${getApiUrl()}/api/agents/${deployAgent.id}/share-link/rotate`, {
        method: "POST",
      })
      if (!res.ok) {
        const errorData = await res.json().catch(() => null)
        throw new Error(errorData?.detail || "Failed to rotate share link")
      }
      const shareData = await res.json() as ShareLinkResponse
      syncShareState(shareData)
      toast.success(t("deploy_agent.messages.share_rotated") || "Share link rotated")
    } catch (err) {
      console.error(err)
      toast.error(t("deploy_agent.messages.share_failed") || "Failed to rotate share link")
    } finally {
      setIsUpdatingShare(false)
    }
  }

  const handleDisableShare = async () => {
    if (!deployAgent) return
    try {
      setIsUpdatingShare(true)
      const res = await apiRequest(`${getApiUrl()}/api/agents/${deployAgent.id}/share-link`, {
        method: "DELETE",
      })
      if (!res.ok) {
        const errorData = await res.json().catch(() => null)
        throw new Error(errorData?.detail || "Failed to disable share link")
      }
      const shareData = await res.json() as ShareLinkResponse
      syncShareState(shareData)
      setCopiedShareLink(false)
      toast.success(t("deploy_agent.messages.share_disabled") || "Share link disabled")
    } catch (err) {
      console.error(err)
      toast.error(t("deploy_agent.messages.share_failed") || "Failed to disable share link")
    } finally {
      setIsUpdatingShare(false)
    }
  }

  const handleOpenChange = (open: boolean) => {
    if (!open) {
      onClose()
      // Reset state when closing
      setTimeout(() => {
        setActiveView("options")
        setApiTab("curl")
      }, 300)
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
      onClick: () => setActiveView("embed"),
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
      className: "cursor-pointer hover:border-primary transition-colors shadow-sm",
      onClick: () => setActiveView("api"),
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
      className: "cursor-pointer hover:border-primary transition-colors shadow-sm",
      onClick: () => setActiveView("share"),
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

        {activeView === "options" ? (
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
        ) : activeView === "api" ? (
          <div className="mt-4 space-y-4">
            <div className="flex items-center text-sm text-muted-foreground cursor-pointer hover:text-foreground" onClick={() => setActiveView("options")}>
              <ArrowRight className="h-4 w-4 mr-1 rotate-180" /> {t("deploy_agent.back_to_options") || "Back to Deploy Options"}
            </div>

            <div className="space-y-1">
              <div className="font-medium">{t("deploy_agent.api_panel.title") || "Call this agent via REST API"}</div>
              <div className="text-sm text-muted-foreground">
                {t("deploy_agent.api_panel.desc") || "Submit a task to the agent. Poll GET /v1/chat/tasks/{id} for the result."}
              </div>
            </div>

            <div className="flex gap-1 border-b">
              {(["curl", "python"] as ApiSnippetTab[]).map((tab) => (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setApiTab(tab)}
                  className={`px-3 py-1.5 text-sm font-medium border-b-2 -mb-px transition-colors ${apiTab === tab ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"}`}
                >
                  {tab === "curl" ? "cURL" : "Python"}
                </button>
              ))}
            </div>

            <div className="bg-muted p-4 rounded-md text-xs font-mono relative overflow-hidden group">
              <pre className="whitespace-pre-wrap break-all text-muted-foreground max-h-80 overflow-auto">
                {apiSnippets[apiTab]}
              </pre>
              <Button
                variant="secondary"
                size="icon"
                className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={handleCopyApiSnippet}
                title={t("deploy_agent.api_panel.copy_btn") || "Copy"}
              >
                {copiedSnippet ? <Check className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
              </Button>
            </div>

            <div className="text-sm text-muted-foreground">
              {t("deploy_agent.api_panel.key_hint") || "Replace YOUR_API_KEY with this agent's API key."}{" "}
              <button
                type="button"
                className="text-primary hover:underline font-medium"
                onClick={() => onManageApiKey?.()}
              >
                {t("deploy_agent.api_panel.manage_key") || "Manage API Key"}
              </button>
            </div>
          </div>
        ) : activeView === "embed" ? (
          <div className="mt-4 space-y-6">
            <div className="flex items-center text-sm text-muted-foreground cursor-pointer hover:text-foreground" onClick={() => setActiveView("options")}>
              <ArrowRight className="h-4 w-4 mr-1 rotate-180" /> {t("deploy_agent.back_to_options") || "Back to Deploy Options"}
            </div>

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
  src="${appOrigin}/widget.js"
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
                  {copiedSnippet ? <Check className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <div className="mt-4 space-y-6">
            <div className="flex items-center text-sm text-muted-foreground cursor-pointer hover:text-foreground" onClick={() => setActiveView("options")}>
              <ArrowRight className="h-4 w-4 mr-1 rotate-180" /> {t("deploy_agent.back_to_options") || "Back to Deploy Options"}
            </div>

            <div className="space-y-4 border rounded-lg p-4">
              <div className="space-y-1">
                <div className="text-base font-medium">{t("deploy_agent.share_link.title") || "Share Link"}</div>
                <div className="text-sm text-muted-foreground">
                  {t("deploy_agent.share_link.desc") || "Generate a public page anyone can open to chat with this agent."}
                </div>
              </div>

              {!isPublished ? (
                <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                  {t("deploy_agent.share_link.publish_required") || "Please publish this agent before generating a share link."}
                </div>
              ) : isLoadingShareLink ? (
                <div className="pt-2 text-sm text-muted-foreground">
                  {t("common.loading") || "Loading..."}
                </div>
              ) : shareEnabled && shareUrl ? (
                <div className="space-y-4 pt-2">
                  <div className="space-y-2">
                    <Label className="text-sm">{t("deploy_agent.share_link.public_url") || "Public URL"}</Label>
                    <div className="flex gap-2">
                      <Input readOnly value={shareUrl} className="flex-1" />
                      <Button variant="secondary" onClick={handleCopyShareLink} disabled={isUpdatingShare}>
                        {copiedShareLink ? <Check className="h-4 w-4 mr-1 text-green-500" /> : <Copy className="h-4 w-4 mr-1" />}
                        {t("common.copy") || "Copy"}
                      </Button>
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {t("deploy_agent.share_link.anyone_access") || "Anyone with this link can start a public chat with this agent."}
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" onClick={handleRotateShare} disabled={isUpdatingShare}>
                      {t("deploy_agent.share_link.rotate_btn") || "Reset Link"}
                    </Button>
                    <Button variant="outline" onClick={handleDisableShare} disabled={isUpdatingShare}>
                      {t("deploy_agent.share_link.disable_btn") || "Disable Link"}
                    </Button>
                  </div>
                </div>
              ) : shareEnabled ? (
                <div className="space-y-4 pt-2">
                  <div className="text-sm text-muted-foreground">
                    {t("deploy_agent.messages.share_failed") || "Share link action failed"}
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" onClick={handleRotateShare} disabled={isUpdatingShare}>
                      {t("deploy_agent.share_link.rotate_btn") || "Reset Link"}
                    </Button>
                    <Button variant="outline" onClick={handleDisableShare} disabled={isUpdatingShare}>
                      {t("deploy_agent.share_link.disable_btn") || "Disable Link"}
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="pt-2">
                  <Button onClick={handleEnableShare} disabled={isUpdatingShare}>
                    {t("deploy_agent.share_link.generate_btn") || "Generate Link"}
                  </Button>
                </div>
              )}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog >
  )
}
