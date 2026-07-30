"use client"

import { useCallback, useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Select } from "@/components/ui/select"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Edit, ExternalLink, MessageSquare, Plus, Settings2, Trash2 } from "lucide-react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "@/components/ui/sonner"

interface Channel {
  id: number;
  channel_type: string;
  channel_name: string;
  config: {
    bot_token?: string;
    app_id?: string;
    app_secret?: string;
    app_token?: string;
    allowed_users?: string[] | null;
    installation_mode?: "manual" | "oauth";
    workspace_name?: string;
    [key: string]: unknown;
  };
  is_active: boolean;
}

function SlackLogo({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path fill="#36C5F0" d="M9.6 2.5a2.1 2.1 0 1 1-4.2 0 2.1 2.1 0 0 1 4.2 0v5.2H7.5a2.1 2.1 0 0 1 0-4.2h2.1v-1Z" />
      <path fill="#2EB67D" d="M21.5 9.6a2.1 2.1 0 1 1 0-4.2 2.1 2.1 0 0 1 0 4.2h-5.2V7.5a2.1 2.1 0 1 1 4.2 0v2.1h1Z" />
      <path fill="#ECB22E" d="M14.4 21.5a2.1 2.1 0 1 1 4.2 0 2.1 2.1 0 0 1-4.2 0v-5.2h2.1a2.1 2.1 0 1 1 0 4.2h-2.1v1Z" />
      <path fill="#E01E5A" d="M2.5 14.4a2.1 2.1 0 1 1 0 4.2 2.1 2.1 0 0 1 0-4.2h5.2v2.1a2.1 2.1 0 1 1-4.2 0v-2.1h-1Z" />
      <path fill="#36C5F0" d="M11 3.5h2v6.1h-2z" />
      <path fill="#2EB67D" d="M14.4 11h6.1v2h-6.1z" />
      <path fill="#ECB22E" d="M11 14.4h2v6.1h-2z" />
      <path fill="#E01E5A" d="M3.5 11h6.1v2H3.5z" />
    </svg>
  )
}

export default function ChannelsPage() {
  const { t } = useI18n()
  const [channels, setChannels] = useState<Channel[]>([])
  const [loading, setLoading] = useState(true)
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [editingChannel, setEditingChannel] = useState<Channel | null>(null)
  const [isConnectingSlack, setIsConnectingSlack] = useState(false)

  const [formData, setFormData] = useState({
    channel_type: "telegram",
    channel_name: "",
    bot_token: "",
    app_id: "",
    app_secret: "",
    app_token: "",
    allowed_users: "",
    is_active: true
  })

  const fetchChannels = useCallback(async () => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/channels`)
      if (response.ok) {
        const data = await response.json()
        setChannels(data)
      } else {
        toast.error(t("channels.messages.load_failed"))
      }
    } catch (error) {
      console.error("Failed to fetch channels:", error)
      toast.error(t("channels.messages.load_failed"))
    } finally {
      setLoading(false)
    }
  }, [t])

  useEffect(() => {
    void fetchChannels()
  }, [fetchChannels])

  const handleOpenDialog = (channel?: Channel, defaultType: string = "telegram") => {
    if (channel) {
      setEditingChannel(channel)
      setFormData({
        channel_type: channel.channel_type,
        channel_name: channel.channel_name || "",
        bot_token: channel.config.bot_token || "",
        app_id: channel.config.app_id || "",
        app_secret: channel.config.app_secret || "",
        app_token: channel.config.app_token || "",
        allowed_users: channel.config.allowed_users ? channel.config.allowed_users.join(", ") : "",
        is_active: channel.is_active
      })
    } else {
      setEditingChannel(null)
      setFormData({
        channel_type: defaultType,
        channel_name: "",
        bot_token: "",
        app_id: "",
        app_secret: "",
        app_token: "",
        allowed_users: "",
        is_active: true
      })
    }
    setIsDialogOpen(true)
  }

  const handleSubmit = async () => {
    try {
      // The API redacts stored secrets, so edit forms start with empty
      // secret fields; an empty secret on edit means "keep the stored one".
      const isCreating = !editingChannel
      if (formData.channel_type === "telegram" && !formData.bot_token && isCreating) {
        toast.error(t("channels.messages.fill_required"))
        return
      }

      if (
        formData.channel_type === "feishu"
        && (!formData.app_id || (!formData.app_secret && isCreating))
      ) {
        toast.error(t("channels.messages.fill_required"))
        return
      }

      const isOAuthSlackEdit = Boolean(
        editingChannel?.channel_type === "slack"
        && editingChannel.config.installation_mode === "oauth"
      )
      if (
        formData.channel_type === "slack"
        && !isOAuthSlackEdit
        && isCreating
        && (!formData.bot_token || !formData.app_token)
      ) {
        toast.error(t("channels.messages.fill_required"))
        return
      }

      const config: Record<string, unknown> = {
        allowed_users: formData.allowed_users.trim()
          ? formData.allowed_users.split(",").map(u => u.trim()).filter(Boolean)
          : null,
      }
      if (formData.channel_type === "telegram" && formData.bot_token) {
        config.bot_token = formData.bot_token
      } else if (formData.channel_type === "feishu") {
        if (formData.app_id) config.app_id = formData.app_id
        if (formData.app_secret) config.app_secret = formData.app_secret
      } else if (formData.channel_type === "slack" && !isOAuthSlackEdit) {
        config.installation_mode = "manual"
        if (formData.bot_token) config.bot_token = formData.bot_token
        if (formData.app_token) config.app_token = formData.app_token
      }

      const payload = {
        channel_type: formData.channel_type,
        channel_name: formData.channel_name.trim(),
        config,
        is_active: formData.is_active
      }

      if (editingChannel) {
        const res = await apiRequest(`${getApiUrl()}/api/channels/${editingChannel.id}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        })
        if (!res.ok) {
          const data = await res.json()
          let errMsg = data.detail || t("channels.messages.save_failed")
          if (errMsg === "Channel name already exists") errMsg = t("channels.messages.name_exists")
          if (errMsg === "Bot token already exists") errMsg = t("channels.messages.token_exists")
          throw new Error(errMsg)
        }
        toast.success(t("channels.messages.update_success"))
      } else {
        const res = await apiRequest(`${getApiUrl()}/api/channels`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        })
        if (!res.ok) {
          const data = await res.json()
          let errMsg = data.detail || t("channels.messages.save_failed")
          if (errMsg === "Channel name already exists") errMsg = t("channels.messages.name_exists")
          if (errMsg === "Bot token already exists") errMsg = t("channels.messages.token_exists")
          throw new Error(errMsg)
        }
        toast.success(t("channels.messages.create_success"))
      }

      setIsDialogOpen(false)
      fetchChannels()
    } catch (error) {
      console.error("Failed to save channel:", error)
      toast.error(
        error instanceof Error
          ? error.message
          : t("channels.messages.save_failed"),
      )
    }
  }

  const handleSlackOAuthConnect = async () => {
    const width = 620
    const height = 760
    const left = window.screenX + Math.max(0, (window.outerWidth - width) / 2)
    const top = window.screenY + Math.max(0, (window.outerHeight - height) / 2)
    const popup = window.open(
      "about:blank",
      "xagent-slack-oauth",
      `width=${width},height=${height},left=${left},top=${top},scrollbars=yes`,
    )
    if (!popup) {
      toast.error(t("channels.messages.slack_popup_blocked"))
      return
    }

    setIsConnectingSlack(true)
    let popupCheck: ReturnType<typeof setInterval> | undefined
    const apiOrigin = new URL(
      getApiUrl() || window.location.origin,
      window.location.origin,
    ).origin
    let callbackOrigin = apiOrigin

    const cleanup = () => {
      window.removeEventListener("message", handleMessage)
      if (popupCheck) clearInterval(popupCheck)
      setIsConnectingSlack(false)
    }
    const handleMessage = (event: MessageEvent) => {
      if (event.origin !== callbackOrigin || event.source !== popup) return
      if (event.data?.type === "slack-oauth-success") {
        cleanup()
        toast.success(
          event.data.message || t("channels.messages.slack_connect_success"),
        )
        void fetchChannels()
      } else if (event.data?.type === "slack-oauth-error") {
        cleanup()
        toast.error(
          event.data.message || t("channels.messages.slack_connect_failed"),
        )
      }
    }
    window.addEventListener("message", handleMessage)

    try {
      const response = await apiRequest(
        `${getApiUrl()}/api/channels/slack/oauth/start`,
        { method: "POST" },
      )
      const data = await response.json().catch(() => ({}))
      if (!response.ok || typeof data.authorize_url !== "string") {
        throw new Error(data.detail || t("channels.messages.slack_connect_failed"))
      }
      if (typeof data.callback_origin === "string") {
        callbackOrigin = new URL(data.callback_origin).origin
      }
      popup.location.href = data.authorize_url
      popupCheck = setInterval(() => {
        if (popup.closed) cleanup()
      }, 500)
    } catch (error) {
      cleanup()
      popup.close()
      toast.error(
        error instanceof Error
          ? error.message
          : t("channels.messages.slack_connect_failed"),
      )
    }
  }

  const handleDelete = async (id: number) => {
    if (!confirm(t("channels.messages.delete_confirm"))) return

    try {
      await apiRequest(`${getApiUrl()}/api/channels/${id}`, {
        method: "DELETE"
      })
      toast.success(t("channels.messages.delete_success"))
      fetchChannels()
    } catch (error) {
      console.error("Failed to delete channel:", error)
      toast.error(t("channels.messages.delete_failed"))
    }
  }

  const toggleActive = async (channel: Channel) => {
    try {
      await apiRequest(`${getApiUrl()}/api/channels/${channel.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          is_active: !channel.is_active
        })
      })
      fetchChannels()
      toast.success(t("channels.messages.update_success"))
    } catch (error) {
      console.error("Failed to toggle channel status:", error)
      toast.error(t("channels.messages.toggle_failed"))
    }
  }

  return (
    <div className="w-full p-8 space-y-6 overflow-auto">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-[22px] font-bold leading-tight">{t("channels.page_title")}</h1>
          <p className="text-[13px] text-muted-foreground mt-0.5">{t("channels.page_description")}</p>
        </div>
      </div>

      <div className="space-y-6">
        {/* Telegram Bots Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5" />
                {t("channels.telegram_bots")}
              </CardTitle>
              <CardDescription>
                {t("channels.description", { platform: t("channels.telegram_bots") })}
              </CardDescription>
            </div>
            <Button onClick={() => handleOpenDialog(undefined, "telegram")} size="sm">
              <Plus className="h-4 w-4 mr-2" />
              {t("channels.add_telegram")}
            </Button>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-sm text-muted-foreground">{t("common.loading")}</div>
            ) : channels.filter(c => c.channel_type === "telegram").length === 0 ? (
              <div className="text-sm text-muted-foreground py-4 text-center border rounded-md bg-muted/20">
                {t("channels.no_channels")}
              </div>
            ) : (
              <div className="space-y-4">
                {channels.filter(c => c.channel_type === "telegram").map((channel) => (
                  <div key={channel.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center gap-4">
                      <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                        <MessageSquare className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <div className="font-medium">{channel.channel_name}</div>
                        <div className="text-xs text-muted-foreground capitalize">
                          {channel.channel_type} • {channel.is_active ? t("channels.status.active") : t("channels.status.inactive")}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <Switch
                        checked={channel.is_active}
                        onCheckedChange={() => toggleActive(channel)}
                      />
                      <Button variant="ghost" size="icon" onClick={() => handleOpenDialog(channel)} title={t("channels.actions.edit")}>
                        <Edit className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon" onClick={() => handleDelete(channel.id)} title={t("channels.actions.delete")}>
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Feishu Bots Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5" />
                {t("channels.feishu_bots")}
              </CardTitle>
              <CardDescription>
                {t("channels.description", { platform: t("channels.feishu_bots") })}
              </CardDescription>
            </div>
            <Button onClick={() => handleOpenDialog(undefined, "feishu")} size="sm">
              <Plus className="h-4 w-4 mr-2" />
              {t("channels.add_feishu")}
            </Button>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-sm text-muted-foreground">{t("common.loading")}</div>
            ) : channels.filter(c => c.channel_type === "feishu").length === 0 ? (
              <div className="text-sm text-muted-foreground py-4 text-center border rounded-md bg-muted/20">
                {t("channels.no_channels")}
              </div>
            ) : (
              <div className="space-y-4">
                {channels.filter(c => c.channel_type === "feishu").map((channel) => (
                  <div key={channel.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center gap-4">
                      <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                        <MessageSquare className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <div className="font-medium">{channel.channel_name}</div>
                        <div className="text-xs text-muted-foreground capitalize">
                          {channel.channel_type} • {channel.is_active ? t("channels.status.active") : t("channels.status.inactive")}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <Switch
                        checked={channel.is_active}
                        onCheckedChange={() => toggleActive(channel)}
                      />
                      <Button variant="ghost" size="icon" onClick={() => handleOpenDialog(channel)} title={t("channels.actions.edit")}>
                        <Edit className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon" onClick={() => handleDelete(channel.id)} title={t("channels.actions.delete")}>
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Slack Bots Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5" />
                {t("channels.slack_bots")}
              </CardTitle>
              <CardDescription>
                {t("channels.description", { platform: t("channels.slack_bots") })}
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                onClick={() => handleOpenDialog(undefined, "slack")}
                size="sm"
              >
                <Settings2 className="h-4 w-4 mr-2" />
                {t("channels.slack_manual_setup")}
              </Button>
              <Button
                onClick={handleSlackOAuthConnect}
                disabled={isConnectingSlack}
                size="sm"
                className="bg-[#2EB67D] text-white hover:bg-[#259c6b]"
              >
                <SlackLogo className="h-4 w-4 mr-2" />
                {isConnectingSlack
                  ? t("channels.slack_connecting")
                  : t("channels.connect_slack")}
                {!isConnectingSlack && <ExternalLink className="h-3.5 w-3.5 ml-2" />}
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-sm text-muted-foreground">{t("common.loading")}</div>
            ) : channels.filter(c => c.channel_type === "slack").length === 0 ? (
              <div className="text-sm text-muted-foreground py-4 text-center border rounded-md bg-muted/20">
                {t("channels.no_channels")}
              </div>
            ) : (
              <div className="space-y-4">
                {channels.filter(c => c.channel_type === "slack").map((channel) => (
                  <div key={channel.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center gap-4">
                      <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                        <MessageSquare className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <div className="font-medium">{channel.channel_name}</div>
                        <div className="text-xs text-muted-foreground capitalize">
                          {channel.channel_type} • {channel.is_active ? t("channels.status.active") : t("channels.status.inactive")}
                        </div>
                        {channel.config.installation_mode === "oauth" && channel.config.workspace_name && (
                          <div className="text-xs text-muted-foreground mt-0.5">
                            {t("channels.slack_workspace")}: {channel.config.workspace_name}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <Switch
                        checked={channel.is_active}
                        onCheckedChange={() => toggleActive(channel)}
                      />
                      <Button variant="ghost" size="icon" onClick={() => handleOpenDialog(channel)} title={t("channels.actions.edit")}>
                        <Edit className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon" onClick={() => handleDelete(channel.id)} title={t("channels.actions.delete")}>
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{editingChannel ? t("channels.dialog.edit_title") : t("channels.dialog.add_title")}</DialogTitle>
            <DialogDescription>
              {t("channels.dialog.description")}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>{t("channels.dialog.platform")}</Label>
              <Select
                value={formData.channel_type}
                onValueChange={(val) => setFormData(prev => ({ ...prev, channel_type: val }))}
                options={[
                  { value: "telegram", label: t("channels.dialog.telegram_bot") },
                  { value: "feishu", label: t("channels.dialog.feishu_bot") },
                  { value: "slack", label: t("channels.dialog.slack_bot") },
                ]}
                disabled={!!editingChannel}
              />
            </div>

            {!!editingChannel && (
              <div className="space-y-2">
                <Label>{t("channels.dialog.name")}</Label>
                <Input
                  type="text"
                  placeholder={t("channels.dialog.name_placeholder")}
                  value={formData.channel_name}
                  onChange={(e) => setFormData(prev => ({ ...prev, channel_name: e.target.value }))}
                />
              </div>
            )}

            {(
              formData.channel_type === "telegram"
              || (
                formData.channel_type === "slack"
                && editingChannel?.config.installation_mode !== "oauth"
              )
            ) && (
              <div className="space-y-2">
                <Label>{t("channels.dialog.bot_token")}</Label>
                <Input
                  type="password"
                  placeholder={formData.channel_type === "slack" ? "xoxb-..." : "123456789:ABCdefGHIjklmNOPqrsTUVwxyz"}
                  value={formData.bot_token}
                  onChange={(e) => setFormData(prev => ({ ...prev, bot_token: e.target.value }))}
                />
              </div>
            )}

            {(
              formData.channel_type === "slack"
              && editingChannel?.config.installation_mode !== "oauth"
            ) && (
              <div className="space-y-2">
                <Label>{t("channels.dialog.slack_app_token")}</Label>
                <Input
                  type="password"
                  placeholder="xapp-..."
                  value={formData.app_token}
                  onChange={(e) => setFormData(prev => ({ ...prev, app_token: e.target.value }))}
                />
                <p className="text-xs text-muted-foreground">
                  {t("channels.dialog.slack_socket_mode_help")}
                </p>
                <p className="text-xs text-muted-foreground">
                  {t("channels.dialog.slack_permissions_help")}
                </p>
              </div>
            )}

            {(
              formData.channel_type === "slack"
              && editingChannel?.config.installation_mode === "oauth"
            ) && (
              <div className="rounded-lg border bg-muted/30 p-4">
                <div className="flex items-center gap-2 font-medium">
                  <SlackLogo className="h-5 w-5" />
                  {t("channels.slack_connected_via_oauth")}
                </div>
                <p className="text-sm text-muted-foreground mt-1">
                  {editingChannel.config.workspace_name || editingChannel.channel_name}
                </p>
              </div>
            )}

            {formData.channel_type === "feishu" && (
              <>
                <div className="space-y-2">
                  <Label>{t("channels.dialog.app_id")}</Label>
                  <Input
                    type="text"
                    placeholder="cli_a1b2c3d4e5f6g7h8"
                    value={formData.app_id}
                    onChange={(e) => setFormData(prev => ({ ...prev, app_id: e.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label>{t("channels.dialog.app_secret")}</Label>
                  <Input
                    type="password"
                    placeholder="a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6"
                    value={formData.app_secret}
                    onChange={(e) => setFormData(prev => ({ ...prev, app_secret: e.target.value }))}
                  />
                </div>
              </>
            )}

            <div className="space-y-2">
              <Label>{t("channels.dialog.allowed_users")}</Label>
              <Input
                placeholder={t("channels.dialog.allowed_users_placeholder")}
                value={formData.allowed_users}
                onChange={(e) => setFormData(prev => ({ ...prev, allowed_users: e.target.value }))}
              />
            </div>
            <div className="flex items-center justify-between">
              <Label>{t("channels.dialog.active")}</Label>
              <Switch
                checked={formData.is_active}
                onCheckedChange={(checked) => setFormData(prev => ({ ...prev, is_active: checked }))}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsDialogOpen(false)}>{t("channels.dialog.cancel")}</Button>
            <Button onClick={handleSubmit}>{t("channels.dialog.save")}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
