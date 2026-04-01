"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Select } from "@/components/ui/select"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { MessageSquare, Plus, Trash2, Edit } from "lucide-react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "sonner"

interface Channel {
  id: number;
  channel_type: string;
  channel_name: string;
  config: Record<string, any>;
  is_active: boolean;
}

export default function ChannelsPage() {
  const { t } = useI18n()
  const [channels, setChannels] = useState<Channel[]>([])
  const [loading, setLoading] = useState(true)
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [editingChannel, setEditingChannel] = useState<Channel | null>(null)

  const [formData, setFormData] = useState({
    channel_type: "telegram",
    bot_token: "",
    app_id: "",
    app_secret: "",
    allowed_users: "",
    is_active: true
  })

  useEffect(() => {
    fetchChannels()
  }, [])

  const fetchChannels = async () => {
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
  }

  const handleOpenDialog = (channel?: Channel, defaultType: string = "telegram") => {
    if (channel) {
      setEditingChannel(channel)
      setFormData({
        channel_type: channel.channel_type,
        bot_token: channel.config.bot_token || "",
        app_id: channel.config.app_id || "",
        app_secret: channel.config.app_secret || "",
        allowed_users: channel.config.allowed_users ? channel.config.allowed_users.join(", ") : "",
        is_active: channel.is_active
      })
    } else {
      setEditingChannel(null)
      setFormData({
        channel_type: defaultType,
        bot_token: "",
        app_id: "",
        app_secret: "",
        allowed_users: "",
        is_active: true
      })
    }
    setIsDialogOpen(true)
  }

  const handleSubmit = async () => {
    try {
      if (formData.channel_type === "telegram" && !formData.bot_token) {
        toast.error(t("channels.messages.fill_required"))
        return
      }

      if (formData.channel_type === "feishu" && (!formData.app_id || !formData.app_secret)) {
        toast.error(t("channels.messages.fill_required"))
        return
      }

      const payload = {
        channel_type: formData.channel_type,
        channel_name: "", // Always send empty to force auto-fetch on backend
        config: {
          bot_token: formData.channel_type === "telegram" ? formData.bot_token : undefined,
          app_id: formData.channel_type === "feishu" ? formData.app_id : undefined,
          app_secret: formData.channel_type === "feishu" ? formData.app_secret : undefined,
          allowed_users: formData.allowed_users.trim() ? formData.allowed_users.split(",").map(u => u.trim()).filter(Boolean) : null
        },
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
    } catch (error: any) {
      console.error("Failed to save channel:", error)
      toast.error(error.message || t("channels.messages.save_failed"))
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
    <div className="w-full p-8 space-y-6">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-1">{t("channels.page_title")}</h1>
          <p className="text-muted-foreground">{t("channels.page_description")}</p>
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
                ]}
                disabled={!!editingChannel}
              />
            </div>

            {formData.channel_type === "telegram" && (
              <div className="space-y-2">
                <Label>{t("channels.dialog.bot_token")}</Label>
                <Input
                  type="password"
                  placeholder="123456789:ABCdefGHIjklmNOPqrsTUVwxyz"
                  value={formData.bot_token}
                  onChange={(e) => setFormData(prev => ({ ...prev, bot_token: e.target.value }))}
                />
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
