"use client"

import React, { useEffect, useState } from "react"
import { Monitor, Settings2, Volume2 } from "lucide-react"

import { SettingsContent } from "@/components/settings/settings-content"
import { VoiceSettings } from "@/components/settings/voice-settings"
import { useAuth } from "@/contexts/auth-context"
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog"
import { useI18n } from "@/contexts/i18n-context"
import { cn } from "@/lib/utils"
import { TaskRuntimeSettingsExtension } from "@/lib/task-runtime-ui-extension"

export type SettingsDialogTab = "general" | "computer" | "voice"

export function SettingsDialog({
  open,
  onOpenChange,
  initialTab = "general",
  targetId,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  initialTab?: SettingsDialogTab
  targetId?: string
}) {
  const { t } = useI18n()
  const { session } = useAuth()
  const [activeTab, setActiveTab] = useState<SettingsDialogTab>(initialTab)
  const tabs = [
    { id: "general" as const, label: t("settings.tabs.general"), icon: Settings2 },
    { id: "voice" as const, label: t("settings.tabs.voice"), icon: Volume2 },
    { id: "computer" as const, label: t("settings.tabs.computerAccess"), icon: Monitor },
  ]

  useEffect(() => {
    if (open) setActiveTab(initialTab)
  }, [initialTab, open])

  useEffect(() => {
    if (!open || activeTab !== "computer" || !targetId) return
    const frame = window.requestAnimationFrame(() => {
      document.getElementById(targetId)?.scrollIntoView?.({ behavior: "smooth", block: "start" })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [activeTab, open, targetId])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="h-[min(720px,calc(100vh-48px))] w-[min(1120px,calc(100vw-48px))] max-w-none gap-0 overflow-hidden rounded-[20px] p-0 sm:max-w-[1120px]">
        <DialogTitle className="sr-only">{t("settings.title")}</DialogTitle>
        <DialogDescription className="sr-only">{t("settings.description")}</DialogDescription>
        <div className="grid h-full min-h-0 grid-cols-[210px_minmax(0,1fr)]">
          <aside className="border-r border-border bg-muted/30 px-3 py-5">
            <div className="px-2 pb-3 text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
              {t("settings.title")}
            </div>
            <nav aria-label={t("settings.title")} className="space-y-1">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  aria-current={activeTab === tab.id ? "page" : undefined}
                  className={cn(
                    "flex w-full items-center gap-2.5 rounded-xl px-3 py-2.5 text-left text-sm transition-colors",
                    activeTab === tab.id
                      ? "bg-background font-semibold text-[#2536e0] shadow-sm"
                      : "text-muted-foreground hover:bg-background/70 hover:text-foreground",
                  )}
                >
                  <tab.icon className="h-4 w-4 shrink-0" />
                  <span className="truncate">{tab.label}</span>
                </button>
              ))}
            </nav>
          </aside>

          <section className="flex min-h-0 min-w-0 flex-col bg-background">
            <header className="shrink-0 border-b border-border px-6 py-5">
              <h2 className="text-base font-semibold">
                {tabs.find((tab) => tab.id === activeTab)?.label}
              </h2>
            </header>
            <div className="min-h-0 flex-1 overflow-hidden">
              {activeTab === "voice" ? (
                <div className="h-full overflow-y-auto p-6"><VoiceSettings key={session?.sessionId} /></div>
              ) : activeTab === "computer" ? (
                <div className="h-full overflow-y-auto p-6">
                  <TaskRuntimeSettingsExtension />
                </div>
              ) : (
                <div className="h-full [&_[data-settings-page-header]]:hidden [&_[data-settings-runtime]]:hidden [&_[data-slot=tabs]]:!mt-0 [&_[role=tablist]]:hidden [&_[role=tabpanel]]:!mt-0 [&>div>div:first-child]:hidden [&>div]:!p-6">
                  <SettingsContent />
                </div>
              )}
            </div>
          </section>
        </div>
      </DialogContent>
    </Dialog>
  )
}
