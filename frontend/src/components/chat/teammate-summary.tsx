"use client"

import React from "react"
import { CheckCircle2, Zap } from "lucide-react"
import type { Interaction } from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"

/** A persisted, read-only welcome card. It neither starts execution nor
 * claims that third-party permissions/approval gates have been configured. */
export function TeammateSummary({ interaction }: { interaction: Interaction }) {
  const { t } = useI18n()
  const rows = [
    [t("onboarding.firstChat.workingFrom"), t("onboarding.firstChat.inputs")],
    [t("onboarding.firstChat.youGet"), interaction.description || interaction.label],
    [t("onboarding.firstChat.dataAccess"), interaction.apps?.length
      ? t("onboarding.firstChat.accessApps", { apps: interaction.apps.join(", ") })
      : t("onboarding.firstChat.accessFiles")],
    [t("onboarding.firstChat.runsAs"), t("onboarding.firstChat.composerModel")],
  ]
  return (
    <section aria-label={t("onboarding.firstChat.summary")} className="max-w-2xl overflow-hidden rounded-2xl border border-border bg-background">
      <div className="flex items-center gap-2 border-b border-border bg-muted/30 px-4 py-3">
        <Zap className="h-4 w-4 text-primary" aria-hidden="true" />
        <h3 className="text-sm font-semibold">{interaction.label}</h3>
      </div>
      <dl className="space-y-3 p-4 text-sm">
        {rows.map(([label, value]) => (
          <div key={label} className="grid gap-1 sm:grid-cols-[110px_1fr] sm:gap-4">
            <dt className="text-muted-foreground">{label}</dt>
            <dd className="min-w-0 break-words">{value}</dd>
          </div>
        ))}
      </dl>
      <p className="flex items-start gap-2 border-t border-border bg-muted/20 px-4 py-3 text-xs leading-relaxed text-muted-foreground">
        <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        {t("onboarding.team.reviewNote")}
      </p>
      <p className="border-t border-border px-4 py-3 text-sm">{interaction.placeholder || t("onboarding.firstChat.prompt")}</p>
    </section>
  )
}
