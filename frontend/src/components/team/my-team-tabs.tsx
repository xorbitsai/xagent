"use client"

import React from "react"
import { useRouter } from "next/navigation"

import { SegmentedTabs } from "@/components/ui/segmented-tabs"
import { useI18n } from "@/contexts/i18n-context"

type TeamView = "teammates" | "workforces"

interface MyTeamTabsProps {
  active: TeamView
  teammateCount?: number
  workforceCount?: number
}

function CountBadge({ value }: { value?: number }) {
  if (value === undefined) return null
  return (
    <span className="ml-1 rounded-full bg-[#e5e7ef] px-1.5 py-0.5 text-[10px] font-bold text-[#666d80]">
      {value}
    </span>
  )
}

/** Shared top-level switch that makes agents and workforces one My Team view. */
export function MyTeamTabs({ active, teammateCount, workforceCount }: MyTeamTabsProps) {
  const router = useRouter()
  const { t } = useI18n()

  return (
    <SegmentedTabs
      value={active}
      onValueChange={(value) => router.push(value === "teammates" ? "/workforces?view=teammates" : "/workforces")}
      items={[
        {
          id: "teammates",
          label: <>{t("teamPage.tabs.teammates")} <CountBadge value={teammateCount} /></>,
        },
        {
          id: "workforces",
          label: <>{t("teamPage.tabs.workforces")} <CountBadge value={workforceCount} /></>,
        },
      ]}
      className="justify-center"
      listClassName="rounded-full bg-[#f1f1f4] p-1"
      triggerClassName="inline-flex items-center rounded-full px-5 py-2 text-[13px]"
      activeTriggerClassName="bg-white text-[#2536e0] shadow-sm ring-1 ring-[#e1e2e8]"
    />
  )
}
