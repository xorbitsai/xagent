import React from "react"
import { useI18n } from "@/contexts/i18n-context"
import { ConnectorLogo, CARD, CARD_W } from "./ConnectorLogo"
import type { ResultCardData } from "./types"

function useWhen(createdAt?: string | null): string {
  const { t } = useI18n()
  const ms = createdAt ? Date.now() - new Date(createdAt).getTime() : 0
  if (!createdAt || !Number.isFinite(ms) || ms < 60_000) return t("connector.result.justNow")
  const minutes = Math.floor(ms / 60_000)
  if (minutes < 60) return t("connector.result.minutesAgo", { count: minutes })
  if (minutes < 1440) return t("connector.result.hoursAgo", { count: Math.floor(minutes / 60) })
  return t("connector.result.daysAgo", { count: Math.floor(minutes / 1440) })
}

/** Source-labelled result card The footnote is the last list item. */
export function ConnectorResultCard({ card }: { card: ResultCardData }) {
  const { t } = useI18n()
  const when = useWhen(card.created_at)
  const items = card.footnote ? [...card.items, card.footnote] : card.items
  return (
    <div className={`${CARD} ${CARD_W}`} data-testid="connector-result-card">
      <div className="flex items-center gap-[11px] px-[15px] py-[13px]">
        <ConnectorLogo logo={card.source.logo} />
        <div className="min-w-0 flex-1">
          <b className="flex items-center gap-[7px] text-[13.5px] font-[620] tracking-[-0.01em]">{card.title}</b>
          <span className="mt-[2px] block text-[12px] leading-[1.4] text-[#6B6B76]">
            {t("connector.result.from", { name: card.source.name, when })}
          </span>
        </div>
      </div>
      {items.length > 0 && (
        <div className="border-t border-[#F0F0F4] px-[15px] pb-[14px] pt-[12px]">
          <ul className="m-0 mt-[6px] list-disc pl-[18px] text-[13px] leading-[1.55] text-[#3D3D46] [&>li+li]:mt-[3px]">
            {items.map((item, i) => <li key={i}>{item}</li>)}
          </ul>
        </div>
      )}
    </div>
  )
}
