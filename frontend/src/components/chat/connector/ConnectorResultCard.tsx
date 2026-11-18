import React from "react"
import { useI18n } from "@/contexts/i18n-context"
import { ConnectorLogo, CARD, CARD_W } from "./ConnectorLogo"
import type { ResultCardData } from "./types"

const UNITS = [
  ["year", 60 * 60 * 24 * 365],
  ["month", 60 * 60 * 24 * 30],
  ["week", 60 * 60 * 24 * 7],
  ["day", 60 * 60 * 24],
  ["hour", 60 * 60],
  ["minute", 60],
] as const

/** "3 hours ago" in the locale; empty when the time is missing or invalid. */
function formatWhen(createdAt: string | null | undefined, locale: string | undefined): string {
  const time = createdAt ? new Date(createdAt).getTime() : NaN
  if (!Number.isFinite(time)) return ""
  const delta = Math.round((time - Date.now()) / 1000)
  const formatter = new Intl.RelativeTimeFormat(locale, { numeric: "auto" })
  for (const [unit, seconds] of UNITS) {
    if (Math.abs(delta) >= seconds) return formatter.format(Math.round(delta / seconds), unit)
  }
  return formatter.format(delta, "second")
}

/** Source-labelled result card. The footnote is the last list item. */
export function ConnectorResultCard({ card }: { card: ResultCardData }) {
  const { t, locale } = useI18n()
  // Each segment is left out when its value is missing.
  const byline = [
    card.source.name ? t("connector.result.from", { name: card.source.name }) : "",
    formatWhen(card.created_at, locale),
  ].filter(Boolean).join(" · ")
  const items = card.footnote ? [...card.items, card.footnote] : card.items
  return (
    <div className={`${CARD} ${CARD_W}`} data-testid="connector-result-card">
      <div className="flex items-center gap-[11px] px-[15px] py-[13px]">
        <ConnectorLogo logo={card.source.logo} />
        <div className="min-w-0 flex-1">
          <b className="flex items-center gap-[7px] text-[13.5px] font-[620] tracking-[-0.01em]">{card.title}</b>
          {byline && <span className="mt-[2px] block text-[12px] leading-[1.4] text-[#6B6B76]">{byline}</span>}
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
