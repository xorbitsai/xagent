import React, { useEffect, useState } from "react"
import { useI18n } from "@/contexts/i18n-context"
import { ConnectorLogo, CARD, CARD_W_MINI } from "./ConnectorLogo"
import type { ConnectorInfo } from "./types"

export const PROBE_STEP_MS = 750

const CHECK_KEYS = {
  native: "connector.probe.native",
  mcp: "connector.probe.mcp",
  api: "connector.probe.api",
} as const
const FOUND_KEYS = {
  native: "connector.probe.foundNative",
  mcp: "connector.probe.foundMcp",
  api: "connector.probe.foundApi",
} as const

/**
 * Probe mini-card (mock `.xc-mini`). Steps through `probe.checks` every 750 ms
 * and stops on the found route. With `animate={false}` (a reload, or a card the
 * user already acted on) it renders the final state at once.
 */
export function ConnectorProbeCard({ connector, animate }: { connector: ConnectorInfo; animate: boolean }) {
  const { t } = useI18n()
  const checks = (connector.probe?.checks ?? []).filter((c): c is keyof typeof CHECK_KEYS => Object.hasOwn(CHECK_KEYS, c))
  const found = connector.probe?.found ?? connector.route
  const [step, setStep] = useState(0)
  // A replay (a reload) shows the final state at once.
  const done = !animate || step >= checks.length

  useEffect(() => {
    if (done) return
    const timer = setTimeout(() => setStep((s) => s + 1), PROBE_STEP_MS)
    return () => clearTimeout(timer)
  }, [step, done])

  const line = done
    ? t(FOUND_KEYS[found])
    : t(CHECK_KEYS[checks[step]])

  return (
    <div className={`${CARD} ${CARD_W_MINI}`} data-testid="connector-probe-card" data-s={done ? "ok" : ""}>
      <div className="flex items-center gap-[11px] px-[14px] py-[11px]">
        <ConnectorLogo logo={connector.logo} />
        <div className="min-w-0 flex-1">
          <b className="flex items-center gap-[7px] text-[14px] font-[620] tracking-[-0.01em]">{connector.name}</b>
          <span
            role="status"
            className={`mt-[2px] block text-[12px] leading-[1.4] transition-colors duration-200 ${done ? "font-[550] text-[#15804A]" : "text-[#6B6B76]"}`}
          >
            {line}
          </span>
        </div>
        {done ? (
          <span aria-hidden="true" className="grid h-5 w-5 shrink-0 basis-5 place-items-center rounded-full bg-[#15804A] text-[11px] font-[800] text-white">✓</span>
        ) : (
          <span aria-hidden="true" className="h-5 w-5 shrink-0 basis-5 animate-[spin_1.4s_linear_infinite] rounded-full border-2 border-dashed border-[#2536E0]" />
        )}
      </div>
    </div>
  )
}
