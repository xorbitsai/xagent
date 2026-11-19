import React from "react"
import type { ConnectorLogoData } from "./types"

/** 36px brand tile. */
export function ConnectorLogo({ logo }: { logo: ConnectorLogoData }) {
  return (
    <span
      aria-hidden="true"
      className="grid h-9 w-9 shrink-0 basis-9 place-items-center overflow-hidden rounded-[10px] text-[15px] font-[800] tracking-[-0.03em] shadow-[inset_0_0_0_1px_rgba(0,0,0,.07)]"
      style={{ background: logo.bg, color: logo.fg }}
    >
      {logo.icon_url ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={logo.icon_url} alt="" className="h-full w-full object-cover" />
      ) : (
        logo.mark
      )}
    </span>
  )
}

export const CARD_W = "w-[min(520px,100%)]"
export const CARD_W_MINI = "w-[min(340px,100%)]"

/** Shared card shell classes. */
export const CARD =
  "mt-[6px] mb-[10px] overflow-hidden rounded-[16px] border border-[#E7E7EC] bg-white text-left text-[14px] leading-[1.5] text-[#1A1A1F] shadow-[0_1px_2px_rgba(20,22,40,.045)]"
