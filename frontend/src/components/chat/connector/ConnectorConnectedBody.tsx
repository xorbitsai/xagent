import React from "react"
import { useI18n } from "@/contexts/i18n-context"
import type { ConnectorToolSummary } from "@/lib/page-extension-contracts"

const SVG = "shrink-0 fill-none stroke-[#15804A] stroke-2 [stroke-linecap:round] [stroke-linejoin:round]"
const VAULT_ROW = "mb-[10px] flex items-center gap-2 text-[12.5px] text-[#3D3D46]"

export function LockIcon({ size, className }: { size: number; className: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true">
      <rect x="5" y="10" width="14" height="10" rx="2" />
      <path d="M8 10V7a4 4 0 0 1 8 0v3" />
    </svg>
  )
}

/** Connected state of the connect card. */
export function ConnectorConnectedBody({
  secretMasked, agentName, scope, tools,
}: {
  secretMasked: string | null
  agentName: string | null
  scope: string | null
  tools: ConnectorToolSummary[]
}) {
  const { t } = useI18n()
  const scopeLabel = t(scope === "remote" ? "connector.connect.scopeRemote" : "connector.connect.scopeLocal")
  return (
    <>
      {secretMasked && (
        <div className={VAULT_ROW}>
          <LockIcon size={15} className={`${SVG} basis-[15px]`} />
          <span>{t("connector.connect.secretStored", { masked: secretMasked })}</span>
        </div>
      )}
      <div className={VAULT_ROW}>
        <svg width="15" height="15" viewBox="0 0 24 24" className={`${SVG} basis-[15px]`} aria-hidden="true"><path d="m5 12.5 4.5 4.5L19 7.5" /></svg>
        <span>
          {agentName
            ? t("connector.connect.addedTo", { agent: agentName, scope: scopeLabel })
            : t("connector.connect.addedToDefault", { scope: scopeLabel })}
        </span>
      </div>
      {tools.length > 0 && (
        <>
          <b className="mb-[6px] block text-[12px] font-[650] text-[#3D3D46]">{t("connector.connect.canDo")}</b>
          <div className="flex flex-wrap gap-[6px]">
            {tools.map((tool) => (
              <span key={tool.name} className="rounded-full border border-[#E7E7EC] bg-white px-[10px] py-[4px] text-[12px]">{tool.label}</span>
            ))}
          </div>
        </>
      )}
    </>
  )
}
