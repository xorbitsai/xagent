import React from "react"
import { useI18n } from "@/contexts/i18n-context"

/**
 * Stands in for a `connect_connector` entry the card cannot show (no usable
 * interaction id or fields). It is read-only on purpose: no input of any kind,
 * so nothing a user could type here (a key, say) ends up in the chat.
 */
export function ConnectorUnusableNotice() {
  const { t } = useI18n()
  return (
    <p role="status" className="m-0 rounded-[12px] border border-[#E7E7EC] bg-[#FAFAFA] px-[12px] py-[10px] text-[12.5px] leading-[1.5] text-[#3D3D46]">
      {t("connector.connect.cannotConnect")}
    </p>
  )
}
