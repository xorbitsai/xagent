import type { Translate } from "@/contexts/i18n-context"

export function getRunDisabledReason(status: string | null | undefined, t: Translate) {
  if (status === "archived") return t("workforces.run.archivedDisabled")
  return null
}

export function getBuilderReadOnlyReason(status: string | null | undefined, t: Translate) {
  if (status === "archived") return t("workforces.builder.archivedReadOnly")
  return null
}
