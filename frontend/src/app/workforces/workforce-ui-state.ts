import type { Translate } from "@/contexts/i18n-context"

export function getRunDisabledReason(status: string | null | undefined, t: Translate) {
  if (status === "active") return null
  if (status === "archived") return t("workforces.run.archivedDisabled")
  return t("workforces.run.inactiveDisabled")
}

// Same status gate as getRunDisabledReason (PR review round 7, finding #3),
// but worded for the Deploy action specifically -- reusing Run's copy
// verbatim reads as confusing on a button that mints API keys / configures
// webhooks rather than running the workforce.
export function getDeployDisabledReason(status: string | null | undefined, t: Translate) {
  if (status === "active") return null
  if (status === "archived") return t("workforces.deploy.archivedDisabled")
  return t("workforces.deploy.inactiveDisabled")
}

export function getBuilderReadOnlyReason(status: string | null | undefined, t: Translate) {
  if (status === "archived") return t("workforces.builder.archivedReadOnly")
  return null
}
