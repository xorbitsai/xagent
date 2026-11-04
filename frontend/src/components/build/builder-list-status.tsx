"use client"

import React from "react"
import { AlertCircle, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useI18n } from "@/contexts/i18n-context"
import type { CatalogStatus } from "./use-builder-catalog"

interface BuilderListStatusProps {
  status: CatalogStatus
  onRetry: () => void
  // Rendered once the list has loaded: the picker, or its empty state.
  children: React.ReactNode
}

// Keeps a picker's loading and failed states apart from a list that really is
// empty, so a slow or failing request never reads as "no data".
export function BuilderListStatus({ status, onRetry, children }: BuilderListStatusProps) {
  const { t } = useI18n()

  if (status === "loading") {
    return (
      <div role="status" className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        {t("common.loading")}
      </div>
    )
  }

  if (status === "error") {
    return (
      <div role="alert" className="flex items-center gap-2 text-sm text-destructive">
        <AlertCircle className="h-3.5 w-3.5 shrink-0" />
        <span>{t("builds.configForm.listLoadFailed")}</span>
        <Button type="button" variant="outline" size="sm" className="h-6 px-2 text-xs" onClick={onRetry}>
          {t("common.retry")}
        </Button>
      </div>
    )
  }

  return <>{children}</>
}
