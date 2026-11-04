"use client"

import React from "react"
import { Badge } from "@/components/ui/badge"
import { useI18n } from "@/contexts/i18n-context"
import type { WorkforceStatus } from "@/types/workforce"

interface WorkforceStatusBadgeProps {
  status: WorkforceStatus
}

function statusVariant(status: string) {
  if (status === "active") return "default"
  if (status === "archived") return "secondary"
  return "outline"
}

export function WorkforceStatusBadge({ status }: WorkforceStatusBadgeProps) {
  const { t } = useI18n()
  return (
    <Badge variant={statusVariant(status)}>
      {t(`workforces.status.${status}`)}
    </Badge>
  )
}
