"use client"

import React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/select"
import { useI18n } from "@/contexts/i18n-context"
import type { WorkforceAgentOption } from "@/types/workforce"

interface ManagerStepProps {
  managerAgentId: string
  onManagerAgentIdChange: (value: string) => void
  agents: WorkforceAgentOption[]
}

export function ManagerStep({
  managerAgentId,
  onManagerAgentIdChange,
  agents,
}: ManagerStepProps) {
  const { t } = useI18n()

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("workforces.fields.manager")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Label>{t("workforces.create.manager.selectLabel")} <span className="text-destructive">*</span></Label>
        <Select
          value={managerAgentId}
          onValueChange={onManagerAgentIdChange}
          placeholder={t("workforces.create.manager.placeholder")}
          options={agents.map((agent) => ({
            value: String(agent.id),
            label: agent.name,
            description: agent.description || undefined,
          }))}
        />
      </CardContent>
    </Card>
  )
}
