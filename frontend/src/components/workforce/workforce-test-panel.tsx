"use client"

import React, { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { useI18n } from "@/contexts/i18n-context"
import { runWorkforce } from "@/lib/workforces-api"
import type { WorkforceRunResponse } from "@/types/workforce"
import { toast } from "sonner"

interface WorkforceTestPanelProps {
  workforceId: number
  disabled?: boolean
  disabledReason?: string
  onRunCreated: (result: WorkforceRunResponse) => void
}

export function WorkforceTestPanel({
  workforceId,
  disabled = false,
  disabledReason,
  onRunCreated,
}: WorkforceTestPanelProps) {
  const { t } = useI18n()
  const [message, setMessage] = useState("")
  const [loading, setLoading] = useState(false)

  const handleRun = async () => {
    const value = message.trim()
    if (!value || loading || disabled) return
    setLoading(true)
    try {
      const result = await runWorkforce(workforceId, { message: value })
      onRunCreated(result)
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.run")
      toast.error(nextError)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("workforces.run.testTitle")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Textarea
          placeholder={t("workforces.run.placeholder")}
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          rows={8}
          disabled={disabled}
        />
        {disabled && disabledReason ? (
          <div className="text-sm text-muted-foreground">{disabledReason}</div>
        ) : null}
        <Button
          onClick={handleRun}
          disabled={loading || disabled || !message.trim()}
          className="w-full"
        >
          {loading ? t("workforces.loading.starting") : t("workforces.actions.runWorkforce")}
        </Button>
      </CardContent>
    </Card>
  )
}
