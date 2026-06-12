"use client"

import React, { useState } from "react"
import { Button } from "@/components/ui/button"
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
    <div className="flex flex-col h-full">
      <div className="p-4 border-b bg-background">
        <h3 className="font-semibold leading-none tracking-tight">{t("workforces.run.testTitle")}</h3>
      </div>
      <div className="flex flex-1 flex-col gap-3 min-h-0 p-4">
        <Textarea
          placeholder={t("workforces.run.placeholder")}
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          rows={6}
          disabled={disabled}
          className="flex-1 min-h-[120px]"
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
      </div>
    </div>
  )
}
