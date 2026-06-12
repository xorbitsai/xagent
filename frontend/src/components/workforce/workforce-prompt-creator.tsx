"use client"

import React, { useState } from "react"
import { Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { useI18n } from "@/contexts/i18n-context"
import { createWorkforceFromPrompt } from "@/lib/workforces-api"
import type { WorkforceDetail } from "@/types/workforce"
import { toast } from "sonner"

interface WorkforcePromptCreatorProps {
  onCreated: (workforce: WorkforceDetail) => void
  onCancel?: () => void
}

export function WorkforcePromptCreator({
  onCreated,
  onCancel,
}: WorkforcePromptCreatorProps) {
  const { t } = useI18n()
  const [prompt, setPrompt] = useState("")
  const [submitting, setSubmitting] = useState(false)

  const handleCreate = async () => {
    const value = prompt.trim()
    if (!value || submitting) return
    try {
      setSubmitting(true)
      const workforce = await createWorkforceFromPrompt({ prompt: value })
      onCreated(workforce)
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.create")
      toast.error(nextError)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="space-y-1.5">
        <h3 className="font-semibold leading-none tracking-tight">{t("workforces.create.prompt.cardTitle")}</h3>
        <p className="text-sm text-muted-foreground">{t("workforces.create.prompt.cardDescription")}</p>
      </div>
      <Textarea
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        placeholder={t("workforces.create.prompt.placeholder")}
        rows={10}
      />
      <div className="flex justify-end gap-3 mt-4">
        {onCancel ? (
          <Button variant="outline" onClick={onCancel} disabled={submitting}>
            {t("common.cancel")}
          </Button>
        ) : null}
        <Button onClick={handleCreate} disabled={submitting || !prompt.trim()}>
          {submitting ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              {t("workforces.loading.creating")}
            </>
          ) : (
            t("workforces.create.prompt.generate")
          )}
        </Button>
      </div>
    </div>
  )
}
