"use client"

import React, { useState } from "react"
import { Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
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
    <Card>
      <CardHeader>
        <CardTitle>{t("workforces.create.prompt.cardTitle")}</CardTitle>
        <CardDescription>{t("workforces.create.prompt.cardDescription")}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Textarea
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder={t("workforces.create.prompt.placeholder")}
          rows={10}
        />
        <div className="flex justify-end gap-3">
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
      </CardContent>
    </Card>
  )
}
