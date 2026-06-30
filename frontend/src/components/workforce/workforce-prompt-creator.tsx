"use client"

import React, { useState } from "react"
import { Loader2, Sparkles } from "lucide-react"
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

  const examples = [
    t("workforces.create.prompt.example1"),
    t("workforces.create.prompt.example2"),
    t("workforces.create.prompt.example3"),
  ]

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
    <div className="flex flex-col gap-6">
      <Textarea
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        placeholder={t("workforces.create.prompt.placeholder")}
        rows={7}
        className="resize-y"
      />

      {examples.length > 0 && (
        <div className="space-y-2.5">
          <div className="text-xs font-semibold uppercase tracking-wider text-indigo-500">
            {t("workforces.create.prompt.tryAnExample")}
          </div>
          {examples.map((example) => (
            <button
              key={example}
              onClick={() => setPrompt(example)}
              className="w-full rounded-lg border border-transparent bg-muted px-4 py-3 text-left text-sm text-foreground transition-colors hover:bg-muted/70 hover:border-border"
            >
              {example}
            </button>
          ))}
        </div>
      )}

      <div className="flex items-center justify-end gap-3">
        {onCancel ? (
          <Button variant="outline" onClick={onCancel} disabled={submitting}>
            {t("common.cancel")}
          </Button>
        ) : null}
        <Button
          onClick={handleCreate}
          disabled={submitting || !prompt.trim()}
          className="bg-violet-600 hover:bg-violet-700 text-white disabled:bg-violet-300"
        >
          {submitting ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              {t("workforces.loading.creating")}
            </>
          ) : (
            <>
              <Sparkles className="mr-2 h-4 w-4" />
              {t("workforces.create.prompt.buildWorkforce")}
            </>
          )}
        </Button>
      </div>
    </div>
  )
}
