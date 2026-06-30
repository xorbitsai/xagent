"use client"

import React, { useState } from "react"
import { ArrowLeft, Sparkles, SlidersHorizontal } from "lucide-react"
import { Button } from "@/components/ui/button"
import { WorkforcePromptCreator } from "./workforce-prompt-creator"
import { WorkforceWizard } from "./workforce-wizard"
import { useI18n } from "@/contexts/i18n-context"
import type { WorkforceDetail } from "@/types/workforce"

type CreateMode = "select" | "ai" | "manual"

interface WorkforceCreateViewProps {
  onBack: () => void
  onCreated: (workforce: WorkforceDetail) => void
}

export function WorkforceCreateView({ onBack, onCreated }: WorkforceCreateViewProps) {
  const { t } = useI18n()
  const [mode, setMode] = useState<CreateMode>("select")

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 sm:px-8">
      {/* Persistent header — always visible in all sub-views */}
      <div className="mb-8 flex items-center gap-3">
        <Button
          variant="outline"
          size="icon"
          className="h-9 w-9 shrink-0 rounded-lg"
          onClick={onBack}
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <div>
          <h1 className="text-xl font-semibold">{t("workforces.create.title")}</h1>
          <p className="text-sm text-muted-foreground">{t("workforces.create.subtitle")}</p>
        </div>
      </div>

      {/* Mode: select */}
      {mode === "select" && (
        <>
          <p className="mb-6 text-sm text-muted-foreground">{t("workforces.create.modeSelectPrompt")}</p>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <button
              onClick={() => setMode("ai")}
              className="flex flex-col gap-4 rounded-xl border-2 border-indigo-200 bg-indigo-50/40 p-6 text-left transition-colors hover:border-indigo-400 hover:bg-indigo-50/70"
            >
              <div className="flex items-center justify-between">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-100">
                  <Sparkles className="h-5 w-5 text-indigo-500" />
                </div>
                <span className="rounded-full bg-indigo-600 px-2.5 py-0.5 text-xs font-semibold text-white">
                  {t("workforces.create.modeSelect.aiRecommended")}
                </span>
              </div>
              <div>
                <div className="mb-1 text-base font-semibold">{t("workforces.create.modeSelect.aiTitle")}</div>
                <p className="text-sm text-muted-foreground">{t("workforces.create.modeSelect.aiDescription")}</p>
              </div>
            </button>

            <button
              onClick={() => setMode("manual")}
              className="flex flex-col gap-4 rounded-xl border-2 border-border bg-card p-6 text-left transition-colors hover:border-muted-foreground/30 hover:bg-muted/20"
            >
              <div className="flex items-center justify-between">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
                  <SlidersHorizontal className="h-5 w-5 text-muted-foreground" />
                </div>
                <span className="text-sm text-muted-foreground">{t("workforces.create.modeSelect.manualSubtitle")}</span>
              </div>
              <div>
                <div className="mb-1 text-base font-semibold">{t("workforces.create.modeSelect.manualTitle")}</div>
                <p className="text-sm text-muted-foreground">{t("workforces.create.modeSelect.manualDescription")}</p>
              </div>
            </button>
          </div>
        </>
      )}

      {/* Mode: AI-assisted */}
      {mode === "ai" && (
        <>
          <button
            onClick={() => setMode("select")}
            className="mb-6 flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            {t("common.back")}
          </button>
          <div className="mb-6 flex items-start gap-4">
            <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-indigo-100">
              <Sparkles className="h-7 w-7 text-indigo-500" />
            </div>
            <div className="pt-1">
              <h2 className="text-xl font-bold">{t("workforces.create.prompt.describeTitle")}</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                {t("workforces.create.prompt.description")}
              </p>
            </div>
          </div>
          <WorkforcePromptCreator onCreated={onCreated} onCancel={() => setMode("select")} />
        </>
      )}

      {/* Mode: manual wizard — wizard has its own Back button that calls onBack */}
      {mode === "manual" && (
        <WorkforceWizard onCreated={onCreated} onBack={() => setMode("select")} />
      )}
    </div>
  )
}
