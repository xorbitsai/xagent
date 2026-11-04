"use client"

import React from "react"
import { CheckCircle2, ChevronDown, ChevronUp } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"

export interface GetStartedStep {
  key: string
  label: string
  done: boolean
}

interface GetStartedChecklistProps {
  steps: GetStartedStep[]
  collapsed: boolean
  onToggleCollapsed: () => void
  className?: string
}

export function GetStartedChecklist({ steps, collapsed, onToggleCollapsed, className = "" }: GetStartedChecklistProps) {
  const { t } = useI18n()
  const doneCount = steps.filter((step) => step.done).length
  const progress = steps.length === 0 ? 0 : (doneCount / steps.length) * 100

  return (
    <div className={`rounded-xl border bg-card shadow-sm ${className}`}>
      <button
        type="button"
        onClick={onToggleCollapsed}
        className="flex w-full items-center justify-between gap-3 px-4 py-3"
      >
        <span className="text-sm font-semibold text-foreground">{t("workforces.getStarted.title")}</span>
        <div className="flex shrink-0 items-center gap-2">
          <span className="rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700">
            {doneCount}/{steps.length}
          </span>
          {collapsed ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          )}
        </div>
      </button>

      {collapsed ? (
        <div className="px-4 pb-3">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      ) : (
        <div className="space-y-0.5 px-2 pb-3">
          {steps.map((step, index) => (
            <div key={step.key} className="flex items-center gap-2.5 rounded-lg px-2 py-1.5">
              {step.done ? (
                <CheckCircle2 className="h-4 w-4 shrink-0 text-green-600" />
              ) : (
                <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-muted-foreground/40 text-[10px] font-medium text-muted-foreground">
                  {index + 1}
                </span>
              )}
              <span className={`text-sm ${step.done ? "text-foreground" : "text-muted-foreground"}`}>
                {step.label}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
