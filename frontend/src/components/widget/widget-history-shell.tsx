"use client"

import React, { useEffect, useRef } from "react"
import { ArrowLeft, Loader2 } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"

export interface HistoryShellProps {
  title: string
  onBack: () => void
  // While true the body shows only the loading status.
  loading: boolean
  children: React.ReactNode
  // Shown below the scrolling body, outside it.
  footer?: React.ReactNode
}

// The frame the history list and the history view share: a back button that
// takes focus when the frame mounts, the title, and a scrolling body.
export function HistoryShell({ title, onBack, loading, children, footer }: HistoryShellProps) {
  const { t } = useI18n()
  const backRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    backRef.current?.focus()
  }, [])

  return (
    <section className="flex min-h-0 flex-1 flex-col" aria-label={title}>
      <div className="flex flex-none items-center gap-2 border-b px-3 py-2">
        <button
          ref={backRef}
          type="button"
          onClick={onBack}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-sm text-muted-foreground hover:bg-muted/50 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          {t("widgetSession.history.back")}
        </button>
        <h2 className="truncate text-sm font-semibold">{title}</h2>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {loading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground" role="status">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
            <span className="sr-only">{t("common.loading")}</span>
          </div>
        ) : children}
      </div>
      {footer}
    </section>
  )
}

export function LoadMoreButton({ loading, onClick }: { loading: boolean; onClick: () => void }) {
  const { t } = useI18n()
  return (
    <button
      type="button"
      disabled={loading}
      onClick={onClick}
      className="mt-3 rounded-md border px-3 py-1.5 text-sm hover:bg-muted/50 disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      {t("widgetSession.history.loadMore")}
    </button>
  )
}
