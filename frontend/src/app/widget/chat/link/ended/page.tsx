"use client"

import React, { Suspense } from "react"
import { useSearchParams } from "next/navigation"
import { useI18n } from "@/contexts/i18n-context"
import { END_REASONS, type EndReason } from "@/components/widget/link-session-constants"

// The URL of this page, in particular `reason`, is the signal the embedding app
// reads from its WebView navigation hooks. The text below is for the person
// looking at the screen when the app did not intercept the navigation.
const COPY = {
  link_invalid: { title: "widgetLink.ended.link_invalid.title", description: "widgetLink.ended.link_invalid.description" },
  session_ended: { title: "widgetLink.ended.session_ended.title", description: "widgetLink.ended.session_ended.description" },
  unavailable: { title: "widgetLink.ended.unavailable.title", description: "widgetLink.ended.unavailable.description" },
  network: { title: "widgetLink.ended.network.title", description: "widgetLink.ended.network.description" },
  closed: { title: "widgetLink.ended.closed.title", description: "widgetLink.ended.closed.description" },
  generic: { title: "widgetLink.ended.generic.title", description: "widgetLink.ended.generic.description" },
} as const

function isEndReason(value: string | null): value is EndReason {
  return value !== null && (END_REASONS as readonly string[]).includes(value)
}

function LinkEnded() {
  const { t } = useI18n()
  const reason = useSearchParams().get("reason")
  const copy = COPY[isEndReason(reason) ? reason : "generic"]

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-2 bg-background p-6 text-center text-foreground">
      <h1 className="text-base font-medium">{t(copy.title)}</h1>
      <p className="text-sm text-muted-foreground">{t(copy.description)}</p>
    </main>
  )
}

export default function LinkEndedPage() {
  // useSearchParams must be inside a Suspense boundary for static export.
  return (
    <Suspense fallback={null}>
      <LinkEnded />
    </Suspense>
  )
}
