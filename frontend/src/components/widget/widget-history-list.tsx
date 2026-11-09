"use client"

import React, { useCallback, useEffect, useRef, useState } from "react"
import { ArrowLeft, Loader2 } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { normalizeTimestampMs } from "@/lib/time-utils"
import {
  listWidgetConversations,
  WidgetSessionHistoryError,
  type WidgetConversationSummary,
  type WidgetSessionHistoryAuth,
  type WidgetSessionHistoryErrorKind,
} from "@/lib/widget-session-history"

export interface WidgetHistoryListProps {
  origin: string
  // Called for every request, so a request always uses the token policy of the
  // conversation phase the session is in at that moment.
  getAuth: () => WidgetSessionHistoryAuth
  // What to say when a request is rejected as unauthorized and may not be
  // refreshed in the current phase.
  unauthorizedNotice: "refreshToLoad" | "tryLater"
  // The conversation the visitor is in; choosing it goes back to the chat.
  currentTaskId: number | null
  onBack: () => void
  onOpenCurrent: () => void
  onOpen: (taskId: number, title: string) => void
}

const readErrorKind = (error: unknown): WidgetSessionHistoryErrorKind =>
  error instanceof WidgetSessionHistoryError ? error.kind : "load_failed"

const formatCreatedAt = (createdAt: string): string =>
  new Date(normalizeTimestampMs(createdAt)).toLocaleString([], {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })

// The earlier conversations of this end user, newest first as the server
// returns them. Every visit reads from the first page again; nothing is kept.
export function WidgetHistoryList({
  origin,
  getAuth,
  unauthorizedNotice,
  currentTaskId,
  onBack,
  onOpenCurrent,
  onOpen,
}: WidgetHistoryListProps) {
  const { t } = useI18n()
  const [items, setItems] = useState<WidgetConversationSummary[]>([])
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  // The first page, and the pages after it, fail and retry separately: a
  // failed later page leaves the loaded conversations on screen.
  const [firstPage, setFirstPage] = useState<"loading" | "ready" | "failed">("loading")
  const [morePages, setMorePages] = useState<"idle" | "loading" | "failed">("idle")
  const [errorKind, setErrorKind] = useState<WidgetSessionHistoryErrorKind>("load_failed")
  const backRef = useRef<HTMLButtonElement>(null)
  // A response is written only while it belongs to the newest request; a
  // closing list also takes a number, so nothing lands after it.
  const latestRequestRef = useRef(0)
  const controllerRef = useRef<AbortController | null>(null)
  const sourceRef = useRef({ origin, getAuth })

  useEffect(() => {
    sourceRef.current = { origin, getAuth }
  }, [origin, getAuth])

  const load = useCallback((cursor: string | null) => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    latestRequestRef.current += 1
    const request = latestRequestRef.current
    if (cursor === null) setFirstPage("loading")
    else setMorePages("loading")
    const source = sourceRef.current
    listWidgetConversations(source.origin, source.getAuth(), { cursor }, controller.signal).then(
      (page) => {
        if (request !== latestRequestRef.current) return
        setItems((current) => (cursor === null ? page.conversations : [...current, ...page.conversations]))
        setNextCursor(page.nextCursor)
        setFirstPage("ready")
        setMorePages("idle")
      },
      (error: unknown) => {
        if (request !== latestRequestRef.current) return
        setErrorKind(readErrorKind(error))
        if (cursor === null) setFirstPage("failed")
        else setMorePages("failed")
      },
    )
  }, [])

  useEffect(() => {
    backRef.current?.focus()
    load(null)
    return () => {
      latestRequestRef.current += 1
      controllerRef.current?.abort()
    }
  }, [load])

  const failureText = errorKind === "unauthorized"
    ? t(`widgetSession.history.${unauthorizedNotice}`)
    : errorKind === "disabled"
      ? t("widgetSession.history.disabled")
      : t("widgetSession.history.loadFailed")

  const failure = (retry: () => void) => (
    <div className="flex flex-col items-start gap-2 text-sm">
      <p className="text-destructive" role="alert">{failureText}</p>
      <button
        type="button"
        onClick={retry}
        className="rounded-md border px-3 py-1.5 text-sm hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {t("widgetSession.history.retry")}
      </button>
    </div>
  )

  return (
    <section className="flex min-h-0 flex-1 flex-col" aria-label={t("widgetSession.history.listTitle")}>
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
        <h2 className="truncate text-sm font-semibold">{t("widgetSession.history.listTitle")}</h2>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {firstPage === "loading" ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground" role="status">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          </div>
        ) : firstPage === "failed" ? (
          failure(() => load(null))
        ) : items.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("widgetSession.history.listEmpty")}</p>
        ) : (
          <>
            <ul className="flex flex-col gap-1">
              {items.map((item) => (
                <li key={item.taskId}>
                  <button
                    type="button"
                    onClick={() => (
                      item.taskId === currentTaskId ? onOpenCurrent() : onOpen(item.taskId, item.title)
                    )}
                    className="flex w-full flex-col items-start gap-0.5 rounded-md border px-3 py-2 text-left hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <span className="w-full truncate text-sm font-medium">{item.title}</span>
                    <span className="text-xs text-muted-foreground">
                      {formatCreatedAt(item.createdAt)}
                      {item.taskId === currentTaskId ? (
                        <span className="ml-2 rounded bg-primary/10 px-1.5 py-0.5 text-primary">
                          {t("widgetSession.history.currentConversation")}
                        </span>
                      ) : null}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            {morePages === "failed" ? (
              <div className="mt-3">{failure(() => load(nextCursor))}</div>
            ) : nextCursor !== null ? (
              <button
                type="button"
                disabled={morePages === "loading"}
                onClick={() => load(nextCursor)}
                className="mt-3 rounded-md border px-3 py-1.5 text-sm hover:bg-muted/50 disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {t("widgetSession.history.loadMore")}
              </button>
            ) : null}
          </>
        )}
      </div>
    </section>
  )
}
