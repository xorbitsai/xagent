"use client"

import React, { useCallback, useEffect, useRef } from "react"
import { useI18n } from "@/contexts/i18n-context"
import { normalizeTimestampMs } from "@/lib/time-utils"
import {
  listWidgetConversations,
  type WidgetSessionHistoryAuth,
} from "@/lib/widget-session-history"
import { usePagedHistory } from "./use-paged-history"
import { HistoryShell, LoadMoreButton } from "./widget-history-shell"

export interface WidgetHistoryListProps {
  origin: string
  // Called for every request, so a request always uses the token policy of the
  // conversation phase the session is in at that moment.
  getAuth: () => WidgetSessionHistoryAuth
  // What to say when a request is rejected as unauthorized and may not be
  // refreshed in the current phase.
  unauthorizedNotice: "refreshToLoad" | "tryLater"
  // The conversation the visitor is in; choosing it goes back to the chat,
  // the same way as the back button.
  currentTaskId: number | null
  onBack: () => void
  onOpen: (taskId: number, title: string) => void
}

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
  onOpen,
}: WidgetHistoryListProps) {
  const { t } = useI18n()
  const sourceRef = useRef({ origin, getAuth })

  useEffect(() => {
    sourceRef.current = { origin, getAuth }
  }, [origin, getAuth])

  const fetchPage = useCallback((cursor: string | null, signal: AbortSignal) => {
    const source = sourceRef.current
    return listWidgetConversations(source.origin, source.getAuth(), { cursor }, signal).then(
      (page) => ({ items: page.conversations, nextCursor: page.nextCursor }),
    )
  }, [])
  const { items, nextCursor, firstPage, morePages, errorKind, load } = usePagedHistory(fetchPage)

  const failureText = errorKind === "unauthorized"
    ? t(`widgetSession.history.${unauthorizedNotice}`)
    : errorKind === "disabled"
      ? t("widgetSession.history.disabled")
      : t("widgetSession.history.loadFailed")

  // A reload is the only way past "refreshToLoad", so a retry would only fail again.
  const offersRetry = !(errorKind === "unauthorized" && unauthorizedNotice === "refreshToLoad")

  const failure = (retry: () => void) => (
    <div className="flex flex-col items-start gap-2 text-sm">
      <p className="text-destructive" role="alert">{failureText}</p>
      {offersRetry ? (
        <button
          type="button"
          onClick={retry}
          className="rounded-md border px-3 py-1.5 text-sm hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {t("widgetSession.history.retry")}
        </button>
      ) : null}
    </div>
  )

  return (
    <HistoryShell title={t("widgetSession.history.listTitle")} onBack={onBack} loading={firstPage === "loading"}>
      {firstPage === "failed" ? (
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
                    item.taskId === currentTaskId ? onBack() : onOpen(item.taskId, item.title)
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
            <LoadMoreButton loading={morePages === "loading"} onClick={() => load(nextCursor)} />
          ) : null}
        </>
      )}
    </HistoryShell>
  )
}
