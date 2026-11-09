"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import {
  WidgetSessionHistoryError,
  type WidgetSessionHistoryErrorKind,
} from "@/lib/widget-session-history"

export const readErrorKind = (error: unknown): WidgetSessionHistoryErrorKind =>
  error instanceof WidgetSessionHistoryError ? error.kind : "load_failed"

export interface HistoryPage<T> {
  items: T[]
  nextCursor: string | null
}

// Reads a history resource page by page, from the first page on mount. The
// first page, and the pages after it, fail and retry separately: a failed
// later page leaves the loaded items in place. Every load aborts the request
// before it, and unmounting aborts the one in flight, so a response is written
// only while its request has not been aborted and nothing lands after unmount.
// `fetchPage` must keep its identity across renders: a new one starts over
// from the first page.
export function usePagedHistory<T>(
  fetchPage: (cursor: string | null, signal: AbortSignal) => Promise<HistoryPage<T>>,
) {
  const [items, setItems] = useState<T[]>([])
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [firstPage, setFirstPage] = useState<"loading" | "ready" | "failed">("loading")
  const [morePages, setMorePages] = useState<"idle" | "loading" | "failed">("idle")
  const [errorKind, setErrorKind] = useState<WidgetSessionHistoryErrorKind>("load_failed")
  const controllerRef = useRef<AbortController | null>(null)

  const load = useCallback((cursor: string | null) => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    if (cursor === null) setFirstPage("loading")
    else setMorePages("loading")
    fetchPage(cursor, controller.signal).then(
      (page) => {
        if (controller.signal.aborted) return
        setItems((current) => (cursor === null ? page.items : [...current, ...page.items]))
        setNextCursor(page.nextCursor)
        setFirstPage("ready")
        setMorePages("idle")
      },
      (error: unknown) => {
        if (controller.signal.aborted) return
        setErrorKind(readErrorKind(error))
        if (cursor === null) setFirstPage("failed")
        else setMorePages("failed")
      },
    )
  }, [fetchPage])

  useEffect(() => {
    load(null)
    return () => {
      controllerRef.current?.abort()
    }
  }, [load])

  return { items, nextCursor, firstPage, morePages, errorKind, load }
}
