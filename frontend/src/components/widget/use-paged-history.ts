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
// later page leaves the loaded items in place. A response is written only
// while it belongs to the newest request; unmounting also takes a number and
// aborts the request in flight, so nothing lands after it.
export function usePagedHistory<T>(
  fetchPage: (cursor: string | null, signal: AbortSignal) => Promise<HistoryPage<T>>,
) {
  const [items, setItems] = useState<T[]>([])
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [firstPage, setFirstPage] = useState<"loading" | "ready" | "failed">("loading")
  const [morePages, setMorePages] = useState<"idle" | "loading" | "failed">("idle")
  const [errorKind, setErrorKind] = useState<WidgetSessionHistoryErrorKind>("load_failed")
  const latestRequestRef = useRef(0)
  const controllerRef = useRef<AbortController | null>(null)
  const fetchPageRef = useRef(fetchPage)

  useEffect(() => {
    fetchPageRef.current = fetchPage
  }, [fetchPage])

  const load = useCallback((cursor: string | null) => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    latestRequestRef.current += 1
    const request = latestRequestRef.current
    if (cursor === null) setFirstPage("loading")
    else setMorePages("loading")
    fetchPageRef.current(cursor, controller.signal).then(
      (page) => {
        if (request !== latestRequestRef.current) return
        setItems((current) => (cursor === null ? page.items : [...current, ...page.items]))
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
    load(null)
    return () => {
      latestRequestRef.current += 1
      controllerRef.current?.abort()
    }
  }, [load])

  return { items, nextCursor, firstPage, morePages, errorKind, load }
}
