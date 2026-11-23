"use client"

import React, { useCallback, useEffect, useRef, useState } from "react"
import { Loader2, X } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import {
  getBrowserLocationHash,
  getBrowserLocationOrigin,
  getBrowserLocationPathname,
  getBrowserLocationSearch,
  replaceBrowserLocation,
} from "@/lib/browser-location"
import {
  createLinkSessionHost,
  type LinkHostSnapshot,
  type LinkSessionHost,
  type LinkWorkerLike,
} from "./link-session-host"

// Wiring of the link host page: mounts the chat iframe and the page's own close
// button, and connects window events to link-session-host.ts. All behaviour
// lives in the host; nothing here reads window while rendering.

// Overridable so tests can supply a Worker and observe navigation.
export interface LinkSessionPageEnvironment {
  createWorker?: () => LinkWorkerLike
  navigate?: (url: string) => void
}

// The Worker is created once, before the chat iframe exists, and bundled by
// Next as its own same-origin script.
function createLinkWorker(): LinkWorkerLike {
  return new Worker(new URL("./link-session-worker.ts", import.meta.url), { type: "module" })
}

const INITIAL_SNAPSHOT: LinkHostSnapshot = {
  phase: "starting",
  iframeSrc: null,
  closeButtonVisible: true,
}

export function LinkSessionPage({ environment }: { environment?: LinkSessionPageEnvironment }) {
  const { t } = useI18n()
  const [snapshot, setSnapshot] = useState<LinkHostSnapshot>(INITIAL_SNAPSHOT)
  const iframeRef = useRef<HTMLIFrameElement | null>(null)
  const hostRef = useRef<LinkSessionHost | null>(null)
  const pendingDisposeRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const environmentRef = useRef(environment)
  environmentRef.current = environment

  useEffect(() => {
    // React strict mode runs this effect, its cleanup and this effect again. The
    // link code can only be read once, so the host survives that sequence and
    // is disposed only if no effect run follows the cleanup.
    if (pendingDisposeRef.current !== null) {
      clearTimeout(pendingDisposeRef.current)
      pendingDisposeRef.current = null
    }
    let host = hostRef.current
    const isNew = host === null
    if (host === null) {
      host = createLinkSessionHost({
        readHash: getBrowserLocationHash,
        readPathname: getBrowserLocationPathname,
        readSearch: getBrowserLocationSearch,
        readOrigin: getBrowserLocationOrigin,
        replaceState: (url) => {
          try {
            window.history.replaceState(null, "", url)
          } catch {
            // The code is single-use; a browser that refuses to rewrite the
            // address must not stop the page from starting.
          }
        },
        navigate: (url) => {
          const navigate = environmentRef.current?.navigate
          if (navigate) navigate(url)
          else replaceBrowserLocation(url)
        },
        createWorker: () => (environmentRef.current?.createWorker ?? createLinkWorker)(),
        getIframeWindow: () => iframeRef.current?.contentWindow ?? null,
        postToIframe: (message) => {
          iframeRef.current?.contentWindow?.postMessage(message, getBrowserLocationOrigin())
        },
        now: () => Date.now(),
        setTimer: (callback, ms) => setTimeout(callback, ms),
        clearTimer: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
        onChange: setSnapshot,
      })
      hostRef.current = host
    }
    const active = host

    const onMessage = (event: MessageEvent) => {
      active.handleMessageEvent({ source: event.source, origin: event.origin, data: event.data })
    }
    const onPageShow = (event: Event) => {
      active.handlePageShow({ persisted: (event as PageTransitionEvent).persisted === true })
    }
    const onOnline = () => active.wake()
    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") active.wake()
    }
    window.addEventListener("message", onMessage)
    window.addEventListener("pageshow", onPageShow)
    window.addEventListener("online", onOnline)
    document.addEventListener("visibilitychange", onVisibilityChange)

    if (isNew) active.start()

    return () => {
      window.removeEventListener("message", onMessage)
      window.removeEventListener("pageshow", onPageShow)
      window.removeEventListener("online", onOnline)
      document.removeEventListener("visibilitychange", onVisibilityChange)
      pendingDisposeRef.current = setTimeout(() => {
        pendingDisposeRef.current = null
        hostRef.current?.dispose()
        hostRef.current = null
      }, 0)
    }
  }, [])

  const close = useCallback(() => hostRef.current?.close(), [])

  return (
    <div className="fixed inset-0 bg-background text-foreground">
      {snapshot.phase === "terminated" ? (
        <div role="status" className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
          <p className="text-base font-medium">{t("widgetLink.ended.generic.title")}</p>
          <p className="text-sm text-muted-foreground">{t("widgetLink.ended.generic.description")}</p>
        </div>
      ) : snapshot.iframeSrc ? (
        <iframe
          ref={iframeRef}
          src={snapshot.iframeSrc}
          title={t("widgetChat.title")}
          allow="clipboard-write"
          className="fixed border-0"
          style={{
            top: "env(safe-area-inset-top)",
            right: "env(safe-area-inset-right)",
            bottom: "env(safe-area-inset-bottom)",
            left: "env(safe-area-inset-left)",
            width: "calc(100vw - env(safe-area-inset-left) - env(safe-area-inset-right))",
            height: "calc(100vh - env(safe-area-inset-top) - env(safe-area-inset-bottom))",
          }}
        />
      ) : (
        <div role="status" className="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          <span>{t("widgetLink.loading")}</span>
        </div>
      )}
      {snapshot.closeButtonVisible && (
        <button
          type="button"
          onClick={close}
          aria-label={t("widgetLink.close")}
          title={t("widgetLink.close")}
          className="fixed z-10 rounded-md p-2 text-muted-foreground transition-colors hover:bg-muted/50 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          style={{
            top: "calc(env(safe-area-inset-top) + 8px)",
            right: "calc(env(safe-area-inset-right) + 8px)",
          }}
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      )}
    </div>
  )
}
