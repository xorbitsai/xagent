"use client"

import React from "react"

const PREVIEW_READY_MESSAGE = "xagent-html-preview-ready"
const PREVIEW_RENDER_MESSAGE = "xagent-html-preview-render"
const PREVIEW_PROTOCOL_VERSION = 1

type PreviewMode =
  | { kind: "pending" }
  | { kind: "srcdoc" }
  | { kind: "runner"; url: string }

interface IsolatedHtmlPreviewProps {
  html: string
  title: string
  className?: string
  /** Override the build-time setting in focused tests and embedding clients. */
  previewOrigin?: string
}

function isLocalHostname(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]"
}

/** Resolve an explicit origin, or XAgent SaaS's environment-specific preview host. */
export function resolveHtmlPreviewRunnerUrl(
  configuredOrigin: string | undefined,
  browserOrigin?: string,
): string | null {
  const configured = configuredOrigin?.trim()
  if (!configured) return null

  let candidate = configured
  if (configured === "auto") {
    if (!browserOrigin) return null
    let browserUrl: URL
    try {
      browserUrl = new URL(browserOrigin)
    } catch {
      return null
    }
    const previewHost = browserUrl.hostname === "cloud.xagent.co"
      ? "preview.xagent.co"
      : browserUrl.hostname === "stage.cloud.xagent.co"
        ? "preview.stage.xagent.co"
        : browserUrl.hostname === "dev.cloud.xagent.co"
          ? "preview.dev.xagent.co"
          : null
    if (!previewHost) return null
    candidate = `https://${previewHost}`
  }

  try {
    const url = new URL(candidate)
    if (url.username || url.password) return null
    if (url.protocol !== "https:" && !(url.protocol === "http:" && isLocalHostname(url.hostname))) {
      return null
    }
    return `${url.origin}/render`
  } catch {
    return null
  }
}

function contentVersion(source: string): string {
  // A short deterministic FNV-1a hash makes a changed document perform a real
  // navigation. A fragment-only change would leave hostile content running and
  // merely dispatch hashchange instead of restoring the trusted runner.
  let hash = 0x811c9dc5
  for (let index = 0; index < source.length; index += 1) {
    hash ^= source.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(16)
}

export function IsolatedHtmlPreview({
  html,
  title,
  className,
  previewOrigin = process.env.NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN,
}: IsolatedHtmlPreviewProps) {
  const iframeRef = React.useRef<HTMLIFrameElement>(null)
  const [mode, setMode] = React.useState<PreviewMode>(() => {
    if (!previewOrigin?.trim()) return { kind: "srcdoc" }
    if (previewOrigin.trim() === "auto") return { kind: "pending" }
    const url = resolveHtmlPreviewRunnerUrl(previewOrigin)
    return url ? { kind: "runner", url } : { kind: "srcdoc" }
  })

  React.useEffect(() => {
    if (previewOrigin?.trim() !== "auto") return
    const url = resolveHtmlPreviewRunnerUrl("auto", window.location.origin)
    setMode(url ? { kind: "runner", url } : { kind: "srcdoc" })
  }, [previewOrigin])

  React.useEffect(() => {
    if (mode.kind !== "runner") return
    const frame = iframeRef.current
    if (!frame) return

    let delivered = false
    const onMessage = (event: MessageEvent<unknown>) => {
      if (delivered || event.source !== frame.contentWindow) return
      const data = event.data
      if (
        !data
        || typeof data !== "object"
        || (data as { type?: unknown }).type !== PREVIEW_READY_MESSAGE
        || (data as { version?: unknown }).version !== PREVIEW_PROTOCOL_VERSION
      ) {
        return
      }

      delivered = true
      window.removeEventListener("message", onMessage)
      // The sandbox deliberately gives the frame an opaque origin, so the
      // browser requires "*" here. The destination is still pinned by the
      // exact contentWindow identity, the listener is one-shot, and the only
      // payload is the document that this frame exists to render.
      frame.contentWindow?.postMessage(
        {
          type: PREVIEW_RENDER_MESSAGE,
          version: PREVIEW_PROTOCOL_VERSION,
          html,
        },
        "*",
      )
    }

    window.addEventListener("message", onMessage)
    const url = new URL(mode.url)
    url.searchParams.set("v", contentVersion(html))
    frame.src = url.toString()
    return () => window.removeEventListener("message", onMessage)
  }, [html, mode])

  return (
    <iframe
      ref={iframeRef}
      {...(mode.kind === "srcdoc" ? { srcDoc: html } : {})}
      className={className}
      // Scripts remain compatible, including third-party HTTPS scripts, but
      // the missing allow-same-origin capability keeps the document opaque.
      sandbox="allow-scripts"
      referrerPolicy="no-referrer"
      title={title}
    />
  )
}
