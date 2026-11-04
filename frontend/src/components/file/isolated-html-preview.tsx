"use client"

import React from "react"
import { resolveHtmlPreviewRunnerUrl } from "../../../html-preview-config.mjs"

const PREVIEW_READY_MESSAGE = "xagent-html-preview-ready"
const PREVIEW_RENDER_MESSAGE = "xagent-html-preview-render"
const PREVIEW_PROTOCOL_VERSION = 1
const READY_TIMEOUT_MS = 8_000

interface IsolatedHtmlPreviewProps {
  html: string
  title: string
  className?: string
}

function PreviewDocument({ html, title, className, runnerUrl }: IsolatedHtmlPreviewProps & { runnerUrl: string | null }) {
  const iframeRef = React.useRef<HTMLIFrameElement>(null)
  const [failed, setFailed] = React.useState(false)
  const [ready, setReady] = React.useState(false)

  React.useEffect(() => {
    if (!runnerUrl || failed) return
    const frame = iframeRef.current
    if (!frame) return
    const timer = window.setTimeout(() => setFailed(true), READY_TIMEOUT_MS)
    const onMessage = (event: MessageEvent<unknown>) => {
      if (event.source !== frame.contentWindow || event.origin !== "null") return
      const data = event.data
      if (!data || typeof data !== "object"
        || (data as { type?: unknown }).type !== PREVIEW_READY_MESSAGE
        || (data as { version?: unknown }).version !== PREVIEW_PROTOCOL_VERSION) return

      window.clearTimeout(timer)
      window.removeEventListener("message", onMessage)
      setReady(true)
      // An opaque sandbox requires "*". Pin the exact, fresh frame window;
      // deliver only once. Do not add ambient app credentials here; generated
      // HTML may already contain file-scoped access tokens from its policy.
      frame.contentWindow?.postMessage({
        type: PREVIEW_RENDER_MESSAGE, version: PREVIEW_PROTOCOL_VERSION, html,
      }, "*")
    }
    window.addEventListener("message", onMessage)
    // Register before navigating, so an immediately ready runner is not lost.
    frame.src = runnerUrl
    return () => {
      window.clearTimeout(timer)
      window.removeEventListener("message", onMessage)
    }
  }, [html, runnerUrl, failed])

  const useRunner = Boolean(runnerUrl) && !failed
  return <iframe
    key={useRunner ? "runner" : "fallback"}
    ref={iframeRef}
    {...(useRunner ? {} : { srcDoc: html })}
    aria-busy={useRunner && !ready}
    className={className}
    sandbox="allow-scripts"
    referrerPolicy="no-referrer"
    title={title}
  />
}

export function IsolatedHtmlPreview(props: IsolatedHtmlPreviewProps) {
  const configured = process.env.NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN?.trim()
  const [browserContext, setBrowserContext] = React.useState<{
    origin: string
    embedded: boolean
  }>()
  React.useEffect(() => setBrowserContext({
    origin: window.location.origin,
    // The runner permits only the app origin in frame-ancestors. If the app
    // itself is embedded, every ancestor must be allowed, so retain srcdoc.
    embedded: window.top !== window.self,
  }), [])
  const runnerUrl = browserContext?.embedded
    ? null
    : resolveHtmlPreviewRunnerUrl(configured, browserContext?.origin)
  const generation = React.useRef({ html: "", runnerUrl: null as string | null, key: 0 })
  if (generation.current.html !== props.html || generation.current.runnerUrl !== runnerUrl) {
    generation.current = { html: props.html, runnerUrl, key: generation.current.key + 1 }
  }
  if (!props.html || !browserContext) return null
  // A different document/config gets a new browsing context, not the previous
  // document's persistent WindowProxy. No hash or cache-busting URL is needed.
  return <PreviewDocument key={generation.current.key} {...props} runnerUrl={runnerUrl} />
}
