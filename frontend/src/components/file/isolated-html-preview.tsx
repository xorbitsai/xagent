"use client"

import React from "react"
import { resolveHtmlPreviewRunnerUrl } from "../../../html-preview-config.mjs"

export { resolveHtmlPreviewRunnerUrl } from "../../../html-preview-config.mjs"

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
    let active = true
    let delivered = false
    const onError = () => setFailed(true)
    const timer = window.setTimeout(() => setFailed(true), READY_TIMEOUT_MS)
    const onMessage = (event: MessageEvent<unknown>) => {
      if (!active || delivered || event.source !== frame.contentWindow || event.origin !== "null") return
      const data = event.data
      if (!data || typeof data !== "object"
        || (data as { type?: unknown }).type !== PREVIEW_READY_MESSAGE
        || (data as { version?: unknown }).version !== PREVIEW_PROTOCOL_VERSION) return

      delivered = true
      window.clearTimeout(timer)
      window.removeEventListener("message", onMessage)
      setReady(true)
      // An opaque sandbox requires "*". Pin the exact, fresh frame window;
      // never send credentials or deliver a second document to a used runner.
      frame.contentWindow?.postMessage({
        type: PREVIEW_RENDER_MESSAGE, version: PREVIEW_PROTOCOL_VERSION, html,
      }, "*")
    }
    window.addEventListener("message", onMessage)
    frame.addEventListener("error", onError)
    // Register before navigating, so an immediately ready runner is not lost.
    frame.src = runnerUrl
    return () => {
      active = false
      window.clearTimeout(timer)
      window.removeEventListener("message", onMessage)
      frame.removeEventListener("error", onError)
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
  const [browserOrigin, setBrowserOrigin] = React.useState<string>()
  React.useEffect(() => setBrowserOrigin(window.location.origin), [])
  const runnerUrl = resolveHtmlPreviewRunnerUrl(configured, browserOrigin)
  if (!props.html || (configured === "auto" && !browserOrigin)) return null
  // A different document/config gets a new browsing context, not the previous
  // document's persistent WindowProxy. No hash or cache-busting URL is needed.
  return <PreviewDocument key={JSON.stringify([runnerUrl, props.html])} {...props} runnerUrl={runnerUrl} />
}
