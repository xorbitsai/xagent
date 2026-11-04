"use client"

import React, { useEffect, useRef, useState } from "react"
import { useI18n } from "@/contexts/i18n-context"
import { renderSanitizedDocx } from "@/components/file/docx-preview-sanitize"

interface DocxPreviewRendererProps {
  base64Content: string
}

/** Scroll to in-document bookmarks instead of changing the app URL hash. */
function handleBookmarkClick(this: ShadowRoot, event: Event) {
  const target = event.target
  const link = target instanceof Element ? target.closest("a[href^='#']") : null
  if (!link) return
  event.preventDefault()
  let id = (link.getAttribute("href") ?? "").slice(1)
  try {
    id = decodeURIComponent(id)
  } catch {
    // Keep the raw fragment.
  }
  if (id) this.getElementById(id)?.scrollIntoView({ block: "start" })
}

/**
 * docx-preview sizes VML shapes with getBBox() once they are attached. Its
 * measurement ran against the detached staging copy, so repeat it for the
 * sanitized copy that is now on screen.
 */
function sizeVmlShapes(root: ShadowRoot) {
  for (const svg of Array.from(root.querySelectorAll("svg"))) {
    const shape = svg.firstElementChild as (Element & { getBBox?: () => DOMRect }) | null
    if (typeof shape?.getBBox !== "function") continue
    try {
      const box = shape.getBBox()
      if (!box.width && !box.height) continue
      svg.setAttribute("width", `${Math.ceil(box.x + box.width)}`)
      svg.setAttribute("height", `${Math.ceil(box.y + box.height)}`)
    } catch {
      // Unrendered or unsupported shape: keep the default size.
    }
  }
}

export function DocxPreviewRenderer({ base64Content }: DocxPreviewRendererProps) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { t } = useI18n()

  useEffect(() => {
    let cancelled = false
    let frame: number | null = null

    const render = async () => {
      const host = hostRef.current
      if (!host || !base64Content) {
        return
      }

      try {
        const binary = atob(base64Content)
        const bytes = new Uint8Array(binary.length)
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i)
        }

        // The document is untrusted. docx-preview renders into a detached
        // container and only the DOMPurify output reaches the page. A shadow
        // root keeps the document's CSS from restyling the rest of the app.
        const docxPreview = await import("docx-preview")
        const content = await renderSanitizedDocx(docxPreview, bytes.buffer)
        if (cancelled) return
        let root = host.shadowRoot
        if (!root) {
          root = host.attachShadow({ mode: "open" })
          root.addEventListener("click", handleBookmarkClick)
        }
        root.replaceChildren(content)
        const shadowRoot = root
        frame = window.requestAnimationFrame(() => sizeVmlShapes(shadowRoot))
        setError(null)
      } catch {
        if (!cancelled) setError(t("files.previewDialog.errors.docxRenderFailed"))
      }
    }

    render()
    return () => {
      cancelled = true
      if (frame !== null) window.cancelAnimationFrame(frame)
    }
  }, [base64Content, t])

  if (error) {
    return <div className="p-4 text-sm text-muted-foreground">{error}</div>
  }

  // Layout and paint containment make this element the containing block for
  // fixed-position content and clip it, so document CSS cannot overlay the app.
  return (
    <div className="h-full overflow-auto bg-muted/30" style={{ contain: "layout paint" }}>
      <div ref={hostRef} data-testid="docx-preview-host" />
    </div>
  )
}
