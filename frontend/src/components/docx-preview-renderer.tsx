"use client"

import { useEffect, useRef, useState } from "react"

interface DocxPreviewRendererProps {
  base64Content: string
}

export function DocxPreviewRenderer({ base64Content }: DocxPreviewRendererProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const render = async () => {
      if (!containerRef.current || !base64Content) {
        return
      }

      try {
        const binary = atob(base64Content)
        const bytes = new Uint8Array(binary.length)
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i)
        }

        containerRef.current.innerHTML = ""
        const docxPreview = await import("docx-preview")
        await docxPreview.renderAsync(bytes.buffer, containerRef.current, undefined, {
          className: "docx-preview",
          inWrapper: true,
          useBase64URL: true,
        })
        setError(null)
      } catch {
        setError("Failed to render DOCX preview")
      }
    }

    render()
  }, [base64Content])

  if (error) {
    return <div className="p-4 text-sm text-muted-foreground">{error}</div>
  }

  return <div ref={containerRef} className="h-full overflow-auto bg-muted/30" />
}
