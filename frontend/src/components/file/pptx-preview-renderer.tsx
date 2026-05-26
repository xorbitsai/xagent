"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { ChevronLeft, ChevronRight, Loader2 } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

interface PptxPreviewRendererProps {
  base64Content: string
  /**
   * Optional fileId. When provided, the renderer tries the high-fidelity
   * server-rendered PDF preview first (LibreOffice → PDF, displayed in an
   * iframe with the browser's native PDF viewer — vector, text-selectable,
   * correct font metrics). On a 503 (LibreOffice not installed on the
   * server) or any other failure, the renderer transparently falls back
   * to the canvas-based pptxviewjs rendering that ships with the bundle,
   * so developer machines without LibreOffice still get a working preview.
   */
  fileId?: string
}

// Minimal subset of the runtime API we rely on. We avoid pulling in the
// library's types at module load time because pptxviewjs is browser-only
// (it touches HTMLCanvas + ResizeObserver), so we dynamic-import it inside
// useEffect to stay SSR-safe under Next.js.
type PPTXViewerHandle = {
  loadFile(input: ArrayBuffer | Uint8Array | File): Promise<unknown>
  render(
    canvas?: HTMLCanvasElement | null,
    options?: { slideIndex?: number },
  ): Promise<unknown>
  nextSlide(): Promise<unknown>
  previousSlide(): Promise<unknown>
  goToSlide(index: number): Promise<unknown>
  getSlideCount(): number
  getCurrentSlideIndex(): number
  on(event: string, cb: (...args: unknown[]) => void): void
  destroy(): void
}

function base64ToUint8Array(b64: string): Uint8Array {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return bytes
}

export function PptxPreviewRenderer({ base64Content, fileId }: PptxPreviewRendererProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const viewerRef = useRef<PPTXViewerHandle | null>(null)

  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [slideCount, setSlideCount] = useState<number>(0)
  const [currentSlide, setCurrentSlide] = useState<number>(0)
  // PDF-first preview: when the server can produce a LibreOffice-rendered
  // PDF we display that in an iframe instead of the canvas. `pdfUrl` is set
  // by the effect below; until it resolves we render the canvas pipeline,
  // and on 503 / fetch failure we leave it null and keep the canvas.
  // `pdfChecked` flips to true once the probe finishes (either way) so the
  // canvas effect can decide whether to bother loading pptxviewjs.
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [pdfChecked, setPdfChecked] = useState<boolean>(!fileId)
  const { t } = useI18n()

  // Probe the server-rendered PDF endpoint. If it 200s, use the PDF in an
  // iframe (high fidelity). If it 503s — LibreOffice not on PATH server-side
  // — silently fall back to the canvas renderer. Any other error also falls
  // back, so the UX never regresses below today's behaviour.
  useEffect(() => {
    if (!fileId) return
    let cancelled = false
    let objectUrl: string | null = null
    ;(async () => {
      try {
        const res = await apiRequest(
          `${getApiUrl()}/api/files/preview-pdf/${encodeURIComponent(fileId)}`,
          { cache: "no-cache" },
        )
        if (cancelled) return
        if (res.ok) {
          const blob = await res.blob()
          if (cancelled) return
          objectUrl = URL.createObjectURL(blob)
          setPdfUrl(objectUrl)
          setIsLoading(false)
        }
      } catch {
        // network error — just fall back
      } finally {
        if (!cancelled) setPdfChecked(true)
      }
    })()
    return () => {
      cancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [fileId])

  // Size the canvas backing store to the container before each render.
  // pptxviewjs uses the canvas's pixel dimensions to lay out slides in
  // 'fit' mode, so we keep them in sync with the visible area. We also
  // multiply by devicePixelRatio so text is rendered at full Retina
  // resolution — without this, glyphs (especially small CJK text) look
  // washed out because the bundle ignores `scale`/`quality` render
  // options at runtime and draws directly into canvas pixels.
  const syncCanvasSize = useCallback(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return false
    const cssW = Math.max(1, Math.floor(container.clientWidth))
    const cssH = Math.max(1, Math.floor(container.clientHeight))
    const rawDpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1
    // Clamp to 3 to avoid blowing memory on rare 4x displays.
    const dpr = Math.max(1, Math.min(3, rawDpr))
    const physW = Math.max(1, Math.floor(cssW * dpr))
    const physH = Math.max(1, Math.floor(cssH * dpr))
    const styleW = `${cssW}px`
    const styleH = `${cssH}px`
    if (
      canvas.width === physW &&
      canvas.height === physH &&
      canvas.style.width === styleW &&
      canvas.style.height === styleH
    ) {
      return false
    }
    canvas.width = physW
    canvas.height = physH
    canvas.style.width = styleW
    canvas.style.height = styleH
    return true
  }, [])

  // Load + initial render.
  useEffect(() => {
    let cancelled = false
    let createdViewer: PPTXViewerHandle | null = null

    const load = async () => {
      // PDF path is winning: skip the canvas pipeline entirely so we don't
      // download pptxviewjs (~1MB chunk), parse the deck, or paint the
      // canvas that the iframe is going to cover anyway.
      if (pdfUrl) {
        setIsLoading(false)
        return
      }
      // We have a fileId but haven't finished probing the PDF endpoint
      // yet. Wait — kicking off pptxviewjs now would race the PDF probe
      // and could leave both pipelines fighting for the same canvas.
      if (fileId && !pdfChecked) return
      // Empty payload: nothing to render. Drop the loading spinner so we
      // don't hang the UI in an infinite loading state.
      if (!base64Content) {
        setIsLoading(false)
        return
      }
      if (!canvasRef.current) return
      setIsLoading(true)
      setError(null)

      try {
        const bytes = base64ToUint8Array(base64Content)

        const mod = await import("pptxviewjs")
        if (cancelled) return

        const ViewerCtor =
          (mod as { PPTXViewer?: new (opts: Record<string, unknown>) => PPTXViewerHandle }).PPTXViewer ??
          (mod as { default?: { PPTXViewer?: new (opts: Record<string, unknown>) => PPTXViewerHandle } }).default?.PPTXViewer
        if (!ViewerCtor) {
          throw new Error("pptxviewjs: PPTXViewer constructor not found")
        }

        // Tear down any previous viewer before creating a new one.
        viewerRef.current?.destroy()
        viewerRef.current = null

        syncCanvasSize()

        const viewer = new ViewerCtor({
          canvas: canvasRef.current,
          slideSizeMode: "fit",
          backgroundColor: "#ffffff",
        })

        viewer.on("slideChanged", (...args: unknown[]) => {
          const idx = args[0]
          if (typeof idx === "number") setCurrentSlide(idx)
        })

        // pptxviewjs@1.1.9: ``loadFile()`` parses the deck and emits
        // ``loadComplete`` but does NOT render a slide — the
        // ``autoRenderFirstSlide`` constructor option is accepted but
        // never read inside the bundle. Without an explicit render here
        // the canvas stays blank until something else (the
        // ResizeObserver below, prev/next click) triggers one. Fresh
        // opens where the container was already the right size show no
        // ResizeObserver fire and the user sees an empty canvas.
        // Explicitly render slide 0 right after the load, then clear
        // isLoading.
        await viewer.loadFile(bytes)
        if (cancelled) {
          viewer.destroy()
          return
        }
        await viewer.render(canvasRef.current, { slideIndex: 0 })
        if (cancelled) {
          viewer.destroy()
          return
        }

        createdViewer = viewer
        viewerRef.current = viewer
        setSlideCount(viewer.getSlideCount())
        setCurrentSlide(viewer.getCurrentSlideIndex())
      } catch (e) {
        if (!cancelled) {
          console.error("pptxviewjs render error", e)
          setError(t("files.previewDialog.errors.pptxRenderFailed"))
        }
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    load()

    return () => {
      cancelled = true
      createdViewer?.destroy()
      if (viewerRef.current === createdViewer) {
        viewerRef.current = null
      }
    }
  }, [base64Content, syncCanvasSize, t, pdfUrl, pdfChecked, fileId])

  // Re-render the current slide when the container is resized.
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    let rafId: number | null = null
    const schedule = () => {
      if (rafId !== null) return
      rafId = window.requestAnimationFrame(() => {
        rafId = null
        const changed = syncCanvasSize()
        const viewer = viewerRef.current
        if (changed && viewer) {
          viewer.render().catch(() => {
            /* ignore intermediate render races */
          })
        }
      })
    }

    const observer = new ResizeObserver(schedule)
    observer.observe(container)
    return () => {
      observer.disconnect()
      if (rafId !== null) window.cancelAnimationFrame(rafId)
    }
  }, [syncCanvasSize])

  const goPrev = useCallback(() => {
    viewerRef.current?.previousSlide().catch(() => undefined)
  }, [])
  const goNext = useCallback(() => {
    viewerRef.current?.nextSlide().catch(() => undefined)
  }, [])

  if (error) {
    return <div className="p-4 text-sm text-muted-foreground">{error}</div>
  }

  // High-fidelity path: server-rendered PDF in an iframe. Browsers (Chrome,
  // Safari, Firefox) all ship a built-in PDF viewer, so no JS library is
  // needed. The viewer comes with its own pagination + zoom controls; we
  // skip our prev/next bar so we don't have two competing controls.
  if (pdfUrl) {
    return (
      <div className="flex flex-col h-full bg-muted/30">
        <iframe
          src={pdfUrl}
          title="pptx preview (server-rendered PDF)"
          className="flex-1 w-full border-0"
        />
      </div>
    )
  }

  const hasNav = slideCount > 1

  return (
    <div className="flex flex-col h-full bg-muted/30">
      <div
        ref={containerRef}
        className="flex-1 relative overflow-hidden flex items-center justify-center"
      >
        {/*
          Canvas is sized in JS by syncCanvasSize (physical = cssSize *
          devicePixelRatio). Don't override width/height here.
        */}
        <canvas ref={canvasRef} className="block" />
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none bg-background/40">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
          </div>
        )}
      </div>
      {hasNav && (
        <div className="flex items-center justify-center gap-3 py-2 border-t bg-background/80 flex-shrink-0">
          <button
            type="button"
            onClick={goPrev}
            disabled={currentSlide <= 0}
            className="p-1 rounded hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label="Previous slide"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="text-xs tabular-nums text-muted-foreground min-w-[60px] text-center">
            {`${currentSlide + 1} / ${slideCount}`}
          </span>
          <button
            type="button"
            onClick={goNext}
            disabled={currentSlide >= slideCount - 1}
            className="p-1 rounded hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label="Next slide"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  )
}
