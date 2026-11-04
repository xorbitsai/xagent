declare module "pptxviewjs" {
  export type PPTXViewerConstructor = new (
    options: Record<string, unknown>
  ) => {
    loadFile(input: ArrayBuffer | Uint8Array | File): Promise<unknown>
    render(
      canvas?: HTMLCanvasElement | null,
      options?: { slideIndex?: number }
    ): Promise<unknown>
    nextSlide(): Promise<unknown>
    previousSlide(): Promise<unknown>
    goToSlide(index: number): Promise<unknown>
    getSlideCount(): number
    getCurrentSlideIndex(): number
    on(event: string, cb: (...args: unknown[]) => void): void
    destroy(): void
  }

  export const PPTXViewer: PPTXViewerConstructor | undefined

  const defaultExport: {
    PPTXViewer?: PPTXViewerConstructor
  }

  export default defaultExport
}
