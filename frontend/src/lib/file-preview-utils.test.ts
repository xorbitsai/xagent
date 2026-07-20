import { describe, expect, it } from "vitest"

import { isTextPreviewFile } from "./file-preview-utils"

describe("isTextPreviewFile", () => {
  it("recognizes subtitle files even when the backend returns octet-stream", () => {
    expect(isTextPreviewFile("the_last_garden.srt", "application/octet-stream")).toBe(true)
    expect(isTextPreviewFile("captions.vtt", "application/octet-stream")).toBe(true)
  })

  it("keeps binary files on the base64 preview path", () => {
    expect(isTextPreviewFile("final-film.mp4", "application/octet-stream")).toBe(false)
    expect(isTextPreviewFile("slides.pptx", "application/octet-stream")).toBe(false)
  })

  it("uses an explicit text MIME type when the extension is unknown", () => {
    expect(isTextPreviewFile("README", "text/plain; charset=utf-8")).toBe(true)
  })

  it("handles a missing file name defensively", () => {
    expect(isTextPreviewFile(undefined, "application/octet-stream")).toBe(false)
    expect(isTextPreviewFile(undefined, "text/plain")).toBe(true)
  })
})
