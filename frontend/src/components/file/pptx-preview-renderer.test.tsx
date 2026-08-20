/// <reference types="@testing-library/jest-dom/vitest" />

import React from "react"
import { cleanup, render, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

vi.mock("pptxviewjs", () => ({
  PPTXViewer: class {
    loadFile = vi.fn().mockResolvedValue(undefined)
    render = vi.fn().mockResolvedValue(undefined)
    nextSlide = vi.fn().mockResolvedValue(undefined)
    previousSlide = vi.fn().mockResolvedValue(undefined)
    goToSlide = vi.fn().mockResolvedValue(undefined)
    getSlideCount = vi.fn().mockReturnValue(1)
    getCurrentSlideIndex = vi.fn().mockReturnValue(0)
    on = vi.fn()
    destroy = vi.fn()
  },
}))

import {
  FileAccessProvider,
  createPublicFileAccessPolicy,
} from "@/contexts/file-access-context"
import { PptxPreviewRenderer } from "./pptx-preview-renderer"

describe("PptxPreviewRenderer public file access", () => {
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it("skips the authenticated PDF route and fetches PPTX bytes through its provider policy", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      arrayBuffer: async () => new ArrayBuffer(3),
    })
    vi.stubGlobal("fetch", fetchMock)

    render(
      <FileAccessProvider policy={createPublicFileAccessPolicy("pptx-guest-token")}>
        <PptxPreviewRenderer fileId="presentation-id" />
      </FileAccessProvider>,
    )

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/files/public/preview/presentation-id?token=pptx-guest-token",
      expect.objectContaining({ credentials: "same-origin" }),
    )
    expect(fetchMock.mock.calls.every(([url]) => !String(url).includes("preview-pdf"))).toBe(true)
  })
})
