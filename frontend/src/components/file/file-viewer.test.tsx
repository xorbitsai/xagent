/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
  }
})

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

import { FileViewer } from "./file-viewer"

describe("FileViewer HTML preview", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("rewrites file id image sources to public preview URLs", () => {
    render(
      <FileViewer
        fileName="gallery.html"
        fileId="html-file-id"
        content={'<img src="file:582e7b79-4de9-4905-b73b-7d5a70ad64fe">'}
        isLoading={false}
        error={null}
        viewMode="preview"
      />,
    )

    const iframe = screen.getByTitle("gallery.html")

    expect(iframe).toHaveAttribute(
      "srcdoc",
      '<img src="http://api.local/api/files/public/preview/582e7b79-4de9-4905-b73b-7d5a70ad64fe">',
    )
  })

  it("keeps public preview image sources usable in HTML previews", () => {
    render(
      <FileViewer
        fileName="gallery.html"
        fileId="html-file-id"
        content={
          '<img src="/api/files/public/preview/582e7b79-4de9-4905-b73b-7d5a70ad64fe">'
        }
        isLoading={false}
        error={null}
        viewMode="preview"
      />,
    )

    const iframe = screen.getByTitle("gallery.html")

    expect(iframe).toHaveAttribute(
      "srcdoc",
      '<img src="http://api.local/api/files/public/preview/582e7b79-4de9-4905-b73b-7d5a70ad64fe">',
    )
  })

  it("keeps relative HTML assets scoped to the previewed HTML file", () => {
    render(
      <FileViewer
        fileName="gallery.html"
        fileId="html-file-id"
        content={'<img src="./assets/image.png">'}
        isLoading={false}
        error={null}
        viewMode="preview"
      />,
    )

    const iframe = screen.getByTitle("gallery.html")

    expect(iframe).toHaveAttribute(
      "srcdoc",
      '<img src="http://api.local/api/files/public/preview/html-file-id?relative_path=.%2Fassets%2Fimage.png">',
    )
  })
})

describe("FileViewer video preview", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("renders video controls with a blob URL for video mime types", async () => {
    const view = render(
      <FileViewer
        fileName="generated"
        fileId="video-file-id"
        content="AAAAIGZ0eXA="
        mimeType="video/mp4"
        isLoading={false}
        error={null}
        viewMode="preview"
      />,
    )

    const video = screen.getByLabelText("generated")
    expect(video.tagName).toBe("VIDEO")
    expect(video).toHaveAttribute("controls")
    expect(video).toHaveAttribute("playsinline")
    await waitFor(() => expect(video).toHaveAttribute("src", "blob:mock-8"))
    const blobArg = vi.mocked(URL.createObjectURL).mock.calls[0]?.[0]
    expect(blobArg).toMatchObject({ size: 8, type: "video/mp4" })

    view.unmount()

    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:mock-8")
  })

  it("infers video mime type from generated mp4 filenames", async () => {
    render(
      <FileViewer
        fileName="generated_video_e0f58746.mp4"
        fileId="video-file-id"
        content="AAAAIGZ0eXA="
        mimeType="application/octet-stream"
        isLoading={false}
        error={null}
        viewMode="preview"
      />,
    )

    const video = screen.getByLabelText("generated_video_e0f58746.mp4")
    expect(video.tagName).toBe("VIDEO")
    await waitFor(() => expect(video).toHaveAttribute("src", "blob:mock-8"))
    expect(screen.queryByText("AAAAIGZ0eXA=")).not.toBeInTheDocument()
  })
})
