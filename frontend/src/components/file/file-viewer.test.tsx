/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { renderToString } from "react-dom/server"
import { act, cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { createPublicFileAccessPolicy, FileAccessProvider } from "@/contexts/file-access-context"

const apiConfig = vi.hoisted(() => ({ origin: "http://api.local" }))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => apiConfig.origin,
  }
})

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

import { FileViewer } from "./file-viewer"

describe("FileViewer HTML preview", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN", "")
  })
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    vi.restoreAllMocks()
    vi.unstubAllEnvs()
    apiConfig.origin = "http://api.local"
  })

  it("delivers absolute app-origin assets to the runner with an empty API URL", () => {
    apiConfig.origin = ""
    vi.stubEnv("NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN", "https://preview.example.test")
    const { rerender } = render(<FileViewer fileName="gallery.html" fileId="html-id"
      content={'<img src="file:image-id"><link href="./style.css"><script src="./app.js"></script>'}
      isLoading={false} error={null} viewMode="preview" />)
    const frame = screen.getByTitle("gallery.html") as HTMLIFrameElement
    const post = vi.spyOn(frame.contentWindow!, "postMessage")
    act(() => window.dispatchEvent(new MessageEvent("message", {
      source: frame.contentWindow, origin: "null", data: { type: "xagent-html-preview-ready", version: 1 },
    })))
    const html = post.mock.calls[0][0].html as string
    expect(html).toContain(`${window.location.origin}/api/files/public/preview/image-id`)
    expect(html).toContain(`${window.location.origin}/api/files/public/preview/html-id?relative_path=.%2Fstyle.css`)
    expect(html).toContain(`${window.location.origin}/api/files/public/preview/html-id?relative_path=.%2Fapp.js`)
    rerender(<FileViewer fileName="next.html" fileId="next-id" content=""
      isLoading={false} error={null} viewMode="preview" />)
    expect(screen.queryByTitle("next.html")).toBeNull()
  })

  it("signs public managed-file assets with a file ticket, never the guest token", async () => {
    apiConfig.origin = ""
    vi.stubEnv("NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN", "https://preview.example.test")
    const fetchMock = vi.fn().mockImplementation(async (url: string) => new Response(
      JSON.stringify({ ticket: `ticket-for-${url.split("/").pop()}`, expires_in: 600 }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ))
    vi.stubGlobal("fetch", fetchMock)
    render(<FileAccessProvider policy={createPublicFileAccessPolicy("guest-token")}>
      <FileViewer fileName="gallery.html" fileId="html-id"
        content={'<img src="file:image-id"><link href="./style.css">'}
        isLoading={false} error={null} viewMode="preview" />
    </FileAccessProvider>)
    // Signing is asynchronous: wait for the frame that carries the signed
    // document, then complete its ready handshake.
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    let html = ""
    await waitFor(() => {
      const frame = screen.getByTitle("gallery.html") as HTMLIFrameElement
      const post = vi.spyOn(frame.contentWindow!, "postMessage")
      act(() => window.dispatchEvent(new MessageEvent("message", {
        source: frame.contentWindow, origin: "null", data: { type: "xagent-html-preview-ready", version: 1 },
      })))
      expect(post).toHaveBeenCalled()
      html = post.mock.calls[0][0].html as string
    })
    expect(html).toContain(
      `${window.location.origin}/api/files/public/preview/image-id?ticket=ticket-for-image-id`,
    )
    expect(html).toContain(
      `${window.location.origin}/api/files/public/preview/html-id?relative_path=.%2Fstyle.css&amp;ticket=ticket-for-html-id`,
    )
    expect(html).not.toContain("guest-token")
    const mints = fetchMock.mock.calls as Array<[string, RequestInit]>
    expect(mints.map(([url]) => url).sort()).toEqual([
      "/api/files/public/tickets/html-id",
      "/api/files/public/tickets/image-id",
    ])
    for (const [, options] of mints) {
      expect(new Headers(options.headers).get("Authorization")).toBe("Bearer guest-token")
    }
    vi.unstubAllGlobals()
  })

  it("declares the scripts-only opaque-origin sandbox contract", () => {
    render(
      <FileViewer
        fileName="untrusted.html"
        fileId="untrusted-file"
        content={'<p>Untrusted document</p>'}
        isLoading={false}
        error={null}
        viewMode="preview"
      />,
    )
    const iframe = screen.getByTitle("untrusted.html")
    expect(iframe).toHaveAttribute("sandbox", "allow-scripts")
    expect(iframe).toHaveAttribute("referrerpolicy", "no-referrer")
  })

  it("does not parse a browser document during server rendering", () => {
    const parser = vi.spyOn(globalThis, "DOMParser").mockImplementation(() => {
      throw new Error("DOMParser is browser-only")
    })
    try {
      expect(() => renderToString(<FileViewer fileName="page.html" fileId="html-id"
        content="<p>Preview</p>" isLoading={false} error={null} viewMode="preview" />)).not.toThrow()
      expect(parser).not.toHaveBeenCalled()
    } finally {
      parser.mockRestore()
    }
  })

  it("prepares untrusted HTML without creating a resource-loading DOM", () => {
    const parser = vi.spyOn(globalThis, "DOMParser").mockImplementation(() => {
      throw new Error("Do not materialize untrusted HTML in the app origin")
    })
    try {
      render(<FileViewer fileName="page.html" fileId="html-id"
        content={'<html lang="zh"><head><script src="https://example.com/app.js"></script></head><body><img src="https://example.com/pixel"></body></html>'}
        isLoading={false} error={null} viewMode="preview" />)
      const source = screen.getByTitle("page.html").getAttribute("srcdoc")!
      expect(source).toContain('<html lang="zh">')
      expect(source.indexOf('content="no-referrer"')).toBeLessThan(source.indexOf("<script"))
      expect(parser).not.toHaveBeenCalled()
    } finally {
      parser.mockRestore()
    }
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

    expect(iframe.getAttribute("srcdoc")).toContain(
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

    expect(iframe.getAttribute("srcdoc")).toContain(
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

    expect(iframe.getAttribute("srcdoc")).toContain(
      '<img src="http://api.local/api/files/public/preview/html-file-id?relative_path=.%2Fassets%2Fimage.png">',
    )
  })

  it("puts a single no-referrer policy before document subresources", () => {
    render(<FileViewer fileName="page.html" fileId="html-file-id"
      content={'<!doctype html><html><head><meta name="referrer" content="unsafe-url"><script src="https://example.com/script.js"></script></head><body><img src="https://example.com/image.png"></body></html>'}
      isLoading={false} error={null} viewMode="preview" />)
    const source = screen.getByTitle("page.html").getAttribute("srcdoc")!
    const document = new DOMParser().parseFromString(source, "text/html")
    expect(document.querySelectorAll('meta[name="referrer"]')).toHaveLength(1)
    expect(document.head.firstElementChild?.outerHTML).toBe('<meta name="referrer" content="no-referrer">')
    expect(source.indexOf('content="no-referrer"')).toBeLessThan(source.indexOf("<script"))
  })
})

describe("FileViewer Markdown preview", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("passes files-disabled presentation constraints into Markdown previews", () => {
    render(
      <FileViewer
        fileName="report.md"
        fileId="markdown-file-id"
        content="[private report](file:tenant-secret)"
        isLoading={false}
        error={null}
        viewMode="preview"
        filesDisabled
      />,
    )

    expect(screen.getByText("private report")).not.toHaveAttribute("href")
    expect(screen.queryByRole("link")).not.toBeInTheDocument()
  })
})

describe("FileViewer PDF preview", () => {
  afterEach(() => { cleanup(); vi.clearAllMocks() })

  it("uses a revocable blob URL allowed by frame-src without a data fetch", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch")
    const view = render(<FileViewer fileName="report.pdf" fileId="pdf-id"
      content="JVBERg==" mimeType="application/pdf" isLoading={false}
      error={null} viewMode="preview" />)
    expect(screen.getByTitle("report.pdf")).toHaveAttribute("src", "blob:mock-4")
    expect(vi.mocked(URL.createObjectURL).mock.calls[0]?.[0]).toMatchObject({ size: 4, type: "application/pdf" })
    expect(fetchSpy).not.toHaveBeenCalled()
    fetchSpy.mockRestore()
    view.unmount()
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:mock-4")
  })

  it("decodes large previews in bounded chunks and yields between them", async () => {
    const decode = vi.spyOn(globalThis, "atob")
    try {
      render(<FileViewer fileName="large.pdf" fileId="pdf-id"
        content={"QUJD".repeat(32768)} mimeType="application/pdf" isLoading={false}
        error={null} viewMode="preview" />)
      expect(URL.createObjectURL).not.toHaveBeenCalled()
      await waitFor(() => expect(screen.getByTitle("large.pdf")).toHaveAttribute("src", "blob:mock-98304"))
      expect(Math.max(...decode.mock.calls.map(([value]) => value.length))).toBeLessThanOrEqual(65536)
    } finally { decode.mockRestore() }
  })

  it("does not publish an object URL after a chunked decode is cancelled", async () => {
    vi.useFakeTimers()
    try {
      const view = render(<FileViewer fileName="large.pdf" fileId="pdf-id"
        content={"QUJD".repeat(32768)} mimeType="application/pdf" isLoading={false}
        error={null} viewMode="preview" />)
      view.unmount()
      await vi.runAllTimersAsync()
      expect(URL.createObjectURL).not.toHaveBeenCalled()
    } finally { vi.useRealTimers() }
  })
})

describe("FileViewer video preview", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("renders video controls with a blob URL for video mime types", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch")
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
    expect(fetchSpy).not.toHaveBeenCalled()
    fetchSpy.mockRestore()

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

describe("FileViewer audio preview", () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it("renders audio controls for audio mime types", () => {
    render(
      <FileViewer
        fileName="speech.mp3"
        fileId="audio-file-id"
        content="SUQzZmFrZQ=="
        mimeType="audio/mpeg"
        isLoading={false}
        error={null}
        viewMode="preview"
      />,
    )

    const audio = screen.getByLabelText("speech.mp3")
    expect(audio.tagName).toBe("AUDIO")
    expect(audio).toHaveAttribute("controls")
    expect(audio).toHaveAttribute("src", "data:audio/mpeg;base64,SUQzZmFrZQ==")
  })

  it("infers mp3 audio from base64 content when mime and extension are missing", () => {
    render(
      <FileViewer
        fileName="generated-speech"
        fileId="audio-file-id"
        content="SUQzZmFrZQ=="
        mimeType="application/octet-stream"
        isLoading={false}
        error={null}
        viewMode="preview"
      />,
    )

    const audio = screen.getByLabelText("generated-speech")
    expect(audio.tagName).toBe("AUDIO")
    expect(audio).toHaveAttribute("src", "data:audio/mpeg;base64,SUQzZmFrZQ==")
  })

  it("decodes only a short base64 prefix while inferring audio", () => {
    const originalAtob = globalThis.atob.bind(globalThis)
    const atobSpy = vi.spyOn(globalThis, "atob").mockImplementation((value) => (
      originalAtob(value)
    ))
    const payload = `${"SUQz"}${"QUFB".repeat(10000)}`
    const content = `data:application/octet-stream;base64,${payload}`

    render(
      <FileViewer
        fileName="generated"
        fileId="large-audio-file-id"
        content={content}
        mimeType="application/octet-stream"
        isLoading={false}
        error={null}
        viewMode="preview"
      />,
    )

    expect(screen.getByLabelText("generated")).toHaveAttribute(
      "src",
      `data:audio/mpeg;base64,${payload}`,
    )
    expect(atobSpy).toHaveBeenCalledOnce()
    expect(atobSpy.mock.calls[0][0].length).toBeLessThanOrEqual(24)
  })

  it("does not infer wav audio from an incomplete RIFF header", () => {
    render(
      <FileViewer
        fileName="generated"
        fileId="short-header-file-id"
        content="UklGRg=="
        mimeType="application/octet-stream"
        isLoading={false}
        error={null}
        viewMode="preview"
      />,
    )

    expect(screen.queryByLabelText("generated")).not.toBeInTheDocument()
    expect(screen.getByText("UklGRg==")).toBeInTheDocument()
  })
})

describe("FileViewer spreadsheet preview", () => {
  afterEach(() => {
    cleanup()
  })

  it("decodes CSV code view from prefixed base64 content", () => {
    const { container } = render(
      <FileViewer
        fileName="data.csv"
        fileId="csv-file-id"
        content="data:text/csv;base64,YSxiCjEsMg=="
        mimeType="text/csv"
        isLoading={false}
        error={null}
        viewMode="code"
      />,
    )

    expect(container.querySelector("pre")?.textContent).toBe("a,b\n1,2")
  })
})
