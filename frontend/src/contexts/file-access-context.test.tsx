/// <reference types="@testing-library/jest-dom/vitest" />

import React from "react"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

import {
  FileAccessProvider,
  createPublicFileAccessPolicy,
  defaultFileAccessPolicy,
  useFileAccess,
} from "./file-access-context"

function FileAccessProbe({ id }: { id: string }) {
  const fileAccess = useFileAccess()

  return (
    <button
      type="button"
      data-testid={id}
      onClick={() => void fileAccess.request(fileAccess.previewUrl("file-id"))}
    >
      {fileAccess.inlinePreviewUrl("file-id")}
    </button>
  )
}

function DefaultFileAccessProbe() {
  const fileAccess = useFileAccess()
  return (
    <button
      type="button"
      data-testid="default"
      data-preview={fileAccess.previewUrl("file/id")}
      data-download={fileAccess.downloadUrl("file/id")}
      data-inline-preview={fileAccess.inlinePreviewUrl("file/id")}
      data-inline-download={fileAccess.inlineDownloadUrl("file/id")}
      data-relative={fileAccess.relativePreviewUrl("file/id", "slides/one.png")}
      data-pdf={fileAccess.pdfPreviewUrl?.("file/id")}
      onClick={() => void fileAccess.listFiles?.(" report ")}
    >
      default
    </button>
  )
}

function StreamingUrlProbe() {
  const fileAccess = useFileAccess()
  const [result, setResult] = React.useState<string>("pending")

  return (
    <button
      type="button"
      data-testid="streaming"
      data-result={result}
      onClick={() => {
        fileAccess
          .getStreamingUrl?.("file/id")
          .then((url) => setResult(url))
          .catch((error: Error) => setResult(`error:${error.message}`))
      }}
    >
      streaming
    </button>
  )
}

describe("FileAccessProvider", () => {
  afterEach(() => {
    cleanup()
    apiRequestMock.mockReset()
    vi.unstubAllGlobals()
  })

  it("preserves the authenticated default policy outside public providers", () => {
    apiRequestMock.mockResolvedValue(new Response())
    render(
      <FileAccessProvider>
        <DefaultFileAccessProbe />
      </FileAccessProvider>,
    )

    const probe = screen.getByTestId("default")
    expect(probe).toHaveAttribute("data-preview", "/api/files/preview/file%2Fid")
    expect(probe).toHaveAttribute("data-download", "/api/files/download/file%2Fid")
    expect(probe).toHaveAttribute("data-inline-preview", "/api/files/public/preview/file%2Fid")
    expect(probe).toHaveAttribute("data-inline-download", "/api/files/public/download/file%2Fid")
    expect(probe.getAttribute("data-relative")).toContain(
      "/api/files/public/preview/file%2Fid?relative_path=slides%2Fone.png",
    )
    expect(probe).toHaveAttribute("data-pdf", "/api/files/preview-pdf/file%2Fid")

    probe.click()
    expect(apiRequestMock).toHaveBeenCalledWith(
      "/api/files/list?page=1&size=20&search=report",
    )
  })

  it("mints a streaming URL from the ticket endpoint's response path", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        path: "/api/files/preview/file%2Fid?ticket=signed-ticket",
      }),
    })
    render(
      <FileAccessProvider>
        <StreamingUrlProbe />
      </FileAccessProvider>,
    )

    screen.getByTestId("streaming").click()

    await waitFor(() => {
      expect(screen.getByTestId("streaming")).toHaveAttribute(
        "data-result",
        "/api/files/preview/file%2Fid?ticket=signed-ticket",
      )
    })
    expect(apiRequestMock).toHaveBeenCalledWith(
      "/api/files/stream-tickets/file%2Fid",
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    )
  })

  it("rejects the streaming URL promise when ticket minting fails", async () => {
    apiRequestMock.mockResolvedValue({ ok: false, status: 500 })
    render(
      <FileAccessProvider>
        <StreamingUrlProbe />
      </FileAccessProvider>,
    )

    screen.getByTestId("streaming").click()

    await waitFor(() => {
      expect(screen.getByTestId("streaming")).toHaveAttribute(
        "data-result",
        "error:Failed to mint stream ticket: 500",
      )
    })
  })

  it("rejects the streaming URL promise when the mint request itself throws", async () => {
    // Distinct from a resolved-but-not-ok response: a network-level
    // rejection (offline, DNS failure, an aborted request) must also
    // surface as a caught rejection so useResolvedMediaUrl's fallback
    // fires, not as an unhandled rejection.
    apiRequestMock.mockRejectedValue(new TypeError("Failed to fetch"))
    render(
      <FileAccessProvider>
        <StreamingUrlProbe />
      </FileAccessProvider>,
    )

    screen.getByTestId("streaming").click()

    await waitFor(() => {
      expect(screen.getByTestId("streaming")).toHaveAttribute(
        "data-result",
        "error:Failed to fetch",
      )
    })
  })

  it("rejects the streaming URL promise when the mint response has an unexpected shape", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => ({ notPath: "/api/files/preview/file-id" }),
    })
    render(
      <FileAccessProvider>
        <StreamingUrlProbe />
      </FileAccessProvider>,
    )

    screen.getByTestId("streaming").click()

    await waitFor(() => {
      expect(screen.getByTestId("streaming")).toHaveAttribute(
        "data-result",
        "error:Stream ticket response has an unexpected shape",
      )
    })
  })

  it("accepts a mint response whose fileId segment the server encoded differently than encodeURIComponent would", async () => {
    // Regression test: the server encodes file_id via Python's
    // urllib.parse.quote(file_id, safe="") when building the ticket path
    // (src/xagent/web/api/files.py), which escapes a different character
    // set than JS's encodeURIComponent -- "!*'()" round-trip unescaped
    // through encodeURIComponent but ARE escaped by quote(safe=""). A fix
    // that re-encodes fileId with encodeURIComponent and string-matches
    // against the response path would reject this genuinely valid
    // response for any fileId containing one of those characters.
    const fileId = "file!*'()id"
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        // Exactly what Python's quote(fileId, safe="") produces.
        path: "/api/files/preview/file%21%2A%27%28%29id?ticket=signed-ticket",
      }),
    })

    const url = await defaultFileAccessPolicy.getStreamingUrl?.(fileId)

    expect(url).toBe("/api/files/preview/file%21%2A%27%28%29id?ticket=signed-ticket")
  })

  it("rejects a mint response whose path is scoped to a different fileId", async () => {
    // Regression test (F5): a prefix-only check ("starts with
    // /api/files/preview/") would accept a path with extra segments after
    // it -- including one naming a completely different fileId, or a
    // dot-segment -- as long as it still started with the generic preview
    // prefix. The path must be scoped to the exact fileId requested.
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        path: "/api/files/preview/a-different-file-id?ticket=signed-ticket",
      }),
    })
    render(
      <FileAccessProvider>
        <StreamingUrlProbe />
      </FileAccessProvider>,
    )

    screen.getByTestId("streaming").click()

    await waitFor(() => {
      expect(screen.getByTestId("streaming")).toHaveAttribute(
        "data-result",
        "error:Stream ticket response has an unexpected shape",
      )
    })
  })

  it("does not expose getStreamingUrl on the public policy", () => {
    const policy = createPublicFileAccessPolicy("guest-token")
    expect(policy.getStreamingUrl).toBeUndefined()
  })

  it("keeps public file URLs and requests scoped to each provider", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response())
    vi.stubGlobal("fetch", fetchMock)

    render(
      <>
        <FileAccessProvider policy={createPublicFileAccessPolicy("first-token")}>
          <FileAccessProbe id="first" />
        </FileAccessProvider>
        <FileAccessProvider policy={createPublicFileAccessPolicy("second-token")}>
          <FileAccessProbe id="second" />
        </FileAccessProvider>
      </>,
    )

    expect(screen.getByTestId("first")).toHaveTextContent("token=first-token")
    expect(screen.getByTestId("first")).not.toHaveTextContent("second-token")
    expect(screen.getByTestId("second")).toHaveTextContent("token=second-token")
    expect(screen.getByTestId("second")).not.toHaveTextContent("first-token")
    const firstPolicy = createPublicFileAccessPolicy("first-token")
    expect(firstPolicy.downloadUrl("file/id")).toContain(
      "/api/files/public/download/file%2Fid?token=first-token",
    )
    expect(firstPolicy.relativePreviewUrl("file/id", "slides/one.png")).toContain(
      "token=first-token&relative_path=slides%2Fone.png",
    )

    screen.getByTestId("first").click()
    screen.getByTestId("second").click()

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining("token=first-token"),
      expect.objectContaining({ credentials: "same-origin" }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      expect.stringContaining("token=second-token"),
      expect.objectContaining({ credentials: "same-origin" }),
    )
  })

  it("keeps a surviving provider isolated when a neighboring provider unmounts", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response())
    vi.stubGlobal("fetch", fetchMock)
    const secondPolicy = createPublicFileAccessPolicy("surviving-token")
    const renderProviders = (showFirst: boolean) => (
      <>
        {showFirst ? (
          <FileAccessProvider policy={createPublicFileAccessPolicy("removed-token")}>
            <FileAccessProbe id="removed" />
          </FileAccessProvider>
        ) : null}
        <FileAccessProvider policy={secondPolicy}>
          <FileAccessProbe id="surviving" />
        </FileAccessProvider>
      </>
    )

    const { rerender } = render(renderProviders(true))
    rerender(renderProviders(false))

    expect(screen.queryByTestId("removed")).toBeNull()
    expect(screen.getByTestId("surviving")).toHaveTextContent("surviving-token")
    screen.getByTestId("surviving").click()

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("token=surviving-token"),
      expect.objectContaining({ credentials: "same-origin" }),
    )
    expect(sessionStorage.getItem("xagent_public_access_token")).toBeNull()
  })

  it("allows same-origin routing cookies but strips ambient Authorization", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response())
    vi.stubGlobal("fetch", fetchMock)
    const policy = createPublicFileAccessPolicy("guest-token")

    await policy.request(policy.previewUrl("file-id"), {
      headers: { Authorization: "Bearer ambient-user-token", "X-Trace": "request" },
      credentials: "include",
    })

    const [, options] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(options.credentials).toBe("same-origin")
    expect(new Headers(options.headers).get("Authorization")).toBeNull()
    expect(new Headers(options.headers).get("X-Trace")).toBe("request")
  })
})
