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

  it("builds public file URLs that never carry the guest token", () => {
    const policy = createPublicFileAccessPolicy("guest-secret-token")
    const urls = [
      policy.previewUrl("file/id"),
      policy.downloadUrl("file/id"),
      policy.inlinePreviewUrl("file/id"),
      policy.inlineDownloadUrl("file/id"),
      policy.relativePreviewUrl("file/id", "slides/one.png"),
      policy.validationUrl?.("file/id") ?? "",
    ]
    for (const url of urls) {
      expect(url).not.toContain("guest-secret-token")
      expect(url).not.toMatch(/[?&]token=/)
    }
    expect(policy.downloadUrl("file/id")).toBe("/api/files/public/download/file%2Fid")
    expect(policy.relativePreviewUrl("file/id", "slides/one.png")).toBe(
      "/api/files/public/preview/file%2Fid?relative_path=slides%2Fone.png",
    )
    expect(policy.validationUrl?.("file/id")).toBe(
      "/api/files/public/preview/file%2Fid?validation_only=true",
    )
    // Media elements cannot send the header: they must not load previewUrl directly.
    expect(policy.requiresBlobFetch).toBe(true)
  })

  it("sends each provider's guest token only as an Authorization header", async () => {
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

    expect(screen.getByTestId("first")).not.toHaveTextContent("token")
    expect(screen.getByTestId("second")).not.toHaveTextContent("token")

    screen.getByTestId("first").click()
    screen.getByTestId("second").click()

    const calls = fetchMock.mock.calls as Array<[string, RequestInit]>
    expect(calls.map(([url]) => url)).toEqual([
      "/api/files/public/preview/file-id",
      "/api/files/public/preview/file-id",
    ])
    expect(new Headers(calls[0][1].headers).get("Authorization")).toBe("Bearer first-token")
    expect(new Headers(calls[1][1].headers).get("Authorization")).toBe("Bearer second-token")
    expect(calls.every(([, options]) => options.credentials === "same-origin")).toBe(true)
    expect(sessionStorage.getItem("xagent_public_access_token")).toBeNull()
  })

  it("replaces ambient Authorization and never sends the guest token elsewhere", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response())
    vi.stubGlobal("fetch", fetchMock)
    const policy = createPublicFileAccessPolicy("guest-token")

    await policy.request(policy.previewUrl("file-id"), {
      headers: { Authorization: "Bearer ambient-user-token", "X-Trace": "request" },
      credentials: "include",
    })
    await policy.request("https://elsewhere.example/api/files/public/preview/file-id", {
      headers: { Authorization: "Bearer ambient-user-token" },
    })
    await policy.request("/api/files/preview/file-id")

    const calls = fetchMock.mock.calls as Array<[string, RequestInit]>
    expect(calls[0][1].credentials).toBe("same-origin")
    expect(new Headers(calls[0][1].headers).get("Authorization")).toBe("Bearer guest-token")
    expect(new Headers(calls[0][1].headers).get("X-Trace")).toBe("request")
    expect(new Headers(calls[1][1].headers).get("Authorization")).toBeNull()
    expect(new Headers(calls[2][1].headers).get("Authorization")).toBeNull()
  })

  it("mints a file-bound ticket over the header for URLs the browser loads itself", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => new Response(
      JSON.stringify({ ticket: "ticket.abc-123", expires_in: 600 }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ))
    vi.stubGlobal("fetch", fetchMock)
    const policy = createPublicFileAccessPolicy("guest-token")

    const streamingUrl = await policy.getStreamingUrl?.("file/id")
    const assetUrl = await policy.authorizeBrowserUrl?.(
      policy.relativePreviewUrl("file/id", "img/a.png"),
    )

    expect(streamingUrl).toBe("/api/files/public/preview/file%2Fid?ticket=ticket.abc-123")
    expect(assetUrl).toBe(
      "/api/files/public/preview/file%2Fid?relative_path=img%2Fa.png&ticket=ticket.abc-123",
    )
    expect(streamingUrl).not.toContain("guest-token")
    // One mint per file while the ticket is fresh.
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [mintUrl, mintOptions] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(mintUrl).toBe("/api/files/public/tickets/file%2Fid")
    expect(mintOptions.method).toBe("POST")
    expect(new Headers(mintOptions.headers).get("Authorization")).toBe("Bearer guest-token")
  })

  it("refuses to sign foreign URLs or accept a malformed ticket response", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => new Response(
      JSON.stringify({ ticket: "bad ticket", expires_in: 600 }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ))
    vi.stubGlobal("fetch", fetchMock)
    const policy = createPublicFileAccessPolicy("guest-token")

    await expect(policy.authorizeBrowserUrl?.("https://elsewhere.example/x.png"))
      .rejects.toThrow("Not a public file URL")
    await expect(policy.authorizeBrowserUrl?.("/api/files/preview/file-id"))
      .rejects.toThrow("Not a public file URL")
    expect(fetchMock).not.toHaveBeenCalled()
    await expect(policy.getStreamingUrl?.("file-id"))
      .rejects.toThrow("File ticket response has an unexpected shape")
  })
  const ticketResponse = (ticket: string, expiresIn: number) => new Response(
    JSON.stringify({ ticket, expires_in: expiresIn }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  )

  it("mints once for a short server TTL instead of looping on a fresh ticket", async () => {
    vi.useFakeTimers()
    try {
      let minted = 0
      const fetchMock = vi.fn().mockImplementation(async () => ticketResponse(`t${++minted}`, 10))
      vi.stubGlobal("fetch", fetchMock)
      const policy = createPublicFileAccessPolicy("guest-token")

      // TTL (10s) is below the 30s refresh margin: the fresh ticket is used.
      await expect(policy.getStreamingUrl?.("file-id")).resolves.toContain("ticket=t1")
      expect(fetchMock).toHaveBeenCalledTimes(1)
      // Still reused within the clamped margin (half the lifetime).
      vi.advanceTimersByTime(4_000)
      await expect(policy.getStreamingUrl?.("file-id")).resolves.toContain("ticket=t1")
      expect(fetchMock).toHaveBeenCalledTimes(1)
      // Refreshed once past it -- exactly one more mint.
      vi.advanceTimersByTime(2_000)
      await expect(policy.getStreamingUrl?.("file-id")).resolves.toContain("ticket=t2")
      expect(fetchMock).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })

  it("refreshes a ticket 30s before a normal expiry and shares concurrent mints", async () => {
    vi.useFakeTimers()
    try {
      let minted = 0
      const fetchMock = vi.fn().mockImplementation(async () => ticketResponse(`t${++minted}`, 600))
      vi.stubGlobal("fetch", fetchMock)
      const policy = createPublicFileAccessPolicy("guest-token")

      const burst = await Promise.all([
        policy.getStreamingUrl?.("file-id"),
        policy.getStreamingUrl?.("file-id"),
      ])
      expect(burst.every((url) => url?.includes("ticket=t1"))).toBe(true)
      expect(fetchMock).toHaveBeenCalledTimes(1)

      vi.advanceTimersByTime(569_000)
      await expect(policy.getStreamingUrl?.("file-id")).resolves.toContain("ticket=t1")
      vi.advanceTimersByTime(2_000)
      await expect(policy.getStreamingUrl?.("file-id")).resolves.toContain("ticket=t2")
      expect(fetchMock).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })

  it("prunes stale settled tickets for other files but keeps pending mints", async () => {
    vi.useFakeTimers()
    const mapDelete = vi.spyOn(Map.prototype, "delete")
    try {
      let releasePending: ((response: Response) => void) | undefined
      const fetchMock = vi.fn().mockImplementation(async (url: string) => {
        if (url.endsWith("/tickets/file-a")) return ticketResponse("ta", 10)
        if (url.endsWith("/tickets/file-b")) return ticketResponse("tb", 600)
        return new Promise<Response>((resolve) => { releasePending = resolve })
      })
      vi.stubGlobal("fetch", fetchMock)
      const policy = createPublicFileAccessPolicy("guest-token")

      await expect(policy.getStreamingUrl?.("file-a")).resolves.toContain("ticket=ta")
      const pending = policy.getStreamingUrl?.("file-pending")
      // Past file-a's reuse window (half its 10s lifetime), inside the mint timeout.
      vi.advanceTimersByTime(6_000)

      await expect(policy.getStreamingUrl?.("file-b")).resolves.toContain("ticket=tb")
      const deleted = mapDelete.mock.calls.map(([key]) => key)
      expect(deleted).toContain("file-a")
      expect(deleted).not.toContain("file-pending")
      expect(fetchMock.mock.calls.filter(([url]) => String(url).endsWith("/tickets/file-b")))
        .toHaveLength(1)

      // The in-flight mint is still shared rather than restarted.
      const joined = policy.getStreamingUrl?.("file-pending")
      releasePending?.(ticketResponse("tp", 600))
      await expect(pending).resolves.toContain("ticket=tp")
      await expect(joined).resolves.toContain("ticket=tp")
      expect(fetchMock).toHaveBeenCalledTimes(3)
    } finally {
      mapDelete.mockRestore()
      vi.useRealTimers()
    }
  })

  it("evicts a failed ticket mint so the next load retries", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response("nope", { status: 503 }))
      .mockResolvedValueOnce(ticketResponse("fresh", 600))
    vi.stubGlobal("fetch", fetchMock)
    const policy = createPublicFileAccessPolicy("guest-token")

    await expect(policy.getStreamingUrl?.("file-id"))
      .rejects.toThrow("Failed to mint file ticket: 503")
    await expect(policy.getStreamingUrl?.("file-id")).resolves.toContain("ticket=fresh")
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("aborts a hung ticket mint after the timeout and evicts it", async () => {
    vi.useFakeTimers()
    try {
      const fetchMock = vi.fn()
        .mockImplementationOnce((_url: string, options: RequestInit) => new Promise((_, reject) => {
          options.signal?.addEventListener("abort", () => reject(new Error("aborted")))
        }))
        .mockImplementationOnce(async () => ticketResponse("after-timeout", 600))
      vi.stubGlobal("fetch", fetchMock)
      const policy = createPublicFileAccessPolicy("guest-token")

      const hung = policy.getStreamingUrl?.("file-id")
      const settled = expect(hung).rejects.toThrow("File ticket mint timed out")
      await vi.advanceTimersByTimeAsync(10_000)
      await settled
      await expect(policy.getStreamingUrl?.("file-id")).resolves.toContain("ticket=after-timeout")
      expect(fetchMock).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })

  it("keeps the mint deadline through a response body that stalls after headers", async () => {
    vi.useFakeTimers()
    try {
      let bodySignal: AbortSignal | undefined
      const stalledBody = {
        ok: true,
        status: 200,
        // Headers arrived; the body never finishes (and ignores the abort).
        json: () => new Promise(() => {}),
      } as unknown as Response
      const fetchMock = vi.fn()
        .mockImplementationOnce(async (_url: string, options: RequestInit) => {
          bodySignal = options.signal ?? undefined
          return stalledBody
        })
        .mockImplementationOnce(async () => ticketResponse("after-stall", 600))
      vi.stubGlobal("fetch", fetchMock)
      const policy = createPublicFileAccessPolicy("guest-token")

      const stalled = policy.authorizeBrowserUrl?.(policy.inlinePreviewUrl("file-id"))
      const settled = expect(stalled).rejects.toThrow("File ticket mint timed out")
      await vi.advanceTimersByTimeAsync(9_999)
      expect(bodySignal?.aborted).toBe(false)
      await vi.advanceTimersByTimeAsync(1)
      await settled
      // The real fetch's body read is cancelled too.
      expect(bodySignal?.aborted).toBe(true)
      // Evicted: the next load mints afresh instead of waiting on the stall.
      await expect(policy.authorizeBrowserUrl?.(policy.inlinePreviewUrl("file-id")))
        .resolves.toContain("ticket=after-stall")
      expect(fetchMock).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })
})
