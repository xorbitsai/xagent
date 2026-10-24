import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { apiRequest } from "@/lib/api-wrapper"
import {
  abortableSleep,
  capabilityRefusal,
  isCanonicalUuid,
  uploadWidgetSessionFile,
  WidgetSessionUploadError,
} from "@/lib/widget-session-upload"

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: vi.fn() }))

const ORIGIN = "https://chat.example.test"
const FILE_ID = "0b1e5c9e-3f1a-4a63-9b0e-6c1d2f3a4b5c"

const fetchMock = vi.fn()

const jsonResponse = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  })

const created = (overrides: Record<string, unknown> = {}) =>
  jsonResponse(201, {
    file_id: FILE_ID,
    filename: "report.pdf",
    file_size: 11,
    mime_type: "application/pdf",
    created_at: "2026-01-01T00:00:00Z",
    ...overrides,
  })

const rateLimited = (headers: Record<string, string> = {}) =>
  jsonResponse(429, { error: { code: "rate_limited", message: "slow down" } }, headers)

const makeFile = () => new File(["hello world"], "report.pdf", { type: "application/pdf" })

function baseOptions(overrides: Partial<Parameters<typeof uploadWidgetSessionFile>[0]> = {}) {
  const sleeps: number[] = []
  return {
    sleeps,
    options: {
      origin: ORIGIN,
      getToken: vi.fn(async () => "st_current"),
      refreshToken: vi.fn(async () => "st_refreshed"),
      file: makeFile(),
      sleep: vi.fn(async (ms: number) => { sleeps.push(ms) }),
      random: () => 0,
      ...overrides,
    },
  }
}

const expectUploadError = async (promise: Promise<unknown>, code: string) => {
  const error = await promise.then(() => null, (reason: unknown) => reason)
  expect(error).toBeInstanceOf(WidgetSessionUploadError)
  expect((error as WidgetSessionUploadError).errorCode).toBe(code)
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal("fetch", fetchMock)
  vi.mocked(apiRequest).mockReset()
  localStorage.clear()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe("capabilityRefusal", () => {
  const limits = { accept: ".pdf,.png", maxFileBytes: 3 }
  const file = (name: string, size = 3) => new File(["x".repeat(size)], name)

  it.each([
    ["a.pdf", 3, null],
    ["Report.PDF", 3, null],
    ["a.pdf", 4, "size"],
    ["a.exe", 1, "type"],
    ["a.exe", 4, "type"],
    ["pdf", 1, "type"],
    ["a.pdf.zip", 1, "type"],
  ])("judges %s of %i bytes as %s", (name, size, refusal) => {
    expect(capabilityRefusal(file(name, size), limits)).toBe(refusal)
  })
})

describe("uploadWidgetSessionFile", () => {
  it("retries once after a rate_limited response and then succeeds", async () => {
    fetchMock.mockResolvedValueOnce(rateLimited()).mockResolvedValueOnce(created())
    const { options } = baseOptions()

    const result = await uploadWidgetSessionFile(options)

    expect(result).toEqual({
      kind: "uploaded",
      fileId: FILE_ID,
      name: "report.pdf",
      size: 11,
      type: "application/pdf",
    })
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("gives up with upload_rate_limited after exactly three retries", async () => {
    fetchMock.mockImplementation(async () => rateLimited())
    const { options, sleeps } = baseOptions()

    await expectUploadError(uploadWidgetSessionFile(options), "upload_rate_limited")

    expect(fetchMock).toHaveBeenCalledTimes(4)
    expect(sleeps).toHaveLength(3)
  })

  it("does not retry quota_exceeded", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(429, { error: { code: "quota_exceeded", message: "full" } }),
    )
    const { options } = baseOptions()

    await expectUploadError(uploadWidgetSessionFile(options), "upload_quota_exceeded")

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(options.sleep).not.toHaveBeenCalled()
  })

  describe("backoff", () => {
    it("waits at least Retry-After before the first retry", async () => {
      fetchMock
        .mockResolvedValueOnce(rateLimited({ "Retry-After": "5" }))
        .mockResolvedValueOnce(created())
      const { options, sleeps } = baseOptions()

      await uploadWidgetSessionFile(options)

      expect(sleeps).toEqual([5000])
    })

    it("grows exponentially from a short Retry-After and adds jitter", async () => {
      fetchMock.mockImplementation(async () => rateLimited({ "Retry-After": "2" }))
      const { options, sleeps } = baseOptions({ random: () => 0.5 })

      await expectUploadError(uploadWidgetSessionFile(options), "upload_rate_limited")

      expect(sleeps).toEqual([2250, 4250, 8250])
    })

    it("caps a large Retry-After at sixty seconds plus jitter", async () => {
      fetchMock
        .mockResolvedValueOnce(rateLimited({ "Retry-After": "600" }))
        .mockResolvedValueOnce(created())
      const { options, sleeps } = baseOptions({ random: () => 0.999 })

      await uploadWidgetSessionFile(options)

      expect(sleeps).toEqual([60_499])
    })

    it.each([
      ["missing", {}],
      ["not a number", { "Retry-After": "soon" }],
      ["zero", { "Retry-After": "0" }],
    ])("falls back to 2^attempt seconds when Retry-After is %s", async (_label, headers) => {
      fetchMock.mockImplementation(async () => rateLimited(headers))
      const { options, sleeps } = baseOptions()

      await expectUploadError(uploadWidgetSessionFile(options), "upload_rate_limited")

      expect(sleeps).toEqual([2000, 4000, 8000])
    })

    describe("with an HTTP-date Retry-After", () => {
      beforeEach(() => {
        vi.useFakeTimers({ toFake: ["Date"] })
        vi.setSystemTime(new Date("2026-07-27T00:00:00.700Z"))
      })

      it("waits until the date, rounded up to a whole second", async () => {
        fetchMock
          .mockResolvedValueOnce(rateLimited({ "Retry-After": "Mon, 27 Jul 2026 00:00:05 GMT" }))
          .mockResolvedValueOnce(created())
        const { options, sleeps } = baseOptions()

        await uploadWidgetSessionFile(options)

        expect(sleeps).toEqual([5000])
      })

      it("caps a far date at sixty seconds plus jitter", async () => {
        fetchMock
          .mockResolvedValueOnce(rateLimited({ "Retry-After": "Mon, 27 Jul 2026 01:00:00 GMT" }))
          .mockResolvedValueOnce(created())
        const { options, sleeps } = baseOptions({ random: () => 0.999 })

        await uploadWidgetSessionFile(options)

        expect(sleeps).toEqual([60_499])
      })

      it("falls back to 2^attempt seconds for a date already past", async () => {
        fetchMock.mockImplementation(async () =>
          rateLimited({ "Retry-After": "Sun, 26 Jul 2026 23:59:00 GMT" }))
        const { options, sleeps } = baseOptions()

        await expectUploadError(uploadWidgetSessionFile(options), "upload_rate_limited")

        expect(sleeps).toEqual([2000, 4000, 8000])
      })
    })

    it("asks for a fresh token before every request, including retries", async () => {
      fetchMock.mockResolvedValueOnce(rateLimited()).mockResolvedValueOnce(created())
      const tokens = ["st_one", "st_two"]
      const { options } = baseOptions({ getToken: vi.fn(async () => tokens.shift() as string) })

      await uploadWidgetSessionFile(options)

      expect(fetchMock.mock.calls.map((call) => call[1].headers.Authorization)).toEqual([
        "Bearer st_one",
        "Bearer st_two",
      ])
    })
  })

  describe("credentials", () => {
    it("sends only the session token, omits cookies and posts a single file field", async () => {
      localStorage.setItem("token", "main-site-token")
      fetchMock.mockResolvedValueOnce(created())
      const { options } = baseOptions()

      await uploadWidgetSessionFile(options)

      const [url, init] = fetchMock.mock.calls[0]
      expect(url).toBe(`${ORIGIN}/v1/external/chat/sessions/files`)
      expect(init.method).toBe("POST")
      expect(init.credentials).toBe("omit")
      expect(init.headers).toEqual({ Authorization: "Bearer st_current" })
      expect(JSON.stringify(init.headers)).not.toContain("main-site-token")
      const entries = Array.from((init.body as FormData).entries())
      expect(entries).toHaveLength(1)
      expect(entries[0][0]).toBe("file")
      expect((entries[0][1] as File).name).toBe("report.pdf")
      expect(apiRequest).not.toHaveBeenCalled()
    })
  })

  describe("failure classification", () => {
    it("maps 413 to upload_too_large", async () => {
      fetchMock.mockResolvedValueOnce(jsonResponse(413, { error: { code: "invalid_input", message: "big" } }))
      const { options } = baseOptions()
      await expectUploadError(uploadWidgetSessionFile(options), "upload_too_large")
      expect(fetchMock).toHaveBeenCalledTimes(1)
    })

    it("maps 422 to upload_rejected", async () => {
      fetchMock.mockResolvedValueOnce(jsonResponse(422, { error: { code: "invalid_input", message: "type" } }))
      const { options } = baseOptions()
      await expectUploadError(uploadWidgetSessionFile(options), "upload_rejected")
      expect(fetchMock).toHaveBeenCalledTimes(1)
    })

    it.each([
      ["400", () => jsonResponse(400, { error: { code: "invalid_input", message: "bad" } })],
      ["404 without an error envelope", () => jsonResponse(404, { detail: "Not Found" })],
      ["500", () => jsonResponse(500, { error: { code: "internal_error", message: "x" } })],
      ["429 with an unknown code", () => jsonResponse(429, { error: { code: "other", message: "x" } })],
      ["429 without a JSON body", () => new Response("<html></html>", { status: 429 })],
      ["a 502 error page", () => new Response("<html></html>", { status: 502 })],
      ["a 201 with a non-JSON body", () => new Response("created", { status: 201 })],
      ["a 201 with a non-object body", () => jsonResponse(201, "ok")],
      ["a 201 without file_id", () => created({ file_id: undefined })],
      ["a 201 with an uppercase UUID", () => created({ file_id: FILE_ID.toUpperCase() })],
      ["a 201 with a non-UUID file_id", () => created({ file_id: "file-1" })],
    ])("maps %s to upload_failed after a single request", async (_label, makeResponse) => {
      fetchMock.mockResolvedValueOnce(makeResponse())
      const { options } = baseOptions()
      await expectUploadError(uploadWidgetSessionFile(options), "upload_failed")
      expect(fetchMock).toHaveBeenCalledTimes(1)
    })

    it("maps a network error to upload_failed after a single request", async () => {
      fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"))
      const { options } = baseOptions()
      await expectUploadError(uploadWidgetSessionFile(options), "upload_failed")
      expect(fetchMock).toHaveBeenCalledTimes(1)
    })

    it("maps an unclassified token failure to upload_failed and keeps a coded one", async () => {
      const { options } = baseOptions({ getToken: vi.fn(async () => { throw new Error("boom") }) })
      await expectUploadError(uploadWidgetSessionFile(options), "upload_failed")

      const coded = baseOptions({
        getToken: vi.fn(async () => { throw new WidgetSessionUploadError("connection_not_ready") }),
      })
      await expectUploadError(uploadWidgetSessionFile(coded.options), "connection_not_ready")
      expect(fetchMock).not.toHaveBeenCalled()
    })

    it("falls back to the File's own metadata when the response omits it", async () => {
      fetchMock.mockResolvedValueOnce(created({ filename: 3, file_size: "x", mime_type: null }))
      const { options } = baseOptions()
      await expect(uploadWidgetSessionFile(options)).resolves.toMatchObject({
        name: "report.pdf",
        size: 11,
        type: "application/pdf",
      })
    })
  })

  describe("401 handling", () => {
    it("refreshes the token once and retries the same file", async () => {
      fetchMock
        .mockResolvedValueOnce(jsonResponse(401, { error: { code: "invalid_api_key", message: "no" } }))
        .mockResolvedValueOnce(created())
      const { options } = baseOptions()

      await expect(uploadWidgetSessionFile(options)).resolves.toMatchObject({ fileId: FILE_ID })

      expect(options.refreshToken).toHaveBeenCalledTimes(1)
      expect(fetchMock.mock.calls.map((call) => call[1].headers.Authorization)).toEqual([
        "Bearer st_current",
        "Bearer st_refreshed",
      ])
    })

    it("passes the rejected token to refreshToken", async () => {
      fetchMock
        .mockResolvedValueOnce(jsonResponse(401, { error: { code: "invalid_api_key", message: "no" } }))
        .mockResolvedValueOnce(created())
      const { options } = baseOptions()

      await uploadWidgetSessionFile(options)

      expect(options.refreshToken).toHaveBeenCalledWith("st_current")
    })

    it("fails with upload_failed after a second 401", async () => {
      fetchMock.mockImplementation(async () => jsonResponse(401, { error: { code: "invalid_api_key", message: "no" } }))
      const { options } = baseOptions()

      await expectUploadError(uploadWidgetSessionFile(options), "upload_failed")

      expect(fetchMock).toHaveBeenCalledTimes(2)
      expect(options.refreshToken).toHaveBeenCalledTimes(1)
    })
  })

  describe("abort", () => {
    it("stops waiting and never sends another request when aborted during a backoff", async () => {
      vi.useFakeTimers()
      fetchMock.mockImplementation(async () => rateLimited())
      const controller = new AbortController()
      const { options } = baseOptions({
        signal: controller.signal,
        sleep: abortableSleep,
      })

      const outcome = uploadWidgetSessionFile(options).then(() => null, (reason: unknown) => reason)
      await vi.advanceTimersByTimeAsync(100)
      expect(fetchMock).toHaveBeenCalledTimes(1)
      controller.abort()
      const error = await outcome

      expect((error as Error).name).toBe("AbortError")
      await vi.advanceTimersByTimeAsync(120_000)
      expect(fetchMock).toHaveBeenCalledTimes(1)
    })

    it("rejects with an AbortError when the request is aborted", async () => {
      const controller = new AbortController()
      fetchMock.mockImplementationOnce(async () => {
        controller.abort()
        throw new DOMException("aborted", "AbortError")
      })
      const { options } = baseOptions({ signal: controller.signal })

      const error = await uploadWidgetSessionFile(options).then(() => null, (reason: unknown) => reason)

      expect((error as Error).name).toBe("AbortError")
      expect(fetchMock).toHaveBeenCalledTimes(1)
    })

    it("does not send anything when already aborted", async () => {
      const controller = new AbortController()
      controller.abort()
      const { options } = baseOptions({ signal: controller.signal })

      const error = await uploadWidgetSessionFile(options).then(() => null, (reason: unknown) => reason)

      expect((error as Error).name).toBe("AbortError")
      expect(fetchMock).not.toHaveBeenCalled()
    })

    it("treats a non-AbortError thrown after an abort as an abort", async () => {
      const controller = new AbortController()
      fetchMock.mockImplementationOnce(async () => {
        controller.abort()
        throw new TypeError("network down")
      })
      const { options } = baseOptions({ signal: controller.signal })

      const error = await uploadWidgetSessionFile(options).then(() => null, (reason: unknown) => reason)

      expect((error as Error).name).toBe("AbortError")
    })

    it("abortableSleep resolves after the delay and rejects when already aborted", async () => {
      vi.useFakeTimers()
      const sleeping = abortableSleep(1000)
      await vi.advanceTimersByTimeAsync(1000)
      await expect(sleeping).resolves.toBeUndefined()

      const controller = new AbortController()
      controller.abort()
      await expect(abortableSleep(1000, controller.signal)).rejects.toMatchObject({ name: "AbortError" })
    })
  })

  it("returns a plain object and never writes onto the File", async () => {
    fetchMock.mockResolvedValueOnce(created())
    const { options } = baseOptions()
    const before = Object.keys(options.file)

    const result = await uploadWidgetSessionFile(options)

    expect(Object.keys(options.file)).toEqual(before)
    expect(options.file).not.toHaveProperty("file_id")
    expect(Object.getPrototypeOf(result)).toBe(Object.prototype)
  })
})

describe("isCanonicalUuid", () => {
  it("accepts only lowercase canonical UUIDs", () => {
    expect(isCanonicalUuid(FILE_ID)).toBe(true)
    expect(isCanonicalUuid(FILE_ID.toUpperCase())).toBe(false)
    expect(isCanonicalUuid(FILE_ID.replace(/-/g, ""))).toBe(false)
    expect(isCanonicalUuid(` ${FILE_ID}`)).toBe(false)
    expect(isCanonicalUuid(42)).toBe(false)
  })
})

describe("uploadWidgetSessionFile cancellation", () => {
  const settleWith = (promise: Promise<unknown>) =>
    promise.then(() => null, (reason: unknown) => reason)
  const pendingToken = () => {
    let resolve!: (value: string) => void
    let reject!: (reason: unknown) => void
    const promise = new Promise<string>((res, rej) => { resolve = res; reject = rej })
    return { promise, resolve, reject }
  }
  const replyWith = (status: number, json: () => Promise<unknown>) =>
    ({ status, headers: new Headers(), json }) as unknown as Response

  it("does not call a token callback or send anything when already aborted", async () => {
    const controller = new AbortController()
    controller.abort()
    const { options } = baseOptions({ signal: controller.signal })

    const error = await settleWith(uploadWidgetSessionFile(options))

    expect((error as Error).name).toBe("AbortError")
    expect(options.getToken).not.toHaveBeenCalled()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("stops waiting for the first token as soon as it is aborted", async () => {
    const token = pendingToken()
    const controller = new AbortController()
    const { options } = baseOptions({ signal: controller.signal, getToken: vi.fn(() => token.promise) })

    const outcome = settleWith(uploadWidgetSessionFile(options))
    await Promise.resolve()
    controller.abort()

    expect(((await outcome) as Error).name).toBe("AbortError")
    token.resolve("st_late")
    await Promise.resolve()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("stops waiting for the token refreshed after a 401", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(401, { error: { code: "invalid_api_key", message: "no" } }))
    const token = pendingToken()
    const controller = new AbortController()
    const { options } = baseOptions({ signal: controller.signal, refreshToken: vi.fn(() => token.promise) })

    const outcome = settleWith(uploadWidgetSessionFile(options))
    await vi.waitFor(() => expect(options.refreshToken).toHaveBeenCalledTimes(1))
    controller.abort()

    expect(((await outcome) as Error).name).toBe("AbortError")
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("stops waiting for the token asked for after a rate-limit backoff", async () => {
    fetchMock.mockResolvedValueOnce(rateLimited())
    const token = pendingToken()
    const controller = new AbortController()
    const getToken = vi.fn()
      .mockResolvedValueOnce("st_first")
      .mockImplementationOnce(() => token.promise)
    const { options } = baseOptions({ signal: controller.signal, getToken })

    const outcome = settleWith(uploadWidgetSessionFile(options))
    await vi.waitFor(() => expect(getToken).toHaveBeenCalledTimes(2))
    controller.abort()

    expect(((await outcome) as Error).name).toBe("AbortError")
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("aborts only the upload whose signal fired when two share one pending token", async () => {
    const shared = pendingToken()
    fetchMock.mockResolvedValueOnce(created())
    const first = new AbortController()
    const second = new AbortController()
    const a = baseOptions({ signal: first.signal, getToken: vi.fn(() => shared.promise) })
    const b = baseOptions({ signal: second.signal, getToken: vi.fn(() => shared.promise) })

    const outcomeA = settleWith(uploadWidgetSessionFile(a.options))
    const outcomeB = uploadWidgetSessionFile(b.options)
    await Promise.resolve()
    first.abort()
    expect(((await outcomeA) as Error).name).toBe("AbortError")

    shared.resolve("st_shared")
    await expect(outcomeB).resolves.toMatchObject({ fileId: FILE_ID })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("keeps the abort when the shared wait fails after the abort", async () => {
    const shared = pendingToken()
    const first = new AbortController()
    const second = new AbortController()
    const a = baseOptions({ signal: first.signal, getToken: vi.fn(() => shared.promise) })
    const b = baseOptions({ signal: second.signal, getToken: vi.fn(() => shared.promise) })

    const outcomeA = settleWith(uploadWidgetSessionFile(a.options))
    const outcomeB = settleWith(uploadWidgetSessionFile(b.options))
    await Promise.resolve()
    first.abort()
    shared.reject(new WidgetSessionUploadError("upload_failed"))

    expect(((await outcomeA) as Error).name).toBe("AbortError")
    await expectUploadError(Promise.resolve(outcomeB).then((e) => { throw e }), "upload_failed")
  })

  it("keeps the abort when a non-abort token error arrives after the signal fired", async () => {
    const token = pendingToken()
    const controller = new AbortController()
    const { options } = baseOptions({ signal: controller.signal, getToken: vi.fn(() => token.promise) })

    const outcome = settleWith(uploadWidgetSessionFile(options))
    await Promise.resolve()
    controller.abort()
    token.reject(new Error("boom"))

    expect(((await outcome) as Error).name).toBe("AbortError")
  })

  it("stops at once when the token callback aborts the signal and never returns", async () => {
    const controller = new AbortController()
    const { options } = baseOptions({
      signal: controller.signal,
      getToken: vi.fn(() => {
        controller.abort()
        return new Promise<string>(() => {})
      }),
    })

    const error = await settleWith(uploadWidgetSessionFile(options))

    expect((error as Error).name).toBe("AbortError")
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("handles a token promise that rejects after the provider aborted the signal", async () => {
    const controller = new AbortController()
    const token = pendingToken()
    const { options } = baseOptions({
      signal: controller.signal,
      // Not a vi.fn(): a mock attaches its own handler to the promise it returns
      // and would hide the unhandled rejection this test exists to catch.
      getToken: () => {
        controller.abort()
        return token.promise
      },
    })

    const error = await settleWith(uploadWidgetSessionFile(options))
    token.reject(new Error("shared wait timed out"))
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect((error as Error).name).toBe("AbortError")
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("removes its abort listener once the token arrives", async () => {
    fetchMock.mockResolvedValueOnce(created())
    const controller = new AbortController()
    const added = vi.spyOn(controller.signal, "addEventListener")
    const removed = vi.spyOn(controller.signal, "removeEventListener")
    const { options } = baseOptions({ signal: controller.signal })

    await uploadWidgetSessionFile(options)

    const tokenListeners = added.mock.calls.filter(([type]) => type === "abort")
    expect(tokenListeners.length).toBeGreaterThan(0)
    for (const [, handler] of tokenListeners) {
      expect(removed).toHaveBeenCalledWith("abort", handler)
    }
  })

  it.each([
    ["a 201", 201],
    ["a 429", 429],
  ])("reports an abort while reading the body of %s as an abort", async (_label, status) => {
    fetchMock.mockResolvedValueOnce(replyWith(status, () => Promise.reject(new DOMException("x", "AbortError"))))
    const { options } = baseOptions()
    const error = await settleWith(uploadWidgetSessionFile(options))
    expect((error as Error).name).toBe("AbortError")
  })

  it.each([
    ["a 201", 201],
    ["a 429", 429],
  ])("reports a read error after the signal fired for %s as an abort", async (_label, status) => {
    const controller = new AbortController()
    fetchMock.mockResolvedValueOnce(replyWith(status, () => {
      controller.abort()
      return Promise.reject(new TypeError("body stream broke"))
    }))
    const { options } = baseOptions({ signal: controller.signal })
    const error = await settleWith(uploadWidgetSessionFile(options))
    expect((error as Error).name).toBe("AbortError")
  })

  it("still treats an unreadable 201 or 429 body as a plain failure when not aborted", async () => {
    fetchMock.mockResolvedValueOnce(replyWith(201, () => Promise.reject(new TypeError("bad"))))
    await expectUploadError(uploadWidgetSessionFile(baseOptions().options), "upload_failed")
    fetchMock.mockResolvedValueOnce(replyWith(429, () => Promise.reject(new TypeError("bad"))))
    await expectUploadError(uploadWidgetSessionFile(baseOptions().options), "upload_failed")
  })
})
