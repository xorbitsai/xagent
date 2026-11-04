import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { WidgetSession } from "./use-widget-session"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

const toastError = vi.hoisted(() => vi.fn())
vi.mock("@/components/ui/sonner", () => ({ toast: { error: toastError } }))

import { useSessionAttachments, type SessionAttachmentsOptions } from "./use-session-attachments"

const FILE_A = "0b1e5c9e-3f1a-4a63-9b0e-6c1d2f3a4b5c"
const FILE_B = "7d2f0c1a-8b3e-4c55-a1f0-2e9d4b6c8a10"
const attachments = { accept: ".pdf", maxFiles: 3, maxFileBytes: 1000 }
const session: WidgetSession = {
  token: "st_hook",
  tokenExpiresAt: "2099-01-01T00:00:00.000Z",
  absoluteExpiresAt: "2099-01-01T01:00:00.000Z",
  generation: 1,
  agent: { id: 1, name: "Agent", suggestedPrompts: [] },
  capabilities: {
    fileUploads: { allowedExtensions: [".pdf"], maxFileBytes: 1000, maxFilesPerMessage: 3 },
  },
}

const pdf = (name: string) => new File(["abc"], name, { type: "application/pdf" })
const created = (fileId: string) => new Response(JSON.stringify({
  file_id: fileId,
  filename: "a.pdf",
  file_size: 3,
  mime_type: "application/pdf",
  created_at: "2026-01-01T00:00:00Z",
}), { status: 201, headers: { "Content-Type": "application/json" } })
// An upload request that never answers and fails only when it is aborted.
const pendingUntilAborted = (_url: string, init: RequestInit) =>
  new Promise<Response>((_resolve, reject) => {
    init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
  })

describe("useSessionAttachments", () => {
  const fetchMock = vi.fn()
  const sendMessage = vi.fn()
  const options = (overrides: Partial<SessionAttachmentsOptions> = {}): SessionAttachmentsOptions => ({
    attachments,
    uploads: {
      ensureFreshSession: vi.fn(async () => session),
      refreshSession: vi.fn(async () => session),
      waitForOpenConnection: vi.fn(async () => undefined),
    },
    status: "active",
    conversationEstablished: false,
    sendMessage,
    sessionConversationState: "unbound",
    isSessionInteractionLocked: false,
    isConversationResetPending: false,
    ...overrides,
  })
  const mount = (initial: SessionAttachmentsOptions) =>
    renderHook((props: SessionAttachmentsOptions) => useSessionAttachments(props), { initialProps: initial })
  const signalOf = (call: number) => fetchMock.mock.calls[call][1].signal as AbortSignal

  beforeEach(() => {
    fetchMock.mockReset()
    sendMessage.mockReset()
    sendMessage.mockResolvedValue(undefined)
    toastError.mockReset()
    vi.stubGlobal("fetch", fetchMock)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("uploads the files in order and hands their ids to sendMessage", async () => {
    const { result } = mount(options())
    const [a, b] = [pdf("a.pdf"), pdf("b.pdf")]
    fetchMock.mockResolvedValueOnce(created(FILE_A)).mockResolvedValueOnce(created(FILE_B))

    await act(async () => { await result.current.sendWithAttachments("hello", { mode: "x" }, [a, b]) })

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(sendMessage).toHaveBeenCalledWith("hello", { mode: "x" }, [], {
      uploadedFiles: [
        expect.objectContaining({ fileId: FILE_A }),
        expect.objectContaining({ fileId: FILE_B }),
      ],
    })
    expect(result.current.statuses.size).toBe(0)
    expect(result.current.isBusy).toBe(false)
  })

  it("counts a send with files as busy until sendMessage settles", async () => {
    const { result } = mount(options())
    fetchMock.mockResolvedValueOnce(created(FILE_A))
    let accept!: () => void
    sendMessage.mockReturnValueOnce(new Promise<void>((resolve) => { accept = resolve }))
    let sending!: Promise<void>

    act(() => { sending = result.current.sendWithAttachments("hello", {}, [pdf("a.pdf")]) })
    await waitFor(() => expect(sendMessage).toHaveBeenCalled())
    expect(result.current.isBusy).toBe(true)
    await act(async () => { accept(); await sending })

    expect(result.current.isBusy).toBe(false)
  })

  it("keeps the start-page files a send did not carry", async () => {
    const { result, rerender } = mount(options())
    const [a, b] = [pdf("a.pdf"), pdf("b.pdf")]
    fetchMock.mockResolvedValueOnce(created(FILE_A))
    act(() => { result.current.onStartFilesChange([a, b]) })

    await act(async () => { await result.current.sendStartMessage("with a", [a]) })
    expect(result.current.startFiles).toEqual([b])

    rerender(options({ attachments: undefined }))
    await act(async () => { await result.current.sendStartMessage("text only", []) })
    expect(sendMessage).toHaveBeenLastCalledWith("text only", undefined, [])
    expect(result.current.startFiles).toEqual([b])
  })

  it("aborts the send and its upload when one of its files is removed", async () => {
    const { result } = mount(options())
    const [a, b] = [pdf("a.pdf"), pdf("b.pdf")]
    fetchMock.mockImplementationOnce(pendingUntilAborted)
    act(() => { result.current.onPanelFilesChange([a, b]) })
    let failure: unknown
    act(() => {
      void result.current.sendWithAttachments("hello", {}, [a, b])
        .then(() => null, (error: unknown) => { failure = error })
    })
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    act(() => { result.current.onPanelFilesChange([b]) })
    await waitFor(() => expect(failure).toBeDefined())

    expect(signalOf(0).aborted).toBe(true)
    expect(failure).toMatchObject({ name: "AbortError", notificationHandled: true })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(sendMessage).not.toHaveBeenCalled()
    expect(result.current.panelFiles).toEqual([b])
    expect(result.current.isBusy).toBe(false)
  })

  it("keeps both selections while the session is unavailable and puts an uploading file back to not uploaded", async () => {
    const { result, rerender } = mount(options({ conversationEstablished: true }))
    const [a, b] = [pdf("a.pdf"), pdf("b.pdf")]
    fetchMock.mockImplementationOnce(pendingUntilAborted)
    act(() => {
      result.current.onStartFilesChange([a])
      result.current.onPanelFilesChange([b])
    })
    act(() => { void result.current.retryAttachment(b) })
    await waitFor(() => expect(result.current.statuses.get(b)).toBe("uploading"))

    rerender(options({ conversationEstablished: true, status: "degraded" }))
    await waitFor(() => expect(result.current.isBusy).toBe(false))
    rerender(options({ conversationEstablished: true }))

    expect(signalOf(0).aborted).toBe(true)
    expect(result.current.statuses.get(b)).toBeUndefined()
    expect(result.current.startFiles).toEqual([a])
    expect(result.current.panelFiles).toEqual([b])
    expect(toastError).not.toHaveBeenCalled()
  })

  it("releases the panel's files once the conversation returns to the start page", async () => {
    const { result, rerender } = mount(options({ conversationEstablished: true }))
    const [a, b] = [pdf("a.pdf"), pdf("b.pdf")]
    fetchMock.mockImplementationOnce(pendingUntilAborted)
    act(() => {
      result.current.onStartFilesChange([a])
      result.current.onPanelFilesChange([b])
    })
    act(() => { void result.current.retryAttachment(b) })
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    rerender(options({ conversationEstablished: false }))

    await waitFor(() => expect(result.current.panelFiles).toEqual([]))
    expect(signalOf(0).aborted).toBe(true)
    expect(result.current.statuses.get(b)).toBeUndefined()
    expect(result.current.startFiles).toEqual([a])
  })
})
