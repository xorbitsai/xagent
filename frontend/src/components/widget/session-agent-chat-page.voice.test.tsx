import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useVoiceCallView } from "@/contexts/voice-call-store"
import { classifyWidgetVoiceEvent, type WidgetVoiceCallOptions } from "@/hooks/use-widget-voice-call"
import type { VoiceAudioTransport } from "@/lib/voice-audio"
import type { WidgetSession, WidgetSessionStatus } from "./use-widget-session"

const bridge = vi.hoisted(() => ({
  value: {
    status: "active" as string,
    session: null as unknown,
    agent: null as unknown,
    terminalCode: null as string | null,
    isAbsoluteExpiryWarningVisible: false,
    handleSessionExpired: vi.fn(),
    handleConnectionClose: () => "handled" as const,
    handleConnectionFailure: () => undefined,
    handleConnectionOpen: () => undefined,
    ensureFreshSession: vi.fn(),
    refreshSession: vi.fn(),
    waitForOpenConnection: vi.fn(),
  },
  rerender: null as null | React.Dispatch<React.SetStateAction<number>>,
}))
const app = vi.hoisted(() => ({
  state: {
    taskId: null as number | null,
    currentTask: null as null | { id: string; status: string; waitingRequestId?: string },
    isProcessing: false,
    messages: [],
  },
  isConnected: true,
  isSendInFlight: false,
  isConversationResetPending: false,
  isMessageDeliveryPending: false,
  isSessionInteractionLocked: false,
  sessionConversationState: "unbound" as string,
  canStopTask: false,
  stopTask: vi.fn(),
  sendMessage: vi.fn(),
}))
// The send handler the task panel was given, as the composer would call it.
const panel = vi.hoisted(() => ({
  onSend: null as null | ((text: string, config: Record<string, unknown>, files: File[]) => Promise<void>),
}))
const voice = vi.hoisted(() => ({
  props: null as null | WidgetVoiceCallOptions,
  transport: null as null | VoiceAudioTransport,
  callbacks: null as null | { transcript: (type: "partial" | "final", text: string, id?: number) => void },
  audios: [] as Array<{ close: ReturnType<typeof vi.fn> }>,
  // Runs the real VoiceAudio, so a test can see the microphone tracks stop.
  real: false,
}))

vi.mock("./use-widget-session", async () => ({
  ...await vi.importActual<typeof import("./use-widget-session")>("./use-widget-session"),
  useWidgetSession: () => {
    const [, rerender] = React.useState(0)
    bridge.rerender = rerender
    return bridge.value
  },
}))
vi.mock("@/contexts/app-context-chat", () => ({
  AppProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useApp: () => ({
    state: app.state,
    filesDisabled: true,
    voiceInputEnabled: false,
    isConnected: app.isConnected,
    isSendInFlight: app.isSendInFlight,
    canStopTask: app.canStopTask,
    stopTask: app.stopTask,
    sendMessage: app.sendMessage,
    startNewConversation: vi.fn(() => Promise.resolve()),
    isConversationResetPending: app.isConversationResetPending,
    isMessageDeliveryPending: app.isMessageDeliveryPending,
    sessionConversationState: app.sessionConversationState,
    sessionConversationAttachment: "attached",
    isSessionInteractionLocked: app.isSessionInteractionLocked,
  }),
}))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))
vi.mock("@/lib/voice-audio", async () => {
  const actual = await vi.importActual<typeof import("@/lib/voice-audio")>("@/lib/voice-audio")
  return {
  ...actual,
  createWidgetVoiceAudio: (transport: VoiceAudioTransport, callbacks: typeof voice.callbacks) => {
    voice.transport = transport
    voice.callbacks = callbacks
    if (voice.real) return actual.createWidgetVoiceAudio(transport, callbacks as Parameters<typeof actual.createWidgetVoiceAudio>[1])
    const audio = { start: vi.fn().mockResolvedValue(undefined), close: vi.fn(), stopAudio: vi.fn(), speak: vi.fn(), level: () => 0, muted: false }
    voice.audios.push(audio)
    return audio
  },
  }
})
vi.mock("@/lib/voice-vad", () => ({ createVoiceVad: vi.fn(async () => ({ process: vi.fn(), destroy: vi.fn().mockResolvedValue(undefined) })) }))
vi.mock("./widget-history-list", () => ({
  WidgetHistoryList: (props: { onOpen: (id: number, title: string) => void; onBack: () => void }) => (
    <div data-testid="history-list">
      <button type="button" onClick={() => props.onOpen(7, "Earlier chat")}>list-open</button>
      <button type="button" onClick={props.onBack}>list-back</button>
    </div>
  ),
}))
vi.mock("./widget-history-view", () => ({
  WidgetHistoryView: (props: { onContinue: (message: string, id: number) => void; onBack: () => void }) => (
    <div data-testid="history-view">
      <button type="button" onClick={() => props.onContinue("a follow-up", 7)}>view-continue</button>
      <button type="button" onClick={props.onBack}>view-back</button>
    </div>
  ),
}))
vi.mock("@/components/chat/widget-voice-conversation", async () => {
  const actual = await vi.importActual<typeof import("@/components/chat/widget-voice-conversation")>("@/components/chat/widget-voice-conversation")
  return {
    WidgetVoiceConversation: (props: WidgetVoiceCallOptions) => {
      voice.props = props
      return <actual.WidgetVoiceConversation {...props} />
    },
  }
})

type Opener = { onVoiceConversation?: () => void; onProvidedVoiceConversation?: () => void; voiceConversationSource?: string }
function Composer({ where, open }: { where: string; open?: () => void }) {
  const call = useVoiceCallView()
  return (
    <div data-testid={where}>
      {open && <button onClick={open}>voice:{where}</button>}
      <span data-testid="call">{call ? call.phase : "none"}</span>
      {call?.alert && <span data-testid="alert">{call.alert}</span>}
      {call?.note && <span data-testid="note">{call.note}</span>}
      {call?.retryable && <button disabled={call.retryDisabled} onClick={call.retry}>retry</button>}
      {call && <button onClick={call.end}>hang up</button>}
    </div>
  )
}
vi.mock("@/components/chat/ChatStartScreen", () => ({
  ChatStartScreen: (props: Opener) => (
    <Composer where="start" open={props.voiceConversationSource === "provided" ? props.onVoiceConversation : undefined} />
  ),
}))
vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: (props: Opener & { onSend?: typeof panel.onSend }) => {
    panel.onSend = props.onSend ?? null
    return <Composer where="panel" open={props.onProvidedVoiceConversation} />
  },
}))

import { SessionAgentChatPage } from "./session-agent-chat-page"

const TOKEN = "st_voice_session_token"
const agent = { id: 42, name: "Support Agent", suggestedPrompts: [] }
const session = (capabilities: unknown, token = TOKEN): WidgetSession => ({
  token, tokenExpiresAt: "2026-11-14T01:00:00.000Z", absoluteExpiresAt: "2026-11-14T02:00:00.000Z",
  generation: 1, agent, ...(capabilities === undefined ? {} : { capabilities: capabilities as WidgetSession["capabilities"] }),
})
function setBridge(status: WidgetSessionStatus, next: WidgetSession | null, terminalCode: string | null = null) {
  bridge.value = { ...bridge.value, status, session: next, terminalCode, agent: status === "terminal" ? null : agent }
}
const rerender = () => act(() => { bridge.rerender?.(value => value + 1) })
const voiceRequests = () => fetchMock.mock.calls.map(([url]) => String(url)).filter(url => url.includes("/voice/"))
const fetchMock = vi.fn()
const authorizations = () => fetchMock.mock.calls.map(([, init]) => init.headers.Authorization)
const sockets: Array<[string, string[]]> = []
const socketInstances: FakeSocket[] = []
class FakeSocket {
  static OPEN = 1
  readyState = 1
  bufferedAmount = 0
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: ((event: { code: number; reason: string }) => void) | null = null
  onerror: (() => void) | null = null
  closeListeners: Array<(event: { code: number; reason: string }) => void> = []
  addEventListener(type: string, listener: (event: { code: number; reason: string }) => void) {
    if (type === "close") this.closeListeners.push(listener)
  }
  // Listeners run in the order they were added, then the onclose property, as a browser does.
  serverClose(code: number, reason = "") {
    const event = { code, reason }
    this.closeListeners.forEach(listener => listener(event))
    this.onclose?.(event)
  }
  send = vi.fn()
  close = vi.fn()
  constructor(url: string, protocols: string[]) { sockets.push([url, protocols]); socketInstances.push(this) }
}
const originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, "mediaDevices")
const status = (code: number) => new Response(null, { status: code })
const ttsUrl = `${window.location.origin}/v1/external/chat/sessions/voice/tts`

// jsdom has no window.isSecureContext, which pageMayUseMicrophone() reads.
const secureContextDescriptor = Object.getOwnPropertyDescriptor(window, "isSecureContext")
function setSecureContext(value: boolean) {
  Object.defineProperty(window, "isSecureContext", { configurable: true, value })
}

function renderPage(options: { capabilities?: unknown; voiceParam?: boolean } = {}) {
  const { voiceParam = true } = options
  window.history.replaceState({}, "", voiceParam ? "/widget/session?voice=1" : "/widget/session")
  setBridge("active", session("capabilities" in options ? options.capabilities : { voice: true }))
  render(<SessionAgentChatPage />)
}
async function openCall(where: "start" | "panel" = "start") {
  fireEvent.click(screen.getByText(`voice:${where}`))
  await waitFor(() => expect(screen.getByTestId("call")).toHaveTextContent("listening"))
  return voice.transport!
}

beforeEach(() => {
  app.state = { taskId: null, currentTask: null, isProcessing: false, messages: [] }
  app.isConnected = true
  app.isSendInFlight = false
  app.isConversationResetPending = false
  app.isMessageDeliveryPending = false
  app.isSessionInteractionLocked = false
  app.sessionConversationState = "unbound"
  app.canStopTask = false
  app.stopTask.mockReset()
  app.sendMessage.mockReset().mockResolvedValue(undefined)
  bridge.value.refreshSession.mockReset()
  bridge.value.ensureFreshSession.mockReset()
  bridge.value.waitForOpenConnection.mockReset().mockResolvedValue(undefined)
  bridge.value.handleSessionExpired.mockReset()
  voice.props = null; voice.transport = null; voice.audios = []; voice.real = false
  sockets.length = 0
  socketInstances.length = 0
  fetchMock.mockReset().mockResolvedValue(status(200))
  vi.stubGlobal("fetch", fetchMock)
  vi.stubGlobal("WebSocket", FakeSocket)
  vi.stubGlobal("parent", { postMessage: vi.fn() })
  setSecureContext(true)
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  if (secureContextDescriptor) Object.defineProperty(window, "isSecureContext", secureContextDescriptor)
  else delete (window as { isSecureContext?: unknown }).isSecureContext
  if (originalMediaDevices) Object.defineProperty(navigator, "mediaDevices", originalMediaDevices)
  else delete (navigator as { mediaDevices?: unknown }).mediaDevices
  window.history.replaceState({}, "", "/")
})

describe("session widget voice call offer", () => {
  it.each([
    { label: "voice on, voice=1", capabilities: { voice: true }, voiceParam: true, secure: true, shown: true },
    { label: "voice on, no voice=1", capabilities: { voice: true }, voiceParam: false, secure: true, shown: false },
    { label: "voice off, voice=1", capabilities: { voice: false }, voiceParam: true, secure: true, shown: false },
    { label: "voice off, no voice=1", capabilities: { voice: false }, voiceParam: false, secure: true, shown: false },
    { label: "voice missing, voice=1", capabilities: undefined, voiceParam: true, secure: true, shown: false },
    { label: "voice not a boolean, voice=1", capabilities: { voice: "true" }, voiceParam: true, secure: true, shown: false },
    { label: "voice on, voice=1, page not secure", capabilities: { voice: true }, voiceParam: true, secure: false, shown: false },
  ])("offers the call only when it may: $label", ({ capabilities, voiceParam, secure, shown }) => {
    setSecureContext(secure)
    renderPage({ capabilities, voiceParam })
    expect(screen.queryByText("voice:start") !== null).toBe(shown)
    expect(voice.props !== null).toBe(shown)
    expect(voiceRequests()).toEqual([])
    expect(sockets).toEqual([])
  })

  it("hides the call where a permissions policy denies the microphone", () => {
    const policy = { allowsFeature: vi.fn(() => false) }
    Object.defineProperty(document, "featurePolicy", { configurable: true, value: policy })
    try {
      renderPage()
      expect(policy.allowsFeature).toHaveBeenCalledWith("microphone")
      expect(screen.queryByText("voice:start")).toBeNull()
      expect(voice.props).toBeNull()
    } finally {
      delete (document as { featurePolicy?: unknown }).featurePolicy
    }
  })

  it("shows the call when a later session payload offers voice, without remounting the page", () => {
    renderPage({ capabilities: { voice: false } })
    const start = screen.getByTestId("start")
    expect(screen.queryByText("voice:start")).toBeNull()
    setBridge("active", session({ voice: true }))
    rerender()
    expect(screen.getByText("voice:start")).toBeInTheDocument()
    expect(screen.getByTestId("start")).toBe(start)
  })

  it("keeps the offer while the token is refreshed", () => {
    app.state.taskId = 71
    renderPage()
    setBridge("refreshing", null)
    rerender()
    expect(screen.getByText("voice:panel")).toBeInTheDocument()
  })
})

describe("session widget voice call transport", () => {
  it("talks to the session voice endpoints with the page's current session token", async () => {
    renderPage()
    const transport = await openCall()
    transport.openAsrSocket()
    const url = new URL(window.location.origin)
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:"
    url.pathname = "/v1/external/chat/sessions/voice/asr"
    expect(sockets).toEqual([[url.toString(), ["xagent-session-v1", `xagent-session-token.${TOKEN}`]]])
    expect(transport.startFrame).toEqual({ type: "start", format: "pcm", sample_rate: 16000 })
    const signal = new AbortController().signal
    await transport.requestTts("Hello", signal)
    expect(fetchMock).toHaveBeenCalledExactlyOnceWith(ttsUrl, {
      method: "POST", signal,
      headers: { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" },
      body: JSON.stringify({ text: "Hello" }),
    })
    expect(bridge.value.ensureFreshSession).not.toHaveBeenCalled()
  })

  it("synthesizes with the token the page holds at the time of each request", async () => {
    renderPage()
    const transport = await openCall()
    setBridge("active", session({ voice: true }, "st_rotated"))
    rerender()
    await transport.requestTts("Later", new AbortController().signal)
    expect(fetchMock.mock.calls[0][1].headers.Authorization).toBe("Bearer st_rotated")
  })

  it("refreshes the rejected token once after a 401 and retries", async () => {
    renderPage()
    const transport = await openCall()
    fetchMock.mockResolvedValueOnce(status(401)).mockResolvedValueOnce(status(200))
    bridge.value.refreshSession.mockResolvedValue(session({ voice: true }, "st_renewed"))
    const response = await transport.requestTts("Hello", new AbortController().signal)
    expect(response.status).toBe(200)
    expect(bridge.value.refreshSession).toHaveBeenCalledExactlyOnceWith(TOKEN)
    expect(authorizations()).toEqual([`Bearer ${TOKEN}`, "Bearer st_renewed"])
  })

  it("ends as unavailable when the retry is rejected again", async () => {
    renderPage()
    const transport = await openCall()
    fetchMock.mockResolvedValue(status(401))
    bridge.value.refreshSession.mockResolvedValue(session({ voice: true }, "st_renewed"))
    const response = await transport.requestTts("Hello", new AbortController().signal)
    expect(response.status).toBe(401)
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("hands back the 401 without a retry when no new token comes", async () => {
    renderPage()
    const transport = await openCall()
    fetchMock.mockResolvedValue(status(401))
    bridge.value.refreshSession.mockRejectedValue(new Error("session unavailable"))
    expect((await transport.requestTts("Hello", new AbortController().signal)).status).toBe(401)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it.each([403, 429, 500, 503])("never refreshes after a %i", async code => {
    renderPage()
    const transport = await openCall()
    fetchMock.mockResolvedValue(status(code))
    expect((await transport.requestTts("Hello", new AbortController().signal)).status).toBe(code)
    expect(bridge.value.refreshSession).not.toHaveBeenCalled()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("shares one replacement when two sentences are refused with the same token together", async () => {
    renderPage()
    const transport = await openCall()
    fetchMock.mockImplementation(async (_url: string, init: { headers: { Authorization: string } }) =>
      status(init.headers.Authorization === `Bearer ${TOKEN}` ? 401 : 200))
    bridge.value.refreshSession.mockResolvedValue(session({ voice: true }, "st_renewed"))
    const signal = new AbortController().signal
    const [first, second] = await Promise.all([transport.requestTts("One", signal), transport.requestTts("Two", signal)])
    expect([first.status, second.status]).toEqual([200, 200])
    expect(bridge.value.refreshSession).toHaveBeenCalledExactlyOnceWith(TOKEN)
  })

  it("does not open recognition while the page holds no token", async () => {
    renderPage()
    const transport = await openCall()
    setBridge("refreshing", null)
    rerender()
    expect(() => transport.openAsrSocket()).toThrow()
  })

  it("stops waiting for the token when the sentence is given up", async () => {
    renderPage()
    const transport = await openCall()
    setBridge("refreshing", null)
    rerender()
    bridge.value.waitForOpenConnection.mockReturnValue(new Promise(() => undefined))
    const controller = new AbortController()
    const request = transport.requestTts("Hello", controller.signal)
    controller.abort()
    await expect(request).rejects.toMatchObject({ name: "AbortError" })
    await expect(transport.requestTts("Again", controller.signal)).rejects.toMatchObject({ name: "AbortError" })
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe("session widget voice call wiring", () => {
  it.each([
    { label: "waiting on this task's question", status: "waiting_for_user", id: "71", metadata: { request_id: "question-9" } },
    { label: "waiting on another task's question", status: "waiting_for_user", id: "70", metadata: undefined },
    { label: "running", status: "running", id: "71", metadata: undefined },
  ])("answers like a typed message while the task is $label", async ({ status: taskStatus, id, metadata }) => {
    app.state.taskId = 71
    app.state.currentTask = { id, status: taskStatus, waitingRequestId: "question-9" }
    renderPage()
    await act(async () => { await voice.props!.onSend("my answer", "voice-id") })
    expect(app.sendMessage).toHaveBeenCalledExactlyOnceWith(
      "my answer", { clientMessageId: "voice-id", ...(metadata ? { metadata } : {}) }, [])
  })

  it("sends the first spoken message the way the start page sends a typed one", async () => {
    renderPage()
    await act(async () => { await voice.props!.onSend("hello", "voice-id") })
    expect(app.sendMessage).toHaveBeenCalledExactlyOnceWith("hello", { clientMessageId: "voice-id" }, [])
  })

  it("shows the start page's failure line when the first spoken message cannot be sent", async () => {
    renderPage()
    app.sendMessage.mockRejectedValue(new Error("boom"))
    await act(async () => { await voice.props!.onSend("hello", "voice-id").catch(() => undefined) })
    expect(screen.getByRole("alert")).toHaveTextContent("widgetSession.startMessageFailed")
  })

  it.each([
    { label: "a stop is possible", canStop: true, interrupted: true },
    { label: "no stop is possible", canStop: false, interrupted: false },
  ])("interrupts only when $label", ({ canStop, interrupted }) => {
    app.canStopTask = canStop
    renderPage()
    expect(voice.props!.onInterrupt()).toBe(interrupted)
    expect(app.stopTask).toHaveBeenCalledTimes(interrupted ? 1 : 0)
  })

  type Held = { delivery?: boolean; reset?: boolean; locked?: boolean; phase?: string }
  it.each([
    { label: "active and connected", status: "active", connected: true, sending: false, processing: false, task: "", busy: false, disabled: false },
    { label: "waiting", status: "waiting", connected: false, sending: false, processing: false, task: "", busy: true, disabled: false },
    { label: "refreshing the token", status: "refreshing", connected: false, sending: false, processing: false, task: "", busy: true, disabled: false },
    { label: "a send in flight", status: "active", connected: true, sending: true, processing: false, task: "", busy: true, disabled: false },
    { label: "processing", status: "active", connected: true, sending: false, processing: true, task: "", busy: true, disabled: false },
    { label: "a queued task", status: "active", connected: true, sending: false, processing: false, task: "pending", busy: true, disabled: false },
    { label: "a question waiting", status: "active", connected: true, sending: false, processing: false, task: "waiting_for_user", busy: false, disabled: false },
    { label: "degraded", status: "degraded", connected: false, sending: false, processing: false, task: "", busy: true, disabled: true },
    { label: "terminal", status: "terminal", connected: false, sending: false, processing: false, task: "", busy: true, disabled: true },
    { label: "a typed message being delivered", status: "active", connected: true, sending: false, processing: false, task: "", busy: true, disabled: false, held: { delivery: true } },
    { label: "a new conversation being set up", status: "active", connected: true, sending: false, processing: false, task: "", busy: true, disabled: false, held: { reset: true, phase: "reset_requested" } },
    { label: "a reload required", status: "active", connected: true, sending: false, processing: false, task: "", busy: false, disabled: true, held: { locked: true, phase: "reload_required" } },
  ] as Array<{ label: string; status: WidgetSessionStatus; connected: boolean; sending: boolean; processing: boolean; task: string; busy: boolean; disabled: boolean; held?: Held }>)("derives busy and disabled when $label", ({ status: bridgeStatus, connected, sending, processing, task, busy, disabled, held }) => {
    app.state.taskId = 71
    renderPage()
    app.isConnected = connected
    app.isSendInFlight = sending
    app.isMessageDeliveryPending = held?.delivery ?? false
    app.isConversationResetPending = held?.reset ?? false
    app.isSessionInteractionLocked = held?.locked ?? false
    app.sessionConversationState = held?.phase ?? "unbound"
    app.state = { ...app.state, isProcessing: processing, currentTask: task ? { id: "71", status: task } : null }
    setBridge(bridgeStatus, bridgeStatus === "active" ? session({ voice: true }) : null)
    rerender()
    expect(voice.props).toMatchObject({ busy, disabled, waiting: task === "waiting_for_user" })
  })

  it("holds spoken words while a typed message's files are uploading", async () => {
    app.state.taskId = 71
    renderPage({ capabilities: { voice: true, fileUploads: { allowedExtensions: [".txt"], maxFileBytes: 1024, maxFilesPerMessage: 1 } } })
    await openCall("panel")
    bridge.value.ensureFreshSession.mockReturnValue(new Promise(() => undefined))
    act(() => { void panel.onSend!("typed", {}, [new File(["x"], "a.txt")]).catch(() => undefined) })
    expect(voice.props!.busy).toBe(true)
    vi.useFakeTimers()
    try {
      act(() => voice.callbacks!.transcript("final", "Hello", 1))
      await act(async () => { await vi.advanceTimersByTimeAsync(350) })
    } finally {
      vi.useRealTimers()
    }
    expect(app.sendMessage).not.toHaveBeenCalled()
  })

  it("holds spoken words while a message from history is handed over", () => {
    renderPage({ capabilities: { voice: true, conversationHistory: true } })
    app.sendMessage.mockReturnValue(new Promise(() => undefined))
    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.history.menuLabel" }))
    fireEvent.click(screen.getByText("list-open"))
    fireEvent.click(screen.getByText("view-continue"))
    expect(voice.props!.busy).toBe(true)
  })
})

describe("session widget voice call across the session", () => {
  it("keeps the call through a token refresh and sends the held words once reconnected", async () => {
    app.state.taskId = 71
    renderPage()
    await openCall("panel")
    app.isConnected = false
    setBridge("refreshing", null)
    rerender()
    expect(screen.getByTestId("call")).toHaveTextContent("listening")
    expect(voice.props!.busy).toBe(true)
    // Past the 300 ms a finished sentence waits before it is sent.
    vi.useFakeTimers()
    try {
      act(() => voice.callbacks!.transcript("final", "Hello", 1))
      await act(async () => { await vi.advanceTimersByTimeAsync(350) })
    } finally {
      vi.useRealTimers()
    }
    expect(app.sendMessage).not.toHaveBeenCalled()
    app.isConnected = true
    setBridge("active", session({ voice: true }, "st_renewed"))
    rerender()
    await waitFor(() => expect(app.sendMessage).toHaveBeenCalledWith("Hello", { clientMessageId: expect.any(String) }, []))
    expect(voice.audios[0].close).not.toHaveBeenCalled()
    expect(voice.audios).toHaveLength(1)
  })

  describe("with a reply due while the token is refreshed", () => {
    const audioHeaders = { "X-Audio-Format": "pcm", "X-Audio-Sample-Rate": "24000" }
    // The visitor's words go out, then the session is refreshed and the reply arrives.
    async function replyDuringRefresh() {
      stubRealAudio()
      renderPage()
      await openRealCall("start")
      act(() => {
        socketInstances[0].onmessage!({ data: JSON.stringify({ type: "final", text: "Hello", segment_id: 1 }) })
      })
      await waitFor(() => expect(app.sendMessage).toHaveBeenCalled())
      let release!: () => void
      let fail!: (error: Error) => void
      bridge.value.waitForOpenConnection.mockReturnValue(new Promise<void>((resolve, reject) => { release = resolve; fail = reject }))
      app.isConnected = false
      setBridge("refreshing", null)
      app.state = {
        ...app.state,
        messages: [{ id: "a1", role: "assistant", content: "Hi there.", isResult: true, status: "completed" }] as never,
      }
      rerender()
      return { release: () => release(), fail: (error: Error) => fail(error) }
    }
    const reconnect = () => {
      app.isConnected = true
      setBridge("active", session({ voice: true }, "st_renewed"))
      rerender()
    }

    it("speaks the sentence with the new token, and the call goes on", async () => {
      const { release } = await replyDuringRefresh()
      await waitFor(() => expect(bridge.value.waitForOpenConnection).toHaveBeenCalled())
      expect(fetchMock).not.toHaveBeenCalled()
      expect(bridge.value.refreshSession).not.toHaveBeenCalled()
      expect(bridge.value.ensureFreshSession).not.toHaveBeenCalled()
      fetchMock.mockResolvedValue(new Response(new Uint8Array(0), { headers: audioHeaders }))
      reconnect()
      release()
      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
      expect(authorizations()).toEqual(["Bearer st_renewed"])
      await waitFor(() => expect(screen.getByTestId("call")).toHaveTextContent("listening"))
      expect(screen.queryByTestId("note")).toBeNull()
      expect(screen.queryByTestId("alert")).toBeNull()
      expect(stopTrack).not.toHaveBeenCalled()
    })

    it("skips the sentence with a note, and the call goes on, when no token comes in time", async () => {
      const { fail } = await replyDuringRefresh()
      await waitFor(() => expect(bridge.value.waitForOpenConnection).toHaveBeenCalled())
      fail(new Error("timed out"))
      await waitFor(() => expect(screen.getByTestId("note")).toHaveTextContent("voiceConversation.widget.playbackFailed"))
      expect(screen.getByTestId("call")).not.toHaveTextContent("none")
      expect(screen.queryByTestId("alert")).toBeNull()
      expect(fetchMock).not.toHaveBeenCalled()
      expect(stopTrack).not.toHaveBeenCalled()
    })
  })

  it.each(["terminal", "degraded"] as const)("hangs up when the session turns %s", async ending => {
    app.state.taskId = 71
    renderPage()
    await openCall("panel")
    setBridge(ending, null)
    rerender()
    expect(voice.audios[0].close).toHaveBeenCalled()
    expect(voice.props!.disabled).toBe(true)
  })

  it("releases the call when the page unmounts", async () => {
    renderPage()
    await openCall()
    cleanup()
    expect(voice.audios[0].close).toHaveBeenCalled()
  })

  it("keeps the call when the first message gives the conversation a task id", async () => {
    renderPage()
    await openCall()
    app.state = { ...app.state, taskId: 71 }
    rerender()
    expect(voice.audios[0].close).not.toHaveBeenCalled()
    expect(screen.getByTestId("call")).toHaveTextContent("listening")
  })

  it("keeps the call when a restored conversation arrives", async () => {
    renderPage()
    await openCall()
    app.state = { ...app.state, taskId: 55, currentTask: { id: "55", status: "completed" } }
    rerender()
    expect(voice.audios[0].close).not.toHaveBeenCalled()
  })

  it("hangs up and closes the bar on a new conversation", async () => {
    app.state.taskId = 71
    renderPage()
    await openCall("panel")
    app.state = { ...app.state, taskId: null }
    rerender()
    expect(voice.audios[0].close).toHaveBeenCalled()
    expect(screen.getByTestId("call")).toHaveTextContent("none")
  })

  it("closes the call when a later session no longer offers voice", async () => {
    renderPage()
    await openCall()
    setBridge("active", session(undefined))
    rerender()
    expect(voice.audios[0].close).toHaveBeenCalled()
    setBridge("active", session({ voice: true }))
    rerender()
    expect(screen.getByTestId("call")).toHaveTextContent("none")
    expect(voice.audios).toHaveLength(1)
  })
})

describe("session widget voice call and abandoned sentences", () => {
  it("neither refreshes nor retries when the sentence was abandoned before its 401 arrived", async () => {
    renderPage()
    const transport = await openCall()
    const controller = new AbortController()
    fetchMock.mockImplementationOnce(async () => { controller.abort(); return status(401) })
    expect((await transport.requestTts("Hello", controller.signal)).status).toBe(401)
    expect(bridge.value.refreshSession).not.toHaveBeenCalled()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("does not retry when the sentence was abandoned while the token was replaced", async () => {
    renderPage()
    const transport = await openCall()
    const controller = new AbortController()
    fetchMock.mockResolvedValue(status(401))
    bridge.value.refreshSession.mockImplementation(async () => {
      controller.abort()
      return session({ voice: true }, "st_renewed")
    })
    expect((await transport.requestTts("Hello", controller.signal)).status).toBe(401)
    expect(bridge.value.refreshSession).toHaveBeenCalledExactlyOnceWith(TOKEN)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})

const stopTrack = vi.fn()
class Context {
  state = "running"
  sampleRate = 48000
  currentTime = 0
  destination = {}
  resume = vi.fn().mockResolvedValue(undefined)
  close = vi.fn().mockResolvedValue(undefined)
  audioWorklet = { addModule: vi.fn().mockResolvedValue(undefined) }
  createMediaStreamSource = () => ({ connect: vi.fn(), disconnect: vi.fn() })
  createAnalyser = () => ({ fftSize: 0, connect: vi.fn(), disconnect: vi.fn(), getFloatTimeDomainData: vi.fn() })
  createGain = () => ({ gain: { value: 0 }, connect: vi.fn(), disconnect: vi.fn() })
}
// Runs the real call audio over a stubbed microphone and audio context, so a
// test sees what the visitor would: the call bar, its notes and the tracks.
function stubRealAudio() {
  voice.real = true
  stopTrack.mockReset()
  vi.stubGlobal("AudioContext", Context)
  vi.stubGlobal("AudioWorkletNode", class {
    port = { onmessage: null }
    connect = () => ({ connect: vi.fn() })
    disconnect = vi.fn()
  })
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: vi.fn(async () => ({ getTracks: () => [{ stop: stopTrack, enabled: true }] })) },
  })
}
// Opens a call from the start page or the panel and brings it to listening.
async function openRealCall(where: "start" | "panel") {
  fireEvent.click(screen.getByText(`voice:${where}`))
  await waitFor(() => expect(socketInstances).toHaveLength(1))
  await act(async () => {
    socketInstances[0].onopen!()
    socketInstances[0].onmessage!({ data: JSON.stringify({ type: "session_started" }) })
  })
  await waitFor(() => expect(screen.getByTestId("call")).toHaveTextContent("listening"))
}

describe("session widget voice call and history", () => {
  beforeEach(stubRealAudio)

  const historyCapabilities = { voice: true, conversationHistory: true }
  async function callOnPanel() {
    app.state.taskId = 71
    renderPage({ capabilities: historyCapabilities })
    await openRealCall("panel")
  }
  const openHistory = () => {
    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.history.menuLabel" }))
  }

  it("hangs up and stops the microphone when the history list opens", async () => {
    await callOnPanel()
    openHistory()
    expect(screen.getByTestId("history-list")).toBeInTheDocument()
    await waitFor(() => expect(stopTrack).toHaveBeenCalled())
    expect(socketInstances[0].close).toHaveBeenCalled()
    expect(screen.getByTestId("call")).toHaveTextContent("none")
    expect(voice.props!.open).toBe(false)
  })

  it("stays hung up through an earlier conversation and back, until the visitor calls again", async () => {
    await callOnPanel()
    openHistory()
    fireEvent.click(screen.getByText("list-open"))
    expect(screen.getByTestId("history-view")).toBeInTheDocument()
    await waitFor(() => expect(stopTrack).toHaveBeenCalled())
    fireEvent.click(screen.getByText("view-back"))
    fireEvent.click(screen.getByText("list-back"))
    expect(screen.queryByTestId("history-list")).toBeNull()
    expect(screen.getByTestId("call")).toHaveTextContent("none")
    expect(socketInstances).toHaveLength(1)
    fireEvent.click(screen.getByText("voice:panel"))
    await waitFor(() => expect(socketInstances).toHaveLength(2))
  })

  it("stays hung up when a message is continued from an earlier conversation", async () => {
    await callOnPanel()
    openHistory()
    fireEvent.click(screen.getByText("list-open"))
    fireEvent.click(screen.getByText("view-continue"))
    expect(screen.queryByTestId("history-view")).toBeNull()
    await waitFor(() => expect(stopTrack).toHaveBeenCalled())
    expect(screen.getByTestId("call")).toHaveTextContent("none")
    expect(socketInstances).toHaveLength(1)
  })
})

describe("session widget voice call and a refused token", () => {
  const connectSocket = async (index: number) => {
    await waitFor(() => expect(socketInstances).toHaveLength(index + 1))
    await act(async () => {
      socketInstances[index].onopen!()
      socketInstances[index].onmessage!({ data: JSON.stringify({ type: "session_started" }) })
    })
    await waitFor(() => expect(screen.getByTestId("call")).toHaveTextContent("listening"))
  }
  const refuseHandshake = async (index: number) => {
    act(() => socketInstances[index].serverClose(4401))
    await waitFor(() => expect(screen.getByTestId("alert")).toHaveTextContent("voiceConversation.widget.connectionEnded"))
  }
  const retryButton = () => screen.getByRole("button", { name: "retry" })
  const handshakeToken = (index: number) => sockets[index][1][1]

  it("ends the session as expired when recognition is closed with 4408", async () => {
    stubRealAudio()
    app.state.taskId = 71
    renderPage()
    await openRealCall("panel")
    act(() => socketInstances[0].serverClose(4408))
    expect(bridge.value.handleSessionExpired).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(screen.getByTestId("alert")).toHaveTextContent("voiceConversation.widget.unavailable"))
    expect(screen.queryByText("retry")).toBeNull()
    expect(stopTrack).toHaveBeenCalled()
    expect(bridge.value.refreshSession).not.toHaveBeenCalled()

    // The bridge then reports what a chat socket closed with 4408 leads to.
    setBridge("terminal", null, "ws_4408")
    rerender()
    expect(screen.getByText("widgetSession.expired.title")).toBeInTheDocument()
    expect(voice.props!.disabled).toBe(true)
    expect(socketInstances).toHaveLength(1)
  })

  it("reads a refused handshake only by what it means for the call", async () => {
    renderPage()
    const transport = await openCall()
    expect(transport.classify).toBe(classifyWidgetVoiceEvent)
  })

  it("replaces a refused handshake token once for that token, and a new token again", async () => {
    stubRealAudio()
    renderPage()
    bridge.value.refreshSession.mockResolvedValue(session({ voice: true }, "st_renewed"))
    await openRealCall("start")
    await refuseHandshake(0)
    expect(bridge.value.refreshSession).toHaveBeenCalledExactlyOnceWith(TOKEN)

    // The bridge still holds the refused token: the retry offers it again, and
    // its refusal does not ask for another replacement.
    fireEvent.click(retryButton())
    await connectSocket(1)
    expect(handshakeToken(1)).toBe(`xagent-session-token.${TOKEN}`)
    await refuseHandshake(1)
    expect(bridge.value.refreshSession).toHaveBeenCalledTimes(1)

    // While the chat socket is rebuilt the words are held and the retry waits.
    app.isConnected = false
    setBridge("refreshing", null)
    rerender()
    expect(retryButton()).toBeDisabled()
    app.isConnected = true
    setBridge("active", session({ voice: true }, "st_renewed"))
    rerender()
    expect(retryButton()).toBeEnabled()
    fireEvent.click(retryButton())
    await connectSocket(2)
    expect(handshakeToken(2)).toBe("xagent-session-token.st_renewed")

    // Another token refused is another replacement.
    await refuseHandshake(2)
    expect(bridge.value.refreshSession.mock.calls).toEqual([[TOKEN], ["st_renewed"]])

    // A new call keeps its own record.
    fireEvent.click(screen.getByText("hang up"))
    await waitFor(() => expect(screen.getByTestId("call")).toHaveTextContent("none"))
    fireEvent.click(screen.getByText("voice:start"))
    await connectSocket(3)
    await refuseHandshake(3)
    expect(bridge.value.refreshSession.mock.calls).toEqual([[TOKEN], ["st_renewed"], ["st_renewed"]])
  })

  it("keeps the record of replaced tokens when start is pressed while the call bar is open", async () => {
    stubRealAudio()
    renderPage()
    bridge.value.refreshSession.mockResolvedValue(session({ voice: true }, "st_renewed"))
    await openRealCall("start")
    await refuseHandshake(0)
    expect(bridge.value.refreshSession).toHaveBeenCalledExactlyOnceWith(TOKEN)

    fireEvent.click(screen.getByText("voice:start"))
    fireEvent.click(retryButton())
    await connectSocket(1)
    expect(handshakeToken(1)).toBe(`xagent-session-token.${TOKEN}`)
    await refuseHandshake(1)
    expect(bridge.value.refreshSession).toHaveBeenCalledTimes(1)
  })

  it("writes a late close of an earlier call's socket to that call's record only", async () => {
    stubRealAudio()
    renderPage()
    bridge.value.refreshSession.mockResolvedValue(session({ voice: true }, "st_renewed"))
    await openRealCall("start")
    fireEvent.click(screen.getByText("hang up"))
    await waitFor(() => expect(screen.getByTestId("call")).toHaveTextContent("none"))
    fireEvent.click(screen.getByText("voice:start"))
    await connectSocket(1)

    act(() => socketInstances[0].serverClose(4401))
    expect(bridge.value.refreshSession).toHaveBeenCalledExactlyOnceWith(TOKEN)

    await refuseHandshake(1)
    expect(bridge.value.refreshSession.mock.calls).toEqual([[TOKEN], [TOKEN]])
  })

  it("asks once for a token that cannot be replaced, and each refusal ends the call retryable", async () => {
    stubRealAudio()
    renderPage()
    bridge.value.refreshSession.mockRejectedValue(new Error("session unavailable"))
    await openRealCall("start")
    await refuseHandshake(0)
    fireEvent.click(retryButton())
    await connectSocket(1)
    await refuseHandshake(1)
    expect(bridge.value.refreshSession).toHaveBeenCalledTimes(1)
  })

  it("keeps one record of a token across the recognition socket and synthesis", async () => {
    renderPage()
    const transport = await openCall()
    bridge.value.refreshSession.mockResolvedValue(session({ voice: true }, "st_renewed"))
    transport.openAsrSocket()
    act(() => socketInstances.at(-1)!.serverClose(4401))
    expect(bridge.value.refreshSession).toHaveBeenCalledExactlyOnceWith(TOKEN)

    // The same token refused for synthesis shares that replacement.
    fetchMock.mockImplementation(async (_url: string, init: { headers: { Authorization: string } }) =>
      status(init.headers.Authorization === `Bearer ${TOKEN}` ? 401 : 200))
    const signal = new AbortController().signal
    expect((await transport.requestTts("One", signal)).status).toBe(200)
    expect(bridge.value.refreshSession).toHaveBeenCalledTimes(1)
    expect(authorizations()).toEqual([`Bearer ${TOKEN}`, "Bearer st_renewed"])

    // A token the bridge has since replaced is refused as a token of its own.
    setBridge("active", session({ voice: true }, "st_renewed"))
    rerender()
    bridge.value.refreshSession.mockResolvedValue(session({ voice: true }, "st_again"))
    fetchMock.mockReset().mockResolvedValueOnce(status(401)).mockResolvedValueOnce(status(200))
    expect((await transport.requestTts("Two", signal)).status).toBe(200)
    expect(bridge.value.refreshSession.mock.calls).toEqual([[TOKEN], ["st_renewed"]])
  })

  it.each([
    "reset_requested", "replacement_ready", "replacement_sending", "replacement_awaiting_task", "reload_required",
  ])("keeps the token while the conversation is in %s", async phase => {
    renderPage()
    const transport = await openCall()
    app.sessionConversationState = phase
    app.isSessionInteractionLocked = phase === "reload_required"
    rerender()
    bridge.value.refreshSession.mockResolvedValue(session({ voice: true }, "st_renewed"))
    fetchMock.mockResolvedValue(status(401))
    expect((await transport.requestTts("Hello", new AbortController().signal)).status).toBe(401)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    transport.openAsrSocket()
    act(() => socketInstances.at(-1)!.serverClose(4401))
    expect(bridge.value.refreshSession).not.toHaveBeenCalled()
  })

  it.each(["bound"])("replaces the token while the conversation is %s", async phase => {
    renderPage()
    const transport = await openCall()
    app.sessionConversationState = phase
    rerender()
    bridge.value.refreshSession.mockResolvedValue(session({ voice: true }, "st_renewed"))
    fetchMock.mockResolvedValueOnce(status(401)).mockResolvedValueOnce(status(200))
    expect((await transport.requestTts("Hello", new AbortController().signal)).status).toBe(200)
    expect(bridge.value.refreshSession).toHaveBeenCalledExactlyOnceWith(TOKEN)
  })

  it("reads the phase when the refusal arrives, not when the call started", async () => {
    renderPage()
    const transport = await openCall()
    transport.openAsrSocket()
    app.sessionConversationState = "replacement_ready"
    rerender()
    act(() => socketInstances.at(-1)!.serverClose(4401))
    expect(bridge.value.refreshSession).not.toHaveBeenCalled()
  })

  it("ends the call as unavailable, with no retry, when the session widget API is switched off (404)", async () => {
    stubRealAudio()
    renderPage()
    await openRealCall("start")
    act(() => {
      socketInstances[0].onmessage!({ data: JSON.stringify({ type: "final", text: "Hello", segment_id: 1 }) })
    })
    await waitFor(() => expect(app.sendMessage).toHaveBeenCalled())
    app.state = {
      ...app.state,
      messages: [{ id: "a1", role: "assistant", content: "Hi there.", isResult: true, status: "completed" }] as never,
    }
    fetchMock.mockResolvedValue(status(404))
    rerender()
    await waitFor(() => expect(screen.getByTestId("alert")).toHaveTextContent("voiceConversation.widget.unavailable"))
    expect(screen.queryByText("retry")).toBeNull()
    expect(stopTrack).toHaveBeenCalled()
    expect(bridge.value.refreshSession).not.toHaveBeenCalled()
  })
})

describe("the test stubs", () => {
  it("leave navigator.mediaDevices as they found it", () => {
    expect(Object.getOwnPropertyDescriptor(navigator, "mediaDevices")).toEqual(originalMediaDevices)
  })
})
