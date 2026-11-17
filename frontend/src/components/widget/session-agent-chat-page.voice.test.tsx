import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useVoiceCallView } from "@/contexts/voice-call-store"
import type { WidgetVoiceCallOptions } from "@/hooks/use-widget-voice-call"
import type { VoiceAudioTransport, VoiceTransportEvent } from "@/lib/voice-audio"
import type { WidgetSession, WidgetSessionStatus } from "./use-widget-session"

const bridge = vi.hoisted(() => ({
  value: {
    status: "active" as string,
    session: null as unknown,
    agent: null as unknown,
    terminalCode: null as string | null,
    isAbsoluteExpiryWarningVisible: false,
    requestReconnect: vi.fn(),
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
  bridge.value.requestReconnect.mockReset()
  voice.props = null; voice.transport = null; voice.audios = []; voice.real = false
  sockets.length = 0
  socketInstances.length = 0
  fetchMock.mockReset().mockResolvedValue(status(200))
  vi.stubGlobal("fetch", fetchMock)
  vi.stubGlobal("WebSocket", FakeSocket)
  vi.stubGlobal("parent", { postMessage: vi.fn() })
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  if (originalMediaDevices) Object.defineProperty(navigator, "mediaDevices", originalMediaDevices)
  else delete (navigator as { mediaDevices?: unknown }).mediaDevices
  window.history.replaceState({}, "", "/")
})

describe("session widget voice call offer", () => {
  it.each([
    { label: "voice on, voice=1", capabilities: { voice: true }, voiceParam: true, shown: true },
    { label: "voice on, no voice=1", capabilities: { voice: true }, voiceParam: false, shown: false },
    { label: "voice off, voice=1", capabilities: { voice: false }, voiceParam: true, shown: false },
    { label: "voice off, no voice=1", capabilities: { voice: false }, voiceParam: false, shown: false },
    { label: "voice missing, voice=1", capabilities: undefined, voiceParam: true, shown: false },
    { label: "voice not a boolean, voice=1", capabilities: { voice: "true" }, voiceParam: true, shown: false },
  ])("offers the call only when it may: $label", ({ capabilities, voiceParam, shown }) => {
    renderPage({ capabilities, voiceParam })
    expect(screen.queryByText("voice:start") !== null).toBe(shown)
    expect(voice.props !== null).toBe(shown)
    expect(voiceRequests()).toEqual([])
    expect(sockets).toEqual([])
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

  const authorizations = () => fetchMock.mock.calls.map(([, init]) => init.headers.Authorization)

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
    expect(transport.classify({ kind: "tts_response", status: 401, code: undefined })).toBe("voice_unavailable")
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

  it("asks the bridge to replace the same rejected token when two sentences are refused together", async () => {
    renderPage()
    const transport = await openCall()
    fetchMock.mockImplementation(async (_url: string, init: { headers: { Authorization: string } }) =>
      status(init.headers.Authorization === `Bearer ${TOKEN}` ? 401 : 200))
    bridge.value.refreshSession.mockResolvedValue(session({ voice: true }, "st_renewed"))
    const signal = new AbortController().signal
    const [first, second] = await Promise.all([transport.requestTts("One", signal), transport.requestTts("Two", signal)])
    expect([first.status, second.status]).toEqual([200, 200])
    expect(bridge.value.refreshSession.mock.calls).toEqual([[TOKEN], [TOKEN]])
  })

  it("fails a sentence, not the call, while the page holds no token", async () => {
    renderPage()
    const transport = await openCall()
    setBridge("refreshing", null)
    rerender()
    await expect(transport.requestTts("Hello", new AbortController().signal)).rejects.toThrow()
    expect(() => transport.openAsrSocket()).toThrow()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it.each([
    { label: "recognition closed 4403", event: { kind: "asr_close", code: 4403, reason: "" }, failure: "voice_unavailable" },
    { label: "recognition closed 4408", event: { kind: "asr_close", code: 4408, reason: "" }, failure: "connection_failed" },
    { label: "synthesis 403", event: { kind: "tts_response", status: 403, code: undefined }, failure: "voice_unavailable" },
    { label: "synthesis 401 after the retry", event: { kind: "tts_response", status: 401, code: undefined }, failure: "voice_unavailable" },
  ] satisfies Array<{ label: string; event: VoiceTransportEvent; failure: string }>)("reads $label as the state table does", async ({ event, failure }) => {
    renderPage()
    const transport = await openCall()
    expect(transport.classify(event)).toBe(failure)
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
    { label: "a message handed over from history", status: "active", connected: true, sending: false, processing: false, task: "", busy: true, disabled: false, held: { delivery: true } },
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
function useRealAudio() {
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
  beforeEach(useRealAudio)

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

describe("session widget voice call and a refused handshake token", () => {
  const rejected = { kind: "asr_close", code: 4401, reason: "" } as const

  it("replaces the rejected handshake token once and ends retryable, so a retry uses the new token", async () => {
    renderPage()
    const transport = await openCall()
    bridge.value.refreshSession.mockResolvedValue(session({ voice: true }, "st_renewed"))
    transport.openAsrSocket()
    expect(transport.classify(rejected)).toBe("connection_failed")
    expect(bridge.value.refreshSession).toHaveBeenCalledExactlyOnceWith(TOKEN)
    setBridge("active", session({ voice: true }, "st_renewed"))
    rerender()
    transport.openAsrSocket()
    expect(sockets.map(([, protocols]) => protocols[1])).toEqual([`xagent-session-token.${TOKEN}`, "xagent-session-token.st_renewed"])
  })

  it("ends retryable without asking again when the token cannot be replaced", async () => {
    renderPage()
    const transport = await openCall()
    bridge.value.refreshSession.mockRejectedValue(new Error("session unavailable"))
    transport.openAsrSocket()
    expect(transport.classify(rejected)).toBe("connection_failed")
    await act(async () => { await Promise.resolve() })
    expect(transport.classify(rejected)).toBe("connection_failed")
    expect(bridge.value.refreshSession).toHaveBeenCalledTimes(1)
  })

  it("ends the call as unavailable, with no retry, when the session widget API is switched off (404)", async () => {
    useRealAudio()
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
