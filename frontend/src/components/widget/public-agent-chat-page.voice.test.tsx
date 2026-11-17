import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useVoiceCallView } from "@/contexts/voice-call-store"
import type { WidgetVoiceCallOptions } from "@/hooks/use-widget-voice-call"
import type { VoiceAudioTransport } from "@/lib/voice-audio"

const app = vi.hoisted(() => ({
  state: { taskId: null as number | null, currentTask: null as null | { id: string; status: string; waitingRequestId?: string }, isHistoryLoading: false, isProcessing: false, messages: [] },
  connectionPhase: "connected",
  isConnected: true,
  sendMessage: vi.fn(),
  setTaskId: vi.fn(),
  rerender: null as null | React.Dispatch<React.SetStateAction<number>>,
}))
const voice = vi.hoisted(() => ({
  props: null as null | WidgetVoiceCallOptions,
  transport: null as null | VoiceAudioTransport,
  audios: [] as Array<{ close: ReturnType<typeof vi.fn> }>,
}))

vi.mock("@/contexts/app-context-chat", () => ({
  AppProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useApp: () => {
    const [, rerender] = React.useState(0)
    app.rerender = rerender
    return {
      state: app.state, dispatch: vi.fn(), sendMessage: app.sendMessage, setTaskId: app.setTaskId,
      connectionError: null, connectionPhase: app.connectionPhase, isConnected: app.isConnected,
      isMessageDeliveryPending: false, isSendInFlight: false, voiceInputEnabled: false,
    }
  },
}))
// A stable translator: the page re-authenticates whenever `t` changes.
const i18n = vi.hoisted(() => ({ t: (key: string) => key }))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => i18n }))
vi.mock("@/lib/utils", async () => ({
  ...await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils"),
  getApiUrl: () => "https://api.example",
  getWsUrl: () => "wss://api.example/",
}))
vi.mock("@/lib/voice-audio", async () => ({
  ...await vi.importActual<typeof import("@/lib/voice-audio")>("@/lib/voice-audio"),
  createWidgetVoiceAudio: (transport: VoiceAudioTransport) => {
    voice.transport = transport
    const audio = { start: vi.fn().mockResolvedValue(undefined), close: vi.fn(), stopAudio: vi.fn(), speak: vi.fn(), level: () => 0, muted: false }
    voice.audios.push(audio)
    return audio
  },
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
    </div>
  )
}
vi.mock("@/components/chat/ChatStartScreen", () => ({
  ChatStartScreen: (props: Opener) => (
    <Composer where="start" open={props.voiceConversationSource === "provided" ? props.onVoiceConversation : undefined} />
  ),
}))
vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: (props: Opener) => <Composer where="panel" open={props.onProvidedVoiceConversation} />,
}))

import { PublicAgentChatPage } from "./public-agent-chat-page"

const fetchMock = vi.fn()
// The host page answers prompt-context requests (with nothing) as widget.js does.
const postToParent = vi.fn()
const answerPromptContext = (message: { type?: string }) => {
  if (message.type !== "prompt_context_request") return
  window.dispatchEvent(new MessageEvent("message", {
    data: { xagent: true, v: 1, type: "prompt_context", values: null },
    source: window.parent as unknown as MessageEventSource,
  }))
}
const auth = (extra: Record<string, unknown>) => ({
  access_token: "public-access-token", agent_id: 17, agent_name: "Support Agent", suggested_prompts: [], workforce_id: null, ...extra,
})
const json = (payload: unknown) => new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } })

function setFrame({ topLevel, voiceParam }: { topLevel: boolean; voiceParam: boolean }) {
  Object.defineProperty(window, "top", { configurable: true, value: topLevel ? window : {} })
  window.history.replaceState({}, "", voiceParam ? "/widget/chat?voice=1" : "/widget/chat")
}

async function renderPage(authExtra: Record<string, unknown>, mode: "widget" | "share" = "widget") {
  fetchMock.mockImplementation(async (url: string) => (url.endsWith("/chat/task/create")
    ? json({ task_id: 60, status: "pending" })
    : json(auth(authExtra))))
  const view = render(<PublicAgentChatPage authMode={mode} routeToken="share-tok" guestId="guest-1" searchAgentId={17} widgetKey="key" />)
  await screen.findByTestId(app.state.taskId === null ? "start" : "panel")
  return view
}

beforeEach(() => {
  app.state = { taskId: null, currentTask: null, isHistoryLoading: false, isProcessing: false, messages: [] }
  app.connectionPhase = "connected"
  app.isConnected = true
  app.sendMessage.mockReset().mockResolvedValue(undefined)
  app.setTaskId.mockReset().mockImplementation((taskId: number | null) => {
    app.state = { ...app.state, taskId }
    app.rerender?.(value => value + 1)
  })
  voice.props = null; voice.transport = null; voice.audios = []
  localStorage.clear()
  vi.stubGlobal("fetch", fetchMock)
  postToParent.mockImplementation(answerPromptContext)
  vi.stubGlobal("parent", { postMessage: postToParent })
  setFrame({ topLevel: false, voiceParam: true })
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  fetchMock.mockReset(); postToParent.mockReset()
  Object.defineProperty(window, "top", { configurable: true, value: window })
  window.history.replaceState({}, "", "/")
})

describe("widget voice call offer", () => {
  it.each([
    { label: "voice on, iframe with voice=1", voiceEnabled: true, topLevel: false, voiceParam: true, shown: true },
    { label: "voice on, iframe without voice=1", voiceEnabled: true, topLevel: false, voiceParam: false, shown: false },
    { label: "voice off, iframe with voice=1", voiceEnabled: false, topLevel: false, voiceParam: true, shown: false },
    { label: "voice off, iframe without voice=1", voiceEnabled: false, topLevel: false, voiceParam: false, shown: false },
    { label: "voice on, top-level page", voiceEnabled: true, topLevel: true, voiceParam: false, shown: true },
    { label: "voice flag missing, top-level page", voiceEnabled: undefined, topLevel: true, voiceParam: false, shown: false },
  ])("offers the call only when it may: $label", async ({ voiceEnabled, topLevel, voiceParam, shown }) => {
    setFrame({ topLevel, voiceParam })
    await renderPage(voiceEnabled === undefined ? {} : { voice_enabled: voiceEnabled })
    expect(screen.queryByText("voice:start") !== null).toBe(shown)
    expect(voice.props !== null).toBe(shown)
    expect(fetchMock.mock.calls.map(([url]) => String(url)).filter(url => url.includes("/voice/"))).toEqual([])
  })

  it("never offers the call on a share link", async () => {
    setFrame({ topLevel: true, voiceParam: true })
    await renderPage({ voice_enabled: true }, "share")
    expect(screen.queryByText("voice:start")).toBeNull()
    expect(voice.props).toBeNull()
  })
})

describe("widget voice call wiring", () => {
  it("talks to the widget voice endpoints with the guest credential", async () => {
    await renderPage({ voice_enabled: true })
    fireEvent.click(screen.getByText("voice:start"))
    await waitFor(() => expect(voice.transport).not.toBeNull())
    const transport = voice.transport!
    const sockets: Array<[string, string[]]> = []
    vi.stubGlobal("WebSocket", class { constructor(url: string, protocols: string[]) { sockets.push([url, protocols]) } })
    transport.openAsrSocket()
    expect(sockets).toEqual([["wss://api.example/api/widget/voice/asr", ["xagent-auth-v1", "xagent-auth-token.public-access-token"]]])
    expect(transport.startFrame).toEqual({ type: "start", format: "pcm", sample_rate: 16000 })
    const signal = new AbortController().signal
    await transport.requestTts("Hello", signal)
    expect(fetchMock).toHaveBeenLastCalledWith("https://api.example/api/widget/voice/tts", {
      method: "POST", signal,
      headers: { Authorization: "Bearer public-access-token", "Content-Type": "application/json" },
      body: JSON.stringify({ text: "Hello" }),
    })
    expect(transport.classify({ kind: "asr_close", code: 4003, reason: "" })).toBe("voice_unavailable")
  })

  it("sends spoken words through the page's own send, creating the task first", async () => {
    await renderPage({ voice_enabled: true })
    await act(async () => { await voice.props!.onSend("spoken words", "voice-id") })
    expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/api/widget/chat/task/create"))).toBe(true)
    expect(app.setTaskId).toHaveBeenCalledWith(60, { navigate: false })
    expect(app.sendMessage).toHaveBeenCalledWith("spoken words", { clientMessageId: "voice-id", targetTaskId: 60 }, [])
  })

  it.each([
    { label: "waiting on a question", status: "waiting_for_user", taskId: "71", metadata: { request_id: "question-9" } },
    { label: "waiting on another task's question", status: "waiting_for_user", taskId: "70", metadata: undefined },
    { label: "running", status: "running", taskId: "71", metadata: undefined },
  ])("answers like a typed message while the task is $label", async ({ status, taskId, metadata }) => {
    localStorage.setItem("widget_task_17_guest-1", "71")
    app.state.taskId = 71
    app.state.currentTask = { id: taskId, status, waitingRequestId: "question-9" }
    await renderPage({ voice_enabled: true })
    await act(async () => { await voice.props!.onSend("my answer", "voice-id") })
    expect(app.sendMessage).toHaveBeenCalledExactlyOnceWith(
      "my answer", { clientMessageId: "voice-id", ...(metadata ? { metadata } : {}) }, [])
  })

  it("never pauses the run when the visitor talks over it", async () => {
    await renderPage({ voice_enabled: true })
    expect(voice.props!.onInterrupt()).toBe(false)
  })

  it.each([
    { label: "start screen", taskId: null, phase: "lost", connected: false, busy: false, disabled: false },
    { label: "task, connected", taskId: 71, phase: "connected", connected: true, busy: false, disabled: false },
    { label: "task, reconnecting", taskId: 71, phase: "reconnecting", connected: false, busy: true, disabled: false },
    { label: "task, connection lost", taskId: 71, phase: "lost", connected: false, busy: true, disabled: true },
  ])("derives busy and disabled on the $label", async ({ taskId, phase, connected, busy, disabled }) => {
    if (taskId) localStorage.setItem("widget_task_17_guest-1", String(taskId))
    app.state.taskId = taskId
    app.connectionPhase = phase
    app.isConnected = connected
    await renderPage({ voice_enabled: true })
    expect(voice.props).toMatchObject({ busy, disabled, taskId })
  })
})

describe("widget voice call across conversations", () => {
  async function openCallOnPanel() {
    localStorage.setItem("widget_task_17_guest-1", "71")
    app.state.taskId = 71
    await renderPage({ voice_enabled: true })
    fireEvent.click(screen.getByText("voice:panel"))
    await waitFor(() => expect(screen.getByTestId("call")).toHaveTextContent("listening"))
  }

  it("hangs up when the visitor starts a new conversation", async () => {
    await openCallOnPanel()
    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    fireEvent.click(screen.getByRole("menuitem", { name: "widgetChat.newConversation" }))
    await screen.findByTestId("start")
    expect(voice.audios[0].close).toHaveBeenCalled()
    expect(screen.getByTestId("call")).toHaveTextContent("none")
  })

  it("hangs up when the host page opens the widget with a message", async () => {
    await openCallOnPanel()
    await waitFor(() => expect(postToParent.mock.calls.some(([data]) => data.type === "open_message_request")).toBe(true))
    await act(async () => {
      window.dispatchEvent(new MessageEvent("message", {
        data: { xagent: true, v: 1, type: "open_message", id: "1", value: "from host" },
        source: window.parent as unknown as MessageEventSource,
      }))
    })
    await waitFor(() => expect(app.setTaskId).toHaveBeenCalledWith(null, { navigate: false }))
    await waitFor(() => expect(voice.audios[0].close).toHaveBeenCalled())
    expect(screen.getByTestId("call")).toHaveTextContent("none")
  })
})
