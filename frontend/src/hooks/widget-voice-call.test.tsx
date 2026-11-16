import React, { useState } from "react"
import { act, cleanup, render, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { VoiceCall, VoiceReplyMessage } from "@/hooks/use-voice-call"
import {
  classifyWidgetVoiceEvent, useWidgetVoiceCall, WIDGET_VOICE_CALL_STATES, type WidgetVoiceCallOptions,
} from "@/hooks/use-widget-voice-call"
import type { VoiceTransportEvent } from "@/lib/voice-audio"

vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))
vi.mock("@/lib/voice-vad", () => ({ createVoiceVad: vi.fn(async () => ({ process: vi.fn(), destroy: vi.fn().mockResolvedValue(undefined) })) }))

const order: string[] = []
class Socket {
  static OPEN = 1
  static instances: Socket[] = []
  readyState = 1
  bufferedAmount = 0
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: ((event: { code: number; reason: string }) => void) | null = null
  onerror: (() => void) | null = null
  send = vi.fn()
  close = vi.fn()
  constructor(public url: string, public protocols: string[]) { Socket.instances.push(this); order.push("socket") }
}
const stopTrack = vi.fn()
const closeContext = vi.fn().mockResolvedValue(undefined)
class Context {
  constructor() { order.push("audio") }
  state = "running"
  sampleRate = 48000
  currentTime = 0
  destination = {}
  resume = vi.fn().mockResolvedValue(undefined)
  close = closeContext
  audioWorklet = { addModule: vi.fn().mockResolvedValue(undefined) }
  createMediaStreamSource = () => ({ connect: vi.fn(), disconnect: vi.fn() })
  createAnalyser = () => ({ fftSize: 0, connect: vi.fn(), disconnect: vi.fn(), getFloatTimeDomainData: vi.fn() })
  createGain = () => ({ gain: { value: 0 }, connect: vi.fn(), disconnect: vi.fn() })
}
const getUserMedia = vi.fn()
const tracks: Array<{ stop: typeof stopTrack; enabled: boolean }> = []
const fetchMock = vi.fn()
const requestTts = vi.fn()

const transport = {
  openAsrSocket: () => new WebSocket("wss://widget.example/api/widget/voice/asr", ["xagent-auth-v1"]),
  startFrame: { type: "start", format: "pcm", sample_rate: 16000 },
  requestTts: (text: string, signal: AbortSignal) => requestTts(text, signal),
  classify: classifyWidgetVoiceEvent,
}

const latest = { call: null as VoiceCall | null, openChanges: [] as boolean[], setOpen: null as null | ((open: boolean) => void) }
type HarnessProps = Partial<Omit<WidgetVoiceCallOptions, "open" | "onOpenChange">>
function Harness(props: HarnessProps) {
  const [open, setOpen] = useState(true)
  latest.setOpen = setOpen
  latest.call = useWidgetVoiceCall({
    transport, taskId: null, messages: [], busy: false, waiting: false, disabled: false,
    onSend: vi.fn().mockResolvedValue(undefined), onInterrupt: () => false, ...props,
    open, onOpenChange: next => { latest.openChanges.push(next); setOpen(next) },
  })
  return null
}
const call = () => latest.call!

beforeEach(() => {
  order.length = 0
  Socket.instances = []
  latest.call = null
  latest.openChanges = []
  tracks.length = 0
  getUserMedia.mockImplementation(async () => {
    order.push("microphone")
    const track = { stop: stopTrack, enabled: true }
    tracks.push(track)
    return { getTracks: () => [track] }
  })
  fetchMock.mockImplementation(async () => { order.push("fetch"); return new Response(null, { status: 500 }) })
  requestTts.mockReset()
  vi.stubGlobal("WebSocket", Socket)
  vi.stubGlobal("AudioContext", Context)
  vi.stubGlobal("fetch", fetchMock)
  vi.stubGlobal("AudioWorkletNode", class {
    port = { onmessage: null }
    connect = () => ({ connect: vi.fn() })
    disconnect = vi.fn()
  })
  Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia } })
})
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.clearAllMocks() })

/** Open a call and let recognition start; returns the render and the socket. */
async function connected(props: HarnessProps = {}) {
  const view = render(<Harness {...props} />)
  await waitFor(() => expect(Socket.instances).toHaveLength(1))
  const socket = Socket.instances[0]
  await act(async () => {
    socket.onopen!()
    socket.onmessage!({ data: JSON.stringify({ type: "session_started" }) })
  })
  await waitFor(() => expect(call().phase).toBe("listening"))
  return { view, socket }
}

/** Say something, let it be sent, then get a reply to speak. */
async function speakReply(props: HarnessProps = {}) {
  const { view, socket } = await connected(props)
  act(() => socket.onmessage!({ data: JSON.stringify({ type: "final", text: "Hello", segment_id: 1 }) }))
  await waitFor(() => expect(call().phase).toBe("thinking"))
  const messages: VoiceReplyMessage[] = [{ id: "a1", role: "assistant", content: "Hi there.", isResult: true, status: "completed" }]
  view.rerender(<Harness {...props} messages={messages} />)
  await waitFor(() => expect(requestTts).toHaveBeenCalled())
  return { view, socket }
}

function expectStopped(alert: string, retryable: boolean) {
  expect(call().phase).toBe("error")
  expect(call().alert).toBe(`voiceConversation.widget.${alert}`)
  expect(call().retryable).toBe(retryable)
  expect(call().settingsLink).toBe(false)
  expect(stopTrack).toHaveBeenCalled()
}

async function each<T>(values: T[], run: (value: T) => Promise<void>) {
  for (const value of values) { await run(value); cleanup(); vi.clearAllMocks(); Socket.instances = []; order.length = 0; latest.openChanges = [] }
}

const closes = (codes: Array<[number, string]>, alert: string, retryable: boolean) => () => each(codes, async ([code, reason]) => {
  const { socket } = await connected()
  act(() => socket.onclose!({ code, reason }))
  expectStopped(alert, retryable)
})
const errorFrames = (codes: string[], alert: string, retryable: boolean) => () => each(codes, async code => {
  const { socket } = await connected()
  act(() => socket.onmessage!({ data: JSON.stringify({ type: "error", code, message: "provider secret sk-123" }) }))
  expectStopped(alert, retryable)
  expect(call().alert).not.toContain("sk-123")
})
const synthesis = (statuses: number[], alert: string, retryable: boolean) => () => each(statuses, async status => {
  requestTts.mockResolvedValue(new Response(JSON.stringify({ detail: "quota detail" }), { status }))
  await speakReply()
  await waitFor(() => expect(call().phase).toBe("error"))
  expectStopped(alert, retryable)
})

const scenarios: Record<number, () => Promise<void>> = {
  1: async () => {
    await connected()
    // Audio is unlocked in the click, the microphone is granted, and only then
    // does recognition open.
    expect(order.slice(0, 3)).toEqual(["audio", "microphone", "socket"])
  },
  2: async () => {
    render(<Harness busy />)
    await waitFor(() => expect(call().phase).toBe("error"))
    expect(call().alert).toBe("voiceConversation.widget.tryLater")
    expect(call().retryable).toBe(true)
    expect(getUserMedia).not.toHaveBeenCalled()
  },
  3: async () => {
    getUserMedia.mockRejectedValue(new DOMException("denied", "NotAllowedError"))
    render(<Harness />)
    await waitFor(() => expect(call().phase).toBe("error"))
    expect(call().alert).toBe("voiceConversation.widget.microphoneDenied")
    expect(Socket.instances).toHaveLength(0)
    expect(fetchMock).not.toHaveBeenCalled()
  },
  4: closes([[4003, ""], [4403, ""]], "unavailable", false),
  5: closes([[1008, "Voice is busy"], [1008, "Too many requests"]], "busy", true),
  6: errorFrames(["voice_budget_exhausted"], "temporarilyUnavailable", true),
  7: errorFrames(["voice_unavailable", "model_unavailable", "invalid_request"], "unavailable", false),
  8: async () => {
    await closes([[1011, ""], [1006, ""], [1000, "Speech session timed out"]], "connectionEnded", true)()
    await errorFrames(["stream_failed"], "connectionEnded", true)()
    // A handshake the server refuses (4008 before accept) reaches the page as
    // a failed connection.
    render(<Harness />)
    await waitFor(() => expect(Socket.instances).toHaveLength(1))
    act(() => { Socket.instances[0].onerror!(); Socket.instances[0].onclose?.({ code: 1006, reason: "" }) })
    expectStopped("connectionEnded", true)
  },
  9: closes([[1012, "Service restart"], [4408, ""]], "connectionEnded", true),
  10: errorFrames(["standalone_usage_refused"], "unavailable", false),
  11: errorFrames(["session_timeout"], "sessionTimeout", true),
  12: async () => {
    const onSend = vi.fn().mockRejectedValue(new Error("Raw backend failure"))
    const { socket } = await connected({ onSend })
    act(() => socket.onmessage!({ data: JSON.stringify({ type: "final", text: "Hello", segment_id: 1 }) }))
    await waitFor(() => expect(call().phase).toBe("error"))
    expectStopped("sendFailed", true)
    expect(call().alert).not.toContain("Raw backend failure")
  },
  13: async () => {
    requestTts.mockResolvedValue(new Response(JSON.stringify({ detail: "Speech synthesis failed" }), { status: 502 }))
    await speakReply()
    await waitFor(() => expect(call().note).toBe("voiceConversation.widget.playbackFailed"))
    expect(call().phase).not.toBe("error")
    expect(Socket.instances[0].close).not.toHaveBeenCalled()
  },
  // The Session page refreshes its credential once inside its transport; a
  // 401 that still reaches the call reads as unavailable.
  14: synthesis([401], "unavailable", false),
  15: synthesis([401, 403], "unavailable", false),
  16: synthesis([402, 403], "unavailable", false),
  17: synthesis([503], "unavailable", false),
  18: synthesis([429], "temporarilyUnavailable", true),
  19: async () => {
    const { view, socket } = await connected()
    view.unmount()
    expect(stopTrack).toHaveBeenCalled()
    expect(socket.close).toHaveBeenCalled()
    expect(closeContext).toHaveBeenCalled()
    // Closing the call from the bar releases the same way.
    cleanup(); vi.clearAllMocks(); Socket.instances = []
    const second = await connected()
    act(() => call().end())
    expect(latest.openChanges).toContain(false)
    expect(second.socket.close).toHaveBeenCalled()
    expect(stopTrack).toHaveBeenCalled()
  },
  20: async () => {
    const { view, socket } = await connected()
    view.rerender(<Harness disabled />)
    await waitFor(() => expect(call().phase).toBe("error"))
    expect(socket.close).toHaveBeenCalled()
    expect(call().alert).toBe("voiceConversation.widget.connectionEnded")
    expect(call().retryDisabled).toBe(true)
    expect(stopTrack).toHaveBeenCalled()
  },
  21: () => each([null, 8], async next => {
    const { view, socket } = await connected({ taskId: 7 })
    view.rerender(<Harness taskId={next} />)
    await waitFor(() => expect(latest.openChanges).toEqual([false]))
    expect(socket.close).toHaveBeenCalled()
    expect(call().open).toBe(false)
  }),
  22: async () => {
    const { view } = await connected()
    // Open again, and make the start path run again: muting passes the call
    // through another phase, and busy changes rerun it as well.
    act(() => latest.setOpen!(true))
    act(() => call().toggleMute())
    act(() => call().toggleMute())
    view.rerender(<Harness busy />)
    view.rerender(<Harness />)
    await act(async () => { await Promise.resolve() })
    expect(Socket.instances).toHaveLength(1)
    expect(getUserMedia).toHaveBeenCalledOnce()
    expect(call().phase).toBe("listening")
  },
  23: async () => {
    const { socket } = await connected()
    // The server's periodic recheck closes the socket once voice is off.
    act(() => socket.onclose!({ code: 4003, reason: "" }))
    expectStopped("unavailable", false)
  },
}

describe("widget voice call state table", () => {
  it("has one scenario for each of its 23 rows", () => {
    expect(WIDGET_VOICE_CALL_STATES).toHaveLength(23)
    expect(Object.keys(scenarios).map(Number)).toEqual(WIDGET_VOICE_CALL_STATES.map(row => row.row))
  })

  it.each(WIDGET_VOICE_CALL_STATES.map(row => [row.row, row.event, row] as const))("row %i: %s", async (_row, _event, row) => {
    await scenarios[row.row]()
  })

  it.each(WIDGET_VOICE_CALL_STATES.filter(row => row.match).flatMap(row => row.match!.flatMap(match => {
    const events: VoiceTransportEvent[] = match.kind === "tts_response"
      ? match.statuses.map(status => ({ kind: "tts_response", status, code: undefined }))
      : match.kind === "asr_error"
        ? match.codes.map(code => ({ kind: "asr_error", code }))
        : match.codes.flatMap(code => (match.reasons ?? [""]).map(reason => ({ kind: "asr_close" as const, code, reason })))
    return events.map(event => [row.row, JSON.stringify(event), event, row.failure] as const)
  })))("row %i classifies %s", (_row, _label, event, failure) => {
    expect(classifyWidgetVoiceEvent(event)).toBe(failure)
  })

  it("leaves unknown refusals to the default reading", () => {
    expect(classifyWidgetVoiceEvent({ kind: "asr_close", code: 1008, reason: "Other" })).toBeUndefined()
    expect(classifyWidgetVoiceEvent({ kind: "asr_error", code: undefined })).toBeUndefined()
  })
})

describe("refusals sent as an error frame before the close", () => {
  // The backend explains these in an error frame and then closes; the frame
  // arrives first, so it alone must decide the outcome.
  it.each([
    { code: "voice_busy", close: [1008, "Voice is busy"], alert: "busy", retryable: true },
    { code: "invalid_request", close: [1008, "Invalid speech stream request"], alert: "unavailable", retryable: false },
    { code: "voice_unavailable", close: [1008, "Voice is unavailable"], alert: "unavailable", retryable: false },
    { code: "model_unavailable", close: [1008, "Model unavailable"], alert: "unavailable", retryable: false },
    { code: "voice_budget_exhausted", close: [1008, "Voice budget exhausted"], alert: "temporarilyUnavailable", retryable: true },
    { code: "standalone_usage_refused", close: [1008, "Usage limit reached"], alert: "unavailable", retryable: false },
    { code: "voice_unavailable", close: [4003, ""], alert: "unavailable", retryable: false },
    { code: "voice_unavailable", close: [4403, ""], alert: "unavailable", retryable: false },
    { code: "stream_failed", close: [1011, "Speech stream failed"], alert: "connectionEnded", retryable: true },
    { code: "session_timeout", close: [1000, "Speech session timed out"], alert: "sessionTimeout", retryable: true },
  ] as const)("$code then $close.0 reads as $alert", async ({ code, close, alert, retryable }) => {
    const { socket } = await connected()
    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: "error", code, message: "fixed text" }) })
      socket.onclose?.({ code: close[0], reason: close[1] })
    })
    expectStopped(alert, retryable)
  })
})

describe("widget voice call around the conversation", () => {
  it("opens a new recognition socket when the visitor retries after a failure", async () => {
    const { socket } = await connected()
    act(() => socket.onclose!({ code: 1006, reason: "" }))
    expect(call().phase).toBe("error")
    act(() => call().retry())
    await waitFor(() => expect(Socket.instances).toHaveLength(2))
    expect(getUserMedia).toHaveBeenCalledTimes(2)
    expect(call().alert).toBeUndefined()
  })

  it.each(["end", "unmount"])("never opens recognition when the call is gone before the microphone answers (%s)", async how => {
    let grant: (media: unknown) => void = () => {}
    getUserMedia.mockImplementation(() => new Promise(resolve => { grant = resolve }))
    const view = render(<Harness />)
    await waitFor(() => expect(getUserMedia).toHaveBeenCalledOnce())
    if (how === "end") act(() => call().end())
    else view.unmount()
    const before = latest.call
    const track = { stop: vi.fn(), enabled: true }
    await act(async () => { grant({ getTracks: () => [track] }); await new Promise(resolve => setTimeout(resolve, 0)) })
    expect(Socket.instances).toHaveLength(0)
    expect(track.stop).toHaveBeenCalled()
    expect(latest.call).toBe(before)
    if (how === "end") expect(call().open).toBe(false)
  })

  it("mutes and unmutes the microphone", async () => {
    await connected()
    act(() => call().toggleMute())
    expect(tracks[0].enabled).toBe(false)
    expect(call().muted).toBe(true)
    expect(call().phase).toBe("muted")
    act(() => call().toggleMute())
    expect(tracks[0].enabled).toBe(true)
    expect(call().muted).toBe(false)
    expect(call().phase).toBe("listening")
  })

  it.each([
    { label: "permission refused", setup: () => getUserMedia.mockRejectedValue(new DOMException("denied", "NotAllowedError")) },
    { label: "no microphone", setup: () => getUserMedia.mockRejectedValue(new DOMException("none", "NotFoundError")) },
    { label: "no media devices", setup: () => Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: undefined }) },
  ])("sends nothing to the backend when the microphone fails ($label)", async ({ setup }) => {
    setup()
    const sockets = vi.fn()
    vi.stubGlobal("WebSocket", class extends Socket { constructor(url: string, protocols: string[]) { super(url, protocols); sockets() } })
    render(<Harness />)
    await waitFor(() => expect(call().phase).toBe("error"))
    expect(call().alert).toBe("voiceConversation.widget.microphoneDenied")
    expect(sockets).not.toHaveBeenCalled()
    expect(fetchMock).not.toHaveBeenCalled()
    expect(requestTts).not.toHaveBeenCalled()
  })

  it("keeps the call when the first message gives the conversation its task id", async () => {
    const { view, socket } = await connected({ taskId: null })
    view.rerender(<Harness taskId={42} />)
    view.rerender(<Harness taskId={42} />)
    expect(latest.openChanges).toEqual([])
    expect(socket.close).not.toHaveBeenCalled()
    expect(call().phase).toBe("listening")
  })

  it("holds the words while busy (a reconnect or a first message in flight) and keeps the call", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined)
    const { view, socket } = await connected({ onSend })
    view.rerender(<Harness busy onSend={onSend} />)
    act(() => socket.onmessage!({ data: JSON.stringify({ type: "final", text: "Hello", segment_id: 1 }) }))
    await new Promise(resolve => setTimeout(resolve, 350))
    expect(onSend).not.toHaveBeenCalled()
    view.rerender(<Harness busy={false} onSend={onSend} />)
    await waitFor(() => expect(onSend).toHaveBeenCalledWith("Hello", expect.any(String)))
    expect(socket.close).not.toHaveBeenCalled()
  })

  it("splits synthesis at 1000 characters", async () => {
    requestTts.mockResolvedValue(new Response(null, { status: 502 }))
    const { view, socket } = await connected()
    act(() => socket.onmessage!({ data: JSON.stringify({ type: "final", text: "Hello", segment_id: 1 }) }))
    await waitFor(() => expect(call().phase).toBe("thinking"))
    view.rerender(<Harness messages={[{ id: "a1", role: "assistant", content: "a".repeat(2500), isResult: true, status: "completed" }]} />)
    await waitFor(() => expect(requestTts).toHaveBeenCalledTimes(3))
    expect(requestTts.mock.calls.map(([text]) => (text as string).length)).toEqual([1000, 1000, 500])
  })
})
