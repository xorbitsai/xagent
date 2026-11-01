import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { VoiceAudio } from "./voice-audio"
import { createVoiceVad } from "./voice-vad"

const detectors = vi.hoisted(() => [] as Array<{ options: { speech: (active: boolean) => void; error: () => void }; process: ReturnType<typeof vi.fn>; destroy: ReturnType<typeof vi.fn> }>)
vi.mock("./voice-vad", () => ({ createVoiceVad: vi.fn(async options => {
  const vad = { options, process: vi.fn(), destroy: vi.fn().mockResolvedValue(undefined) }
  detectors.push(vad)
  return vad
}) }))

vi.mock("./auth-cache", () => ({ readAuthSessionSnapshot: () => ({ accessToken: "current-login-token", sessionId: "login" }) }))
vi.mock("./utils", () => ({ getWsUrl: () => "wss://example.com", getApiUrl: () => "https://example.com" }))

class Socket {
  static OPEN = 1
  static instances: Socket[] = []
  readyState = 1
  bufferedAmount = 0
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  send = vi.fn()
  close = vi.fn()
  constructor(public url: string, public protocols: string[]) { Socket.instances.push(this) }
}
const captures: Array<{ port: { onmessage: ((event: { data: Float32Array }) => void) | null } }> = []
const playback: Array<{ start: ReturnType<typeof vi.fn>; stop: ReturnType<typeof vi.fn>; onended: (() => void) | null }> = []
const stopTrack = vi.fn()
const closeContext = vi.fn().mockResolvedValue(undefined)
class Context {
  sampleRate = 48000
  currentTime = 0
  destination = {}
  resume = vi.fn().mockResolvedValue(undefined)
  close = closeContext
  audioWorklet = { addModule: vi.fn().mockResolvedValue(undefined) }
  createMediaStreamSource = () => ({ connect: vi.fn(), disconnect: vi.fn() })
  createGain = () => ({ gain: { value: 0 }, connect: vi.fn(), disconnect: vi.fn() })
  createBuffer = (_channels: number, length: number) => ({ getChannelData: () => new Float32Array(length) })
  createBufferSource = () => {
    const source = { buffer: null, connect: vi.fn(), disconnect: vi.fn(), start: vi.fn(), stop: vi.fn(), onended: null as (() => void) | null }
    playback.push(source)
    return source
  }
}

const options = () => ({ asr: "live", tts: "speaker", token: "login-token", sessionId: "login", activity: vi.fn(), transcript: vi.fn(), error: vi.fn(), vadFallback: vi.fn() })

beforeEach(() => {
  Socket.instances = []; captures.length = 0; playback.length = 0
  detectors.length = 0
  vi.stubGlobal("WebSocket", Socket)
  vi.stubGlobal("AudioContext", Context)
  vi.stubGlobal("AudioWorkletNode", class {
    port = { onmessage: null }
    constructor() { captures.push(this) }
    connect = () => ({ connect: vi.fn() })
    disconnect = vi.fn()
  })
  Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: {
    getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop: stopTrack }] }),
  } })
})
afterEach(() => { vi.unstubAllGlobals(); vi.clearAllMocks() })

async function start(call: VoiceAudio) {
  const opening = call.start()
  for (let step = 0; step < 20 && !Socket.instances.length; step++) await Promise.resolve()
  const socket = Socket.instances[0]
  socket.onopen!()
  socket.onmessage!({ data: JSON.stringify({ type: "session_started" }) })
  await opening
  return socket
}

describe("browser voice resources", () => {
  it("uses the backend ASR/VAD protocol when the local detector cannot load", async () => {
    vi.mocked(createVoiceVad).mockRejectedValueOnce(new Error("WASM unavailable"))
    const events = options()
    const call = new VoiceAudio(events)
    const socket = await start(call)
    expect(JSON.parse(socket.send.mock.calls[0][0])).toMatchObject({ vad: true })
    expect(events.vadFallback).toHaveBeenCalledOnce()
    captures[0].port.onmessage!({ data: new Float32Array(1024).fill(0.9) })
    expect(events.activity).not.toHaveBeenCalled()
    socket.onmessage!({ data: JSON.stringify({ type: "partial", text: "Hello", segment_id: 1 }) })
    socket.onmessage!({ data: JSON.stringify({ type: "final", text: "Hello there.", segment_id: 1 }) })
    expect(events.activity.mock.calls).toEqual([[true], [false]])
    expect(events.transcript).toHaveBeenLastCalledWith("final", "Hello there.", 1)
    expect(events.error).not.toHaveBeenCalled()
    call.close()
  })

  it("starts without waiting for local VAD and releases a detector arriving after close", async () => {
    let release!: (vad: Awaited<ReturnType<typeof createVoiceVad>>) => void
    vi.mocked(createVoiceVad).mockReturnValueOnce(new Promise(resolve => { release = resolve }))
    const events = options()
    const call = new VoiceAudio(events)
    const socket = await start(call)
    socket.onmessage!({ data: JSON.stringify({ type: "final", text: "Server VAD works.", segment_id: 1 }) })
    expect(events.activity.mock.calls).toEqual([[true], [false]])
    call.close()
    const detector = { process: vi.fn(), destroy: vi.fn().mockResolvedValue(undefined) }
    release(detector)
    await Promise.resolve()
    expect(detector.destroy).toHaveBeenCalledOnce()
  })

  it("keeps streaming after a running local detector fails", async () => {
    const events = options()
    const call = new VoiceAudio(events)
    const socket = await start(call)
    detectors[0].options.error()
    expect(detectors[0].destroy).toHaveBeenCalledOnce()
    socket.onmessage!({ data: JSON.stringify({ type: "final", text: "Continue.", segment_id: 1 }) })
    expect(events.transcript).toHaveBeenCalledWith("final", "Continue.", 1)
    expect(events.vadFallback).toHaveBeenCalledOnce()
    expect(events.error).not.toHaveBeenCalled()
    call.close()
  })

  it("finishes server speech activity when a local detector joins mid-utterance", async () => {
    let release!: (vad: Awaited<ReturnType<typeof createVoiceVad>>) => void
    vi.mocked(createVoiceVad).mockReturnValueOnce(new Promise(resolve => { release = resolve }))
    const events = options()
    const call = new VoiceAudio(events)
    const socket = await start(call)
    socket.onmessage!({ data: JSON.stringify({ type: "partial", text: "Hello", segment_id: 1 }) })
    release({ process: vi.fn(), destroy: vi.fn().mockResolvedValue(undefined) })
    await Promise.resolve()
    socket.onmessage!({ data: JSON.stringify({ type: "final", text: "Hello there.", segment_id: 1 }) })
    expect(events.activity.mock.calls).toEqual([[true], [false]])
    expect(events.transcript).toHaveBeenLastCalledWith("final", "Hello there.", 1)
    call.close()
  })
  it("uses header login, mono PCM capture, local activity, and releases every resource", async () => {
    const callbacks = options()
    const call = new VoiceAudio(callbacks)
    const socket = await start(call)
    expect(socket.url).not.toContain("token")
    expect(socket.protocols).toEqual(["xagent-auth-v1", "xagent-auth-token.login-token"])
    expect(JSON.parse(socket.send.mock.calls[0][0])).toMatchObject({ model_id: "live", sample_rate: 16000, vad: true })
    for (let frame = 0; frame < 4; frame++) captures[0].port.onmessage!({ data: new Float32Array(1024).fill(0.1) })
    expect(callbacks.activity).not.toHaveBeenCalled()
    detectors[0].options.speech(true)
    expect(callbacks.activity).toHaveBeenCalledWith(true)
    expect(socket.send.mock.calls[1][0]).toBeInstanceOf(ArrayBuffer)
    call.close(); call.close()
    expect(socket.close).toHaveBeenCalledOnce()
    expect(stopTrack).toHaveBeenCalledOnce()
    expect(closeContext).toHaveBeenCalledOnce()
    expect(detectors[0].destroy).toHaveBeenCalledOnce()
  })

  it("stops a microphone granted after the user has already left", async () => {
    let grant!: (media: MediaStream) => void
    vi.mocked(navigator.mediaDevices.getUserMedia).mockImplementation(() => new Promise<MediaStream>(resolve => { grant = resolve }))
    const call = new VoiceAudio(options())
    const opening = call.start()
    await Promise.resolve()
    call.close()
    grant({ getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream)
    await opening
    expect(stopTrack).toHaveBeenCalledOnce()
    expect(Socket.instances).toHaveLength(0)
  })

  it("plays the first PCM chunk before completion and aborts the fetch and scheduled audio on interruption", async () => {
    const call = new VoiceAudio({ ...options(), voice: "selected-voice" })
    await start(call)
    let stream!: ReadableStreamDefaultController<Uint8Array>
    let signal!: AbortSignal
    const body = new ReadableStream<Uint8Array>({ start(controller) { stream = controller } })
    vi.stubGlobal("fetch", vi.fn((_url: string, request: RequestInit) => {
      signal = request.signal as AbortSignal
      signal.addEventListener("abort", () => stream.error(new DOMException("Aborted", "AbortError")), { once: true })
      return Promise.resolve(new Response(body, { headers: { "X-Audio-Format": "pcm", "X-Audio-Sample-Rate": "24000" } }))
    }))
    const playing = call.speak("Hello.")
    const settled = playing.catch(error => error)
    stream.enqueue(new Uint8Array([0, 0, 1, 0]))
    for (let step = 0; step < 30 && !playback.length; step++) await Promise.resolve()
    expect(playback[0].start).toHaveBeenCalledOnce()
    expect(signal.aborted).toBe(false)
    call.stopAudio()
    expect(signal.aborted).toBe(true)
    expect(playback[0].stop).toHaveBeenCalledOnce()
    expect((await settled).name).toBe("AbortError")
    expect(vi.mocked(fetch)).toHaveBeenCalledOnce()
    expect(JSON.parse(vi.mocked(fetch).mock.calls[0][1]!.body as string)).toMatchObject({ model_id: "speaker", voice: "selected-voice" })
    call.close()
  })
})
