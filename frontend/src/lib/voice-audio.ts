import { readAuthSessionSnapshot } from "./auth-cache"
import { getApiUrl, getWsUrl } from "./utils"
import { websocketAuthProtocols } from "./websocket-auth"
import { encodePcm16, VoiceResampler } from "./voice-conversation"
import { createVoiceVad, type VoiceVad } from "./voice-vad"

export type VoiceAudioFailure = "session_timeout" | "authentication_required" | "connection_failed"
export class VoiceAudioError extends Error {
  constructor(public code: VoiceAudioFailure) { super(code) }
}

/** One call owns its microphone, socket, playback sources and fetches. */
export class VoiceAudio {
  private closed = false
  private context: AudioContext | null = null
  private microphone: MediaStream | null = null
  private capture: AudioWorkletNode | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private sink: GainNode | null = null
  private socket: WebSocket | null = null
  private playback = new Set<AudioBufferSourceNode>()
  private synthesis: AbortController | null = null
  private vad: VoiceVad | null = null
  private serverSpeaking = false
  private vadLoading = new AbortController()
  private rejectReady: ((error: Error) => void) | null = null
  private readyTimeout: ReturnType<typeof setTimeout> | undefined
  private _muted = false
  get muted() { return this._muted }
  set muted(value: boolean) {
    this._muted = value
    this.microphone?.getTracks().forEach(track => { track.enabled = !value })
  }

  constructor(private options: {
    asr: string
    tts: string
    voice?: string
    token: string
    sessionId: string | null
    activity: (active: boolean) => void
    transcript: (type: "partial" | "final", text: string, segmentId: number) => void
    error: (reason?: VoiceAudioFailure) => void
    vadFallback?: () => void
  }) {}

  async start(): Promise<void> {
    this.context = new AudioContext()
    // Resume during the start button's user activation, before network awaits.
    await this.context.resume()
    if (this.closed) return
    const context = this.context
    const media = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    })
    if (this.closed) { media.getTracks().forEach(track => track.stop()); return }
    this.microphone = media
    await context.audioWorklet.addModule("/voice-capture-worklet.js")
    if (this.closed) return
    // Backend ASR/VAD owns the baseline protocol. Local activity refines turn
    // boundaries and must not block a call or SDK parity.
    void createVoiceVad({
      signal: this.vadLoading.signal,
      speech: speaking => { if (!this.closed && !this.muted) this.options.activity(speaking) },
      error: () => this.useServerVad(),
    }).then(async vad => {
      if (this.closed) { await vad.destroy(); return }
      if (this.serverSpeaking) { this.serverSpeaking = false; this.options.activity(false) }
      this.vad = vad
    }).catch(() => this.useServerVad())
    const socket = new WebSocket(`${getWsUrl().replace(/\/$/, "")}/api/models/speech/asr/stream`, websocketAuthProtocols(this.options.token))
    this.socket = socket
    await new Promise<void>((resolve, reject) => {
      this.rejectReady = reject
      this.readyTimeout = setTimeout(() => this.fail("connection_failed"), 30000)
      socket.onopen = () => socket.send(JSON.stringify({
        type: "start", model_id: this.options.asr, format: "pcm", sample_rate: 16000,
        vad: true, vad_silence_threshold_secs: 0.5,
      }))
      socket.onmessage = ({ data }) => {
        if (this.closed) return
        try {
          const event = JSON.parse(data)
          if (event.type === "session_started") {
            clearTimeout(this.readyTimeout)
            this.rejectReady = null
            resolve()
          } else if ((event.type === "partial" || event.type === "final") && typeof event.text === "string") {
            if (!this.muted) {
              if (!this.vad && event.text.trim() && !this.serverSpeaking) {
                this.serverSpeaking = true
                this.options.activity(true)
              }
              this.options.transcript(event.type, event.text, event.segment_id)
              if (!this.vad && event.type === "final") {
                this.serverSpeaking = false
                this.options.activity(false)
              }
            }
          } else if (event.type === "error") this.fail(event.code === "session_timeout" ? "session_timeout" : "connection_failed")
        } catch { this.fail("connection_failed") }
      }
      socket.onerror = () => this.fail("connection_failed")
      socket.onclose = event => this.fail(event.code === 4001 ? "authentication_required" : event.reason === "Speech session timed out" ? "session_timeout" : "connection_failed")
    })
    if (this.closed) return
    const resampler = new VoiceResampler(context.sampleRate)
    this.source = context.createMediaStreamSource(media)
    this.capture = new AudioWorkletNode(context, "xagent-voice-capture", { channelCount: 1, channelCountMode: "explicit" })
    this.sink = context.createGain()
    this.sink.gain.value = 0
    this.capture.port.onmessage = ({ data }: MessageEvent<Float32Array>) => {
      if (this.closed || socket.readyState !== WebSocket.OPEN) return
      if (socket.bufferedAmount > 256 * 1024) { this.fail("connection_failed"); return }
      const samples = this.muted ? new Float32Array(data.length) : data
      const frame = resampler.process(samples)
      this.vad?.process(frame)
      socket.send(encodePcm16(frame))
    }
    this.source.connect(this.capture)
    this.capture.connect(this.sink).connect(context.destination)
  }

  private fail(reason: VoiceAudioFailure) {
    if (this.closed) return
    const reject = this.rejectReady
    this.rejectReady = null
    reject?.(new VoiceAudioError(reason))
    this.close()
    this.options.error(reason)
  }

  private useServerVad() {
    const vad = this.vad
    this.vad = null
    if (vad) {
      void vad.destroy().catch(() => {})
      if (!this.closed) this.options.activity(false)
    }
    if (!this.closed) this.options.vadFallback?.()
  }

  stopAudio() {
    this.synthesis?.abort()
    this.synthesis = null
    for (const source of this.playback) { try { source.stop() } catch { /* ended */ } source.disconnect() }
    this.playback.clear()
  }

  async speak(text: string): Promise<void> {
    const context = this.context
    if (!context || this.closed) return
    const controller = new AbortController()
    this.synthesis = controller
    const signal = controller.signal
    const login = readAuthSessionSnapshot()
    const token = login.accessToken
    if (!token || login.sessionId !== this.options.sessionId) throw new VoiceAudioError("authentication_required")
    // A synthesis request is billable; never automatically replay a failed POST.
    const response = await fetch(`${getApiUrl()}/api/models/speech/tts/stream`, {
      method: "POST", signal,
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: this.options.tts, voice: this.options.voice, text, format: "pcm", sample_rate: 24000 }),
    })
    if (!response.ok || !response.body) {
      await response.body?.cancel()
      if (response.status === 401 || response.status === 403) throw new VoiceAudioError("authentication_required")
      throw new Error("Speech synthesis failed")
    }
    const rate = Number(response.headers.get("X-Audio-Sample-Rate"))
    if (response.headers.get("X-Audio-Format") !== "pcm" || !Number.isFinite(rate) || rate < 8000 || rate > 48000) {
      await response.body.cancel()
      throw new Error("Invalid speech audio format")
    }
    const reader = response.body.getReader()
    let pending = new Uint8Array(0)
    let next = context.currentTime + 0.025
    const wait = () => new Promise<void>((resolve, reject) => {
      if (signal.aborted) { reject(new DOMException("Aborted", "AbortError")); return }
      const abort = () => { clearTimeout(timer); reject(new DOMException("Aborted", "AbortError")) }
      const timer = setTimeout(() => { signal.removeEventListener("abort", abort); resolve() }, 20)
      signal.addEventListener("abort", abort, { once: true })
    })
    try {
      while (!signal.aborted && !this.closed) {
        const { done, value } = await reader.read()
        if (done) break
        const bytes = new Uint8Array(pending.length + value.length)
        bytes.set(pending); bytes.set(value, pending.length)
        const aligned = bytes.length - bytes.length % 2
        const view = new DataView(bytes.buffer)
        for (let offset = 0; offset < aligned; offset += 24000) {
          while (next - context.currentTime > 1) await wait()
          if (signal.aborted || this.closed) return
          const length = Math.min(24000, aligned - offset) / 2
          const buffer = context.createBuffer(1, length, rate)
          const samples = buffer.getChannelData(0)
          for (let index = 0; index < length; index++) samples[index] = view.getInt16(offset + index * 2, true) / 32768
          const source = context.createBufferSource()
          source.buffer = buffer
          source.connect(context.destination)
          source.onended = () => { source.disconnect(); this.playback.delete(source) }
          this.playback.add(source)
          next = Math.max(next, context.currentTime + 0.01)
          source.start(next)
          next += length / rate
        }
        pending = bytes.slice(aligned)
      }
      if (pending.length && !signal.aborted) throw new Error("Truncated PCM sample")
      while (!signal.aborted && !this.closed && this.playback.size) await wait()
    } catch (error) {
      if (this.synthesis === controller) this.stopAudio()
      throw error
    } finally {
      await reader.cancel().catch(() => {})
      reader.releaseLock()
      if (this.synthesis === controller) this.synthesis = null
    }
  }

  close() {
    if (this.closed) return
    this.closed = true
    this.vadLoading.abort()
    void this.vad?.destroy().catch(() => {})
    clearTimeout(this.readyTimeout)
    this.rejectReady?.(new Error("Call ended"))
    this.rejectReady = null
    this.stopAudio()
    if (this.socket) { this.socket.onclose = this.socket.onerror = this.socket.onmessage = this.socket.onopen = null; this.socket.close() }
    this.capture?.disconnect(); this.source?.disconnect(); this.sink?.disconnect()
    this.microphone?.getTracks().forEach(track => track.stop())
    void this.context?.close().catch(() => {})
  }
}
