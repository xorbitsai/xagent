import { unwrapFinalAnswerContent } from "./final-answer"

export type VoicePhase = "idle" | "connecting" | "listening" | "thinking" | "speaking" | "muted" | "error"

export function speechText(value: string): string {
  return unwrapFinalAnswerContent(value)
    .replace(/```[\s\S]*?(?:```|$)/g, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/[*_`#>|]/g, "")
    .replace(/\[|\]/g, "")
    .replace(/\s+/g, " ").trim()
}

/** Encode resampled 16 kHz mono frames as raw PCM16 LE for ASR. */
export function encodePcm16(input: Float32Array): ArrayBuffer {
  const bytes = new ArrayBuffer(input.length * 2)
  const view = new DataView(bytes)
  input.forEach((sample, index) => {
    const value = Math.max(-1, Math.min(1, sample))
    view.setInt16(index * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true)
  })
  return bytes
}

/** Preserve device-rate frame boundaries while downsampling to 16 kHz. */
export class VoiceResampler {
  private sum = 0
  private weight = 0
  constructor(private inputRate: number) {
    if (inputRate < 16000) throw new Error("Unsupported microphone sample rate")
  }
  process(input: Float32Array): Float32Array {
    const ratio = this.inputRate / 16000
    const output: number[] = []
    for (const value of input) {
      let left = 1
      while (left > 1e-8) {
        const take = Math.min(left, ratio - this.weight)
        this.sum += value * take; this.weight += take; left -= take
        if (this.weight >= ratio - 1e-8) {
          output.push(this.sum / this.weight)
          this.sum = this.weight = 0
        }
      }
    }
    return Float32Array.from(output)
  }
}

export interface VoiceReply {
  id: string
  text: string
  complete: boolean
}

/** Own a voice turn independently of React renders and late network callbacks. */
export class VoiceTurns {
  private segments: string[] = []
  private segmentIds = new Set<number>()
  private timer: ReturnType<typeof setTimeout> | undefined
  private pending = ""
  private sending = false
  private replyAllowed = false
  private replyOffsets = new Map<string, string>()
  private epoch = 0
  private speaking = false
  private interrupted = false
  private interruptWanted = false
  private userSpeaking = false
  private speechAccepted = false
  private transcriptPending = false
  private stopped = false
  private queue: string[] = []
  private draining = false
  muted = false
  busy = false

  constructor(private actions: {
    send: (text: string) => Promise<void>
    interrupt: () => boolean | void
    stopAudio: () => void
    speak: (text: string) => Promise<void>
    phase: (phase: VoicePhase) => void
    caption: (text: string) => void
    error: (error: unknown) => void
    playbackError: (error: unknown) => void
  }) {}

  activity(active: boolean) {
    if (this.stopped || this.muted) return
    this.userSpeaking = active
    if (active) {
      clearTimeout(this.timer)
    } else this.schedule()
  }

  private cutOff() {
    this.actions.stopAudio()
    this.queue = []
    this.epoch += 1
    this.replyAllowed = false
    this.speaking = false
    this.actions.phase("listening")
  }

  transcript(type: "partial" | "final", text: string, segmentId: number) {
    if (this.stopped || this.muted || !text.trim() || this.segmentIds.has(segmentId)) return
    if (!this.speechAccepted) {
      this.speechAccepted = true
      this.cutOff()
    }
    this.transcriptPending = type === "partial"
    // A transcript is speech confirmation: pause generation only after this,
    // rather than cancelling an agent because of a short microphone noise.
    if ((this.busy || this.sending) && !this.interrupted) {
      this.interruptWanted = true
      this.interrupted = this.actions.interrupt() !== false
    }
    if (type === "final") {
      this.segmentIds.add(segmentId)
      this.segments.push(text.trim())
      this.actions.caption(this.segments.join(" "))
      this.schedule()
    } else {
      clearTimeout(this.timer)
      this.actions.caption([...this.segments, text.trim()].join(" "))
    }
  }

  private schedule() {
    clearTimeout(this.timer)
    if (this.userSpeaking || (!this.segments.length && !this.pending) || this.muted || this.stopped) return
    this.timer = setTimeout(() => {
      this.pending = [this.pending, ...this.segments].filter(Boolean).join(" ")
      this.segments = []
      void this.flush()
    }, 300)
  }

  updateBusy(busy: boolean) {
    this.busy = busy
    if (busy && this.interruptWanted && !this.interrupted) {
      this.interrupted = this.actions.interrupt() !== false
    }
    if (!busy) void this.flush()
  }

  private async flush() {
    if (!this.pending || this.busy || this.sending || this.stopped || this.muted || this.userSpeaking || this.transcriptPending || this.segments.length) return
    const text = this.pending
    this.pending = ""
    this.speechAccepted = false
    this.sending = true
    this.replyOffsets.clear()
    this.replyAllowed = true
    this.interrupted = false
    this.interruptWanted = false
    this.actions.phase("thinking")
    try { await this.actions.send(text) }
    catch (error) { this.replyAllowed = false; this.actions.error(error) }
    finally { this.sending = false }
  }

  reply(reply: VoiceReply) {
    if (!this.replyAllowed || this.stopped) return
    const text = speechText(reply.text)
    if (!text || text.startsWith("{")) return
    const previous = this.replyOffsets.get(reply.id) || ""
    if (text.length < previous.length) return
    // Streaming markdown can revise an earlier prefix. Keep the spoken
    // position without replaying it or blocking all subsequent sentences.
    let consumed = text.slice(0, previous.length)
    let rest = text.slice(consumed.length)
    while (rest) {
      const sentence = rest.match(/^[\s\S]*?(?:[.!?]\s+|[。！？]\s*)/)?.[0]
      const chunk = sentence || (reply.complete ? rest : "")
      if (!chunk) break
      // Each synthesis request stays within the transport's text limit.
      for (let index = 0; index < chunk.length; index += 4000) this.queue.push(chunk.slice(index, index + 4000))
      consumed += chunk
      rest = rest.slice(chunk.length)
    }
    this.replyOffsets.set(reply.id, consumed)
    void this.drain()
  }

  private async drain() {
    if (this.draining || this.stopped) return
    this.draining = true
    try {
      while (this.queue.length && !this.stopped) {
        const epoch = this.epoch
        const text = this.queue.shift()!
        this.speaking = true
        this.actions.phase("speaking")
        try { await this.actions.speak(text) }
        catch (error) {
          if (epoch === this.epoch && !this.stopped && !(error instanceof Error && error.name === "AbortError")) this.actions.playbackError(error)
        }
        if (epoch === this.epoch) this.speaking = false
      }
    } finally {
      this.draining = false
      if (!this.stopped && !this.speaking) this.actions.phase(this.muted ? "muted" : this.userSpeaking ? "listening" : this.busy ? "thinking" : "listening")
    }
  }

  mute(muted: boolean) {
    this.muted = muted
    this.segments = []
    this.pending = ""
    this.userSpeaking = false
    this.speechAccepted = false
    this.transcriptPending = false
    clearTimeout(this.timer)
    this.actions.phase(muted ? "muted" : this.busy ? "thinking" : "listening")
  }

  stop() {
    this.stopped = true
    this.epoch += 1
    this.queue = []
    this.pending = ""
    clearTimeout(this.timer)
    this.actions.stopAudio()
  }
}
