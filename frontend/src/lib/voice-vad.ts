export interface VoiceVad {
  process: (samples: Float32Array) => void
  destroy: () => Promise<void>
}

/** Report Silero speech activity on 32 ms mono frames. */
export async function createVoiceVad(options: {
  speech: (speaking: boolean) => void
  error: () => void
  signal: AbortSignal
}): Promise<VoiceVad> {
  const [ort, { FrameProcessor }, { Message }, { Silero }] = await Promise.all([
    import("onnxruntime-web/wasm"),
    import("@ricky0123/vad-web/dist/frame-processor"),
    import("@ricky0123/vad-web/dist/messages"),
    import("@ricky0123/vad-web/dist/models"),
  ])
  ort.env.wasm.wasmPaths = "/voice-vad/"
  ort.env.wasm.numThreads = 1
  ort.env.wasm.proxy = false
  const model = await Silero.new(ort, async () => {
    const response = await fetch("/voice-vad/silero_vad_v5.onnx", { signal: options.signal })
    if (!response.ok) throw new Error("VAD model unavailable")
    return response.arrayBuffer()
  })
  if (options.signal.aborted) {
    await model.release()
    throw new DOMException("Aborted", "AbortError")
  }
  const processor = new FrameProcessor(model.process, model.reset_state, {
    positiveSpeechThreshold: 0.6, negativeSpeechThreshold: 0.35,
    minSpeechMs: 160, redemptionMs: 400, preSpeechPadMs: 160,
    submitUserSpeechOnPause: false,
  }, 32)
  processor.resume()
  let pending = new Float32Array(0)
  let queued = 0
  let closed = false
  let tail = Promise.resolve()
  let closing: Promise<void> | undefined
  return {
    process(samples) {
      if (closed) return
      const combined = new Float32Array(pending.length + samples.length)
      combined.set(pending); combined.set(samples, pending.length)
      let offset = 0
      for (; offset + 512 <= combined.length; offset += 512) {
        if (++queued > 16) { options.error(); return }
        const frame = combined.slice(offset, offset + 512)
        tail = tail.then(async () => {
          if (closed) return
          await processor.process(frame, event => {
            if (closed) return
            if (event.msg === Message.SpeechRealStart) options.speech(true)
            if (event.msg === Message.SpeechEnd) options.speech(false)
          })
        }).catch(() => { if (!closed) options.error() }).finally(() => { queued -= 1 })
      }
      pending = combined.slice(offset)
    },
    destroy() {
      closed = true
      pending = new Float32Array(0)
      closing ??= tail.then(() => model.release())
      return closing
    },
  }
}
