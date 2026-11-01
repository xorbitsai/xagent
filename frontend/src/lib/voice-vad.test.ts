import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createVoiceVad } from "./voice-vad"

const state = vi.hoisted(() => ({
  probability: 0.01,
  release: vi.fn(),
  process: vi.fn(),
}))
vi.mock("onnxruntime-web/wasm", () => ({ env: { wasm: {} } }))
vi.mock("@ricky0123/vad-web/dist/models", () => ({ Silero: {
  new: async (_ort: unknown, fetcher: () => Promise<ArrayBuffer>) => {
    await fetcher()
    return { process: state.process, reset_state: vi.fn(), release: state.release }
  },
} }))

beforeEach(() => {
  state.probability = 0.01
  state.process.mockReset().mockImplementation(async () => ({ isSpeech: state.probability, notSpeech: 1 - state.probability }))
  state.release.mockReset().mockResolvedValue(undefined)
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(new ArrayBuffer(8))))
})
afterEach(() => vi.unstubAllGlobals())

describe("Silero probability-based speech confirmation", () => {
  it("ignores loud nonspeech and short candidates, then confirms sustained speech and silence", async () => {
    const speech = vi.fn(), error = vi.fn()
    const vad = await createVoiceVad({ speech, error, signal: new AbortController().signal })
    let count = 0
    const frame = async () => {
      vad.process(new Float32Array(512).fill(0.8)) // Loudness alone is irrelevant.
      count += 1
      await vi.waitFor(() => expect(state.process).toHaveBeenCalledTimes(count))
      for (let step = 0; step < 10; step++) await Promise.resolve()
    }
    for (let index = 0; index < 10; index++) await frame()
    expect(speech).not.toHaveBeenCalled()
    state.probability = 0.9
    await frame() // A single high-probability frame is still only a candidate.
    state.probability = 0.01
    for (let index = 0; index < 14; index++) await frame()
    expect(speech).not.toHaveBeenCalled()
    state.probability = 0.9
    for (let index = 0; index < 5; index++) await frame()
    expect(speech).toHaveBeenCalledExactlyOnceWith(true)
    state.probability = 0.01
    for (let index = 0; index < 14; index++) await frame()
    expect(speech.mock.calls.map(([active]) => active)).toEqual([true, false])
    await vad.destroy(); await vad.destroy()
    expect(state.release).toHaveBeenCalledOnce()
    expect(error).not.toHaveBeenCalled()
  })

  it("drains an in-flight inference before releasing the model and discards late events", async () => {
    let finish!: (result: { isSpeech: number; notSpeech: number }) => void
    state.process.mockImplementation(() => new Promise(resolve => { finish = resolve }))
    const speech = vi.fn()
    const vad = await createVoiceVad({ speech, error: vi.fn(), signal: new AbortController().signal })
    vad.process(new Float32Array(512))
    await vi.waitFor(() => expect(state.process).toHaveBeenCalledOnce())
    const closing = vad.destroy()
    expect(state.release).not.toHaveBeenCalled()
    finish({ isSpeech: 0.99, notSpeech: 0.01 })
    await closing
    expect(state.release).toHaveBeenCalledOnce()
    expect(speech).not.toHaveBeenCalled()
  })
})
