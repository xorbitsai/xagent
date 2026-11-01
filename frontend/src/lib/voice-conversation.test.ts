import { afterEach, describe, expect, it, vi } from "vitest"
import { encodePcm16, VoiceResampler, VoiceTurns, speechText } from "./voice-conversation"

afterEach(() => vi.useRealTimers())

function coordinator() {
  const actions = {
    send: vi.fn().mockResolvedValue(undefined), interrupt: vi.fn(), stopAudio: vi.fn(),
    speak: vi.fn().mockResolvedValue(undefined), phase: vi.fn(), caption: vi.fn(), error: vi.fn(),
  }
  return { actions, turns: new VoiceTurns(actions) }
}

describe("voice turn ownership", () => {
  it("coalesces VAD segments, never sends partials, and ignores repeated finals", async () => {
    vi.useFakeTimers()
    const { turns, actions } = coordinator()
    turns.activity(true)
    turns.transcript("partial", "Hello", 0)
    await vi.advanceTimersByTimeAsync(500)
    expect(actions.send).not.toHaveBeenCalled()
    turns.transcript("final", "Hello.", 0)
    turns.transcript("final", "How are you?", 1)
    turns.activity(false)
    await vi.advanceTimersByTimeAsync(300)
    expect(actions.send).toHaveBeenCalledExactlyOnceWith("Hello. How are you?")
    turns.updateBusy(true)
    turns.transcript("final", "How are you?", 1)
    expect(actions.interrupt).not.toHaveBeenCalled()
    turns.stop()
  })

  it("automatically cuts off speech, interrupts confirmed generation, and rejects late old replies", async () => {
    vi.useFakeTimers()
    const { turns, actions } = coordinator()
    let abortAudio: (() => void) | undefined
    actions.speak.mockImplementationOnce(() => new Promise<void>((_, reject) => {
      abortAudio = () => reject(new DOMException("Aborted", "AbortError"))
    }))
    actions.stopAudio.mockImplementation(() => abortAudio?.())
    turns.activity(true)
    turns.transcript("final", "First question.", 0)
    turns.activity(false)
    await vi.advanceTimersByTimeAsync(300)
    turns.updateBusy(true)
    turns.reply({ id: "old", text: "First sentence. More old speech.", complete: true })
    expect(actions.speak).toHaveBeenCalledExactlyOnceWith("First sentence. ")
    const previousStops = actions.stopAudio.mock.calls.length
    turns.activity(true)
    expect(actions.stopAudio).toHaveBeenCalledTimes(previousStops + 1)
    expect(actions.interrupt).not.toHaveBeenCalled()
    turns.transcript("partial", "Actually", 1)
    turns.transcript("partial", "Actually change that", 1)
    expect(actions.interrupt).toHaveBeenCalledOnce()
    turns.reply({ id: "old", text: "First sentence. More old speech. Too late.", complete: true })
    turns.transcript("final", "Actually change that.", 1)
    turns.activity(false)
    await vi.advanceTimersByTimeAsync(300)
    expect(actions.send).toHaveBeenCalledTimes(1)
    turns.updateBusy(false) // The old run has acknowledged its interruption.
    await vi.advanceTimersByTimeAsync(0)
    expect(actions.send).toHaveBeenLastCalledWith("Actually change that.")
    turns.reply({ id: "new", text: "New answer.", complete: true })
    await vi.advanceTimersByTimeAsync(0)
    expect(actions.speak.mock.calls.map(([text]) => text)).toEqual(["First sentence. ", "New answer."])
    expect(actions.error).not.toHaveBeenCalled()
    turns.stop()
  })

  it("begins speaking completed sentences before the whole answer ends, without replaying them", async () => {
    vi.useFakeTimers()
    const { turns, actions } = coordinator()
    turns.activity(true)
    turns.transcript("final", "Hi.", 0)
    turns.activity(false)
    await vi.advanceTimersByTimeAsync(300)
    turns.reply({ id: "answer", text: "First sentence. Sec", complete: false })
    await vi.advanceTimersByTimeAsync(0)
    expect(actions.speak).toHaveBeenCalledExactlyOnceWith("First sentence. ")
    turns.reply({ id: "answer", text: "First sentence. Second sentence.", complete: true })
    turns.reply({ id: "answer", text: "First sentence. Second sentence.", complete: true })
    await vi.advanceTimersByTimeAsync(0)
    expect(actions.speak.mock.calls.map(([text]) => text)).toEqual(["First sentence. ", "Second sentence."])
    turns.stop()
  })

  it("does not send muted, unfinished, or ended-call transcripts", async () => {
    vi.useFakeTimers()
    const { turns, actions } = coordinator()
    turns.activity(true)
    turns.transcript("final", "Unfinished", 0)
    turns.activity(false)
    turns.mute(true)
    turns.transcript("final", "Muted", 1)
    await vi.advanceTimersByTimeAsync(500)
    expect(actions.send).not.toHaveBeenCalled()
    turns.mute(false)
    turns.activity(true)
    turns.transcript("final", "After unmute", 2)
    turns.stop()
    await vi.advanceTimersByTimeAsync(500)
    turns.transcript("final", "After ending", 3)
    turns.reply({ id: "late", text: "Old reply.", complete: true })
    expect(actions.send).not.toHaveBeenCalled()
    expect(actions.speak).not.toHaveBeenCalled()
  })

  it("retains an interruption that arrives before the submitted task starts", async () => {
    vi.useFakeTimers()
    const { turns, actions } = coordinator()
    actions.interrupt.mockReturnValue(false)
    turns.updateBusy(true)
    turns.activity(true)
    turns.transcript("partial", "Wait", 0)
    expect(actions.interrupt).toHaveBeenCalledTimes(1)
    actions.interrupt.mockReturnValue(true)
    turns.updateBusy(true)
    expect(actions.interrupt).toHaveBeenCalledTimes(2)
    turns.updateBusy(true)
    expect(actions.interrupt).toHaveBeenCalledTimes(2)
    turns.stop()
  })
})

describe("audio format and activity", () => {
  it.each([44100, 48000, 16000])("preserves duration across %i Hz worklet block boundaries", rate => {
    const input = new Float32Array(rate).fill(0.5)
    const encoder = new VoiceResampler(rate)
    const bytes: number[] = []
    for (let offset = 0; offset < input.length; offset += 1024) bytes.push(...new Uint8Array(encodePcm16(encoder.process(input.slice(offset, offset + 1024)))))
    expect(bytes).toHaveLength(32000)
    const view = new DataView(Uint8Array.from(bytes).buffer)
    expect(view.getInt16(0, true)).toBe(16384)
    expect(view.getInt16(31998, true)).toBe(16384)
  })

  it("cannot interrupt or submit an ASR frame without VAD speech confirmation", async () => {
    vi.useFakeTimers()
    const { turns, actions } = coordinator()
    turns.updateBusy(true)
    turns.transcript("partial", "Background noise", 0)
    turns.transcript("final", "Background noise", 0)
    await vi.advanceTimersByTimeAsync(500)
    expect(actions.interrupt).not.toHaveBeenCalled()
    expect(actions.stopAudio).not.toHaveBeenCalled()
    expect(actions.send).not.toHaveBeenCalled()
    turns.stop()
  })

  it("reads final answer text while omitting code blocks, image URLs and markdown", () => {
    expect(speechText('**Hello** [there](https://example.com). ```python\nsecret()\n```')).toBe("Hello there.")
    expect(speechText('{"action":"final_answer","action_input":"Hello."}')).toBe("Hello.")
  })
})
