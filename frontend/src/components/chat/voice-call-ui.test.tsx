import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))

import { VoiceCallBar, VoiceOrb, VoiceWaveform } from "./voice-call-ui"
import type { VoiceCall } from "@/hooks/use-voice-call"

const call = (over: Partial<VoiceCall> = {}): VoiceCall => ({
  open: true, phase: "listening", active: true, caption: "", muted: false,
  status: "voiceConversation.states.listening", settingsLink: false, retryable: true, retryDisabled: false,
  level: () => 0, retry: vi.fn(), end: vi.fn(), toggleMute: vi.fn(), ...over,
})

beforeEach(() => { vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null) })
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.useRealTimers() })

describe("voice call bar", () => {
  it("ends the call on Escape, as the dialog it replaced did", () => {
    const active = call()
    render(<VoiceCallBar call={active} />)
    fireEvent.keyDown(document, { key: "Escape" })
    expect(active.end).toHaveBeenCalledOnce()
    fireEvent.keyDown(document, { key: "Enter" })
    expect(active.end).toHaveBeenCalledOnce()
  })

  it("stops listening for Escape once it is gone", () => {
    const active = call()
    const view = render(<VoiceCallBar call={active} />)
    view.unmount()
    fireEvent.keyDown(document, { key: "Escape" })
    expect(active.end).not.toHaveBeenCalled()
  })

  it("takes focus when focus would otherwise be lost, but not from somewhere else", () => {
    const view = render(<VoiceCallBar call={call()} />)
    expect(screen.getByRole("group", { name: "voiceConversation.title" })).toHaveFocus()
    view.unmount()
    const elsewhere = document.createElement("input")
    document.body.appendChild(elsewhere)
    elsewhere.focus()
    render(<VoiceCallBar call={call()} />)
    expect(elsewhere).toHaveFocus()
    elsewhere.remove()
  })

  it("announces what is being said", () => {
    render(<VoiceCallBar call={call({ caption: "nothing much" })} />)
    expect(screen.getByTestId("voice-caption")).toHaveAttribute("aria-live", "polite")
  })

  it("hides the retry button when trying again cannot help and disables it while a run is going", () => {
    const view = render(<VoiceCallBar call={call({ phase: "error", active: false, retryable: false, alert: "Add a model." })} />)
    expect(screen.queryByRole("button", { name: "voiceConversation.retry" })).not.toBeInTheDocument()
    view.rerender(<VoiceCallBar call={call({ phase: "error", active: false, retryDisabled: true, alert: "Dropped." })} />)
    expect(screen.getByRole("button", { name: "voiceConversation.retry" })).toBeDisabled()
  })

  it("draws without roundRect support", () => {
    const context = { clearRect: vi.fn(), beginPath: vi.fn(), fill: vi.fn(), arc: vi.fn(), rect: vi.fn(), setTransform: vi.fn(), fillStyle: "" }
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context as unknown as CanvasRenderingContext2D)
    vi.useFakeTimers()
    vi.spyOn(window, "requestAnimationFrame").mockImplementation(callback => window.setTimeout(() => callback(performance.now() + 100), 16))
    render(<VoiceCallBar call={call({ level: () => 0.8 })} />)
    expect(() => vi.advanceTimersByTime(200)).not.toThrow()
    expect(context.rect).toHaveBeenCalled()
    vi.useRealTimers()
  })

  it("shows the note beside what is being said in both layouts", () => {
    const withNote = call({ caption: "nothing much", note: "Using server voice detection." })
    const view = render(<VoiceCallBar call={withNote} />)
    expect(screen.getByTestId("voice-note")).toHaveTextContent("Using server voice detection.")
    expect(screen.getByTestId("voice-caption")).toHaveTextContent("nothing much")
    view.rerender(<VoiceCallBar call={withNote} compact />)
    expect(screen.getByTestId("voice-note")).toHaveTextContent("Using server voice detection.")
    // And a note on its own still shows.
    view.rerender(<VoiceCallBar call={call({ note: "Part of the reply could not be played." })} compact />)
    expect(screen.getByTestId("voice-note")).toBeInTheDocument()
    expect(screen.queryByTestId("voice-caption")).not.toBeInTheDocument()
  })

  it("lets the page's other fields keep their Escape", () => {
    const active = call()
    render(<div><input aria-label="elsewhere" /><form><VoiceCallBar call={active} /></form></div>)
    fireEvent.keyDown(screen.getByLabelText("elsewhere"), { key: "Escape" })
    expect(active.end).not.toHaveBeenCalled()
    // From inside the composer, or with nothing focused, it ends the call.
    fireEvent.keyDown(screen.getByRole("button", { name: "voiceConversation.mute" }), { key: "Escape" })
    expect(active.end).toHaveBeenCalledOnce()
    fireEvent.keyDown(document.body, { key: "Escape" })
    expect(active.end).toHaveBeenCalledTimes(2)
  })

  it("lets menus and dialogs have their Escape too", () => {
    const active = call()
    render(<div><div role="dialog"><button>inside dialog</button></div><form><VoiceCallBar call={active} /></form></div>)
    fireEvent.keyDown(screen.getByText("inside dialog"), { key: "Escape" })
    expect(active.end).not.toHaveBeenCalled()
  })

  it("tells the composer the user ended the call, from End and from Escape", () => {
    const onEnd = vi.fn()
    const active = call()
    render(<form><VoiceCallBar call={active} onEnd={onEnd} /></form>)
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.end" }))
    expect(onEnd).toHaveBeenCalledTimes(1)
    expect(active.end).toHaveBeenCalledTimes(1)
    fireEvent.keyDown(document.body, { key: "Escape" })
    expect(onEnd).toHaveBeenCalledTimes(2)
    expect(active.end).toHaveBeenCalledTimes(2)
  })

  it("reads the waveform colour a few times a second, not on every frame", () => {
    const context = { clearRect: vi.fn(), beginPath: vi.fn(), fill: vi.fn(), arc: vi.fn(), rect: vi.fn(), setTransform: vi.fn(), fillStyle: "" }
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context as unknown as CanvasRenderingContext2D)
    const read = vi.spyOn(window, "getComputedStyle")
    vi.useFakeTimers()
    let now = 0
    vi.spyOn(window, "requestAnimationFrame").mockImplementation(callback => window.setTimeout(() => { now += 16; callback(now) }, 16))
    render(<VoiceWaveform call={call()} />)
    vi.advanceTimersByTime(2000)
    // ~125 frames in two seconds.
    expect(read.mock.calls.length).toBeLessThan(10)
    expect(read.mock.calls.length).toBeGreaterThan(0)
    vi.useRealTimers()
  })

  it("does not swell the orb for people who ask for less motion", () => {
    const frames = vi.spyOn(window, "requestAnimationFrame").mockImplementation(() => 1)
    const matchMedia = vi.fn().mockReturnValue({ matches: true })
    vi.stubGlobal("matchMedia", matchMedia)
    render(<VoiceOrb call={call()} />)
    expect(matchMedia).toHaveBeenCalledWith("(prefers-reduced-motion: reduce)")
    expect(frames).not.toHaveBeenCalled()
    vi.unstubAllGlobals()
    cleanup()
    vi.stubGlobal("matchMedia", vi.fn().mockReturnValue({ matches: false }))
    render(<VoiceOrb call={call()} />)
    expect(frames).toHaveBeenCalled()
    vi.unstubAllGlobals()
  })
})
