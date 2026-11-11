import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))

import { VoiceCallBar } from "./voice-call-ui"
import type { VoiceCall } from "@/hooks/use-voice-call"

const call = (over: Partial<VoiceCall> = {}): VoiceCall => ({
  open: true, phase: "listening", active: true, caption: "", muted: false,
  status: "voiceConversation.states.listening", settingsLink: false, retryable: true, retryDisabled: false,
  level: () => 0, retry: vi.fn(), end: vi.fn(), toggleMute: vi.fn(), ...over,
})

beforeEach(() => { vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null) })
afterEach(() => { cleanup(); vi.restoreAllMocks() })

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
})
