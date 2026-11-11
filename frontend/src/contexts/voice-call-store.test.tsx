import React from "react"
import { act, cleanup, render } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { VoiceCallStoreProvider, usePublishVoiceCall, useVoiceCallView } from "./voice-call-store"
import type { VoiceCall } from "@/hooks/use-voice-call"

const call = (name: string, over: Partial<VoiceCall> = {}): VoiceCall & { name: string } => ({
  name, open: true, phase: "listening", active: false, caption: "", muted: false, status: "", settingsLink: false,
  retryable: true, retryDisabled: false, level: () => 0, retry: vi.fn(), end: vi.fn(), toggleMute: vi.fn(), ...over,
})

function Publisher({ value }: { value: VoiceCall | null }) {
  // A host that is not open publishes a call with open: false.
  usePublishVoiceCall(value ?? call("closed", { open: false }))
  return null
}
let seen: Array<VoiceCall | null> = []
function Viewer() {
  const view = useVoiceCallView()
  seen.push(view)
  return <span data-testid="who">{view ? (view as VoiceCall & { name: string }).name : "none"}</span>
}
const who = (view: ReturnType<typeof render>) => view.getByTestId("who").textContent

afterEach(() => { cleanup(); seen = [] })

describe("voice call store", () => {
  it("shows nothing when no call is open", () => {
    const view = render(<VoiceCallStoreProvider><Publisher value={null} /><Viewer /></VoiceCallStoreProvider>)
    expect(who(view)).toBe("none")
  })

  it("shows the call that is open", () => {
    const view = render(<VoiceCallStoreProvider><Publisher value={call("panel")} /><Publisher value={null} /><Viewer /></VoiceCallStoreProvider>)
    expect(who(view)).toBe("panel")
  })

  it("prefers a running call over one that has only been opened", () => {
    const view = render(<VoiceCallStoreProvider>
      <Publisher value={call("opened-first")} />
      <Publisher value={call("running", { active: true })} />
      <Viewer />
    </VoiceCallStoreProvider>)
    expect(who(view)).toBe("running")
  })

  it("falls back to the first when none is running", () => {
    const view = render(<VoiceCallStoreProvider>
      <Publisher value={call("first")} />
      <Publisher value={call("second")} />
      <Viewer />
    </VoiceCallStoreProvider>)
    expect(who(view)).toBe("first")
  })

  it("follows a host when its call starts, and lets go when it closes or unmounts", () => {
    const tree = (a: VoiceCall | null, b: VoiceCall | null, showB = true) => (
      <VoiceCallStoreProvider><Publisher value={a} />{showB && <Publisher value={b} />}<Viewer /></VoiceCallStoreProvider>
    )
    const first = call("first")
    const view = render(tree(first, call("second")))
    expect(who(view)).toBe("first")
    view.rerender(tree(first, call("second", { active: true })))
    expect(who(view)).toBe("second")
    view.rerender(tree(first, call("second", { active: true }), false))
    expect(who(view)).toBe("first")
    view.rerender(tree(null, null, false))
    expect(who(view)).toBe("none")
  })

  it("does not re-render viewers when nothing visible changed", () => {
    const stable = call("stable")
    const view = render(<VoiceCallStoreProvider><Publisher value={stable} /><Viewer /></VoiceCallStoreProvider>)
    const before = seen.length
    act(() => { view.rerender(<VoiceCallStoreProvider><Publisher value={stable} /><Viewer /></VoiceCallStoreProvider>) })
    // The same call object is the same snapshot: only the parent's own re-render shows up.
    expect(seen.slice(before).every(entry => entry === stable)).toBe(true)
  })

  it("reads as no call outside a provider", () => {
    const view = render(<Viewer />)
    expect(who(view)).toBe("none")
  })
})
