import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const calls = vi.hoisted(() => [] as Array<{
  options: { activity: (active: boolean) => void; transcript: (type: "partial" | "final", text: string, id: number) => void }
  close: ReturnType<typeof vi.fn>; stopAudio: ReturnType<typeof vi.fn>; speak: ReturnType<typeof vi.fn>
}>)
const auth = vi.hoisted(() => ({ token: "user-token", user: { id: 1 }, session: { sessionId: "login-1" } }))
vi.mock("@/contexts/auth-context", () => ({ useAuth: () => auth }))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))
vi.mock("@/lib/voice-audio", () => ({ VoiceAudio: class {
  close = vi.fn(); stopAudio = vi.fn(); speak = vi.fn().mockResolvedValue(undefined)
  constructor(public options: typeof calls[number]["options"]) { calls.push(this) }
  async start() {}
} }))
vi.mock("@/lib/api-wrapper", async importOriginal => ({
  ...await importOriginal<typeof import("@/lib/api-wrapper")>(),
  apiRequest: vi.fn(async () => new Response(JSON.stringify([
    { id: 1, model_id: "live", model_name: "scribe_v2_realtime", abilities: ["asr"] },
    { id: 2, model_id: "batch", model_name: "scribe_v2", abilities: ["asr"] },
    { id: 3, model_id: "speaker", model_name: "eleven_flash_v2_5", abilities: ["tts"] },
  ].map(model => ({
    category: "speech", model_provider: "elevenlabs", base_url: null, temperature: null,
    context_window: null, dimension: null, description: null, created_at: null, updated_at: null,
    is_active: true, is_owner: true, can_edit: true, can_delete: true, is_shared: false, ...model,
  }))), { status: 200 })),
}))

import { VoiceConversation } from "./voice-conversation"

afterEach(() => { cleanup(); calls.length = 0; vi.clearAllMocks() })

describe("voice conversation entry and call lifecycle", () => {
  it("selects live models, sends a voice turn, speaks the reply and interrupts without a button", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined)
    const onInterrupt = vi.fn()
    const props = { taskId: 42, messages: [], busy: false, waiting: false, disabled: false, onSend, onInterrupt }
    const view = render(<VoiceConversation {...props} />)
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.open" }))
    await waitFor(() => expect(screen.getByLabelText("voiceConversation.transcriptionModel")).toHaveValue("live"))
    expect(screen.queryByRole("option", { name: "batch" })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.start" }))
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("voiceConversation.states.listening"))
    const call = calls[0]
    await act(async () => {
      call.options.activity(true)
      call.options.transcript("final", "Hello there.", 0)
      call.options.activity(false)
      await new Promise(resolve => setTimeout(resolve, 320))
    })
    expect(onSend).toHaveBeenCalledWith("Hello there.", expect.any(String))
    view.rerender(<VoiceConversation {...props} busy messages={[
      { id: "answer", role: "assistant", isResult: true, status: "running", content: "Hello back. Still thinking" },
    ]} />)
    await waitFor(() => expect(call.speak).toHaveBeenCalledWith("Hello back. "))
    const stops = call.stopAudio.mock.calls.length
    act(() => { call.options.activity(true) })
    expect(call.stopAudio).toHaveBeenCalledTimes(stops + 1)
    act(() => { call.options.transcript("partial", "Change that", 1) })
    expect(onInterrupt).toHaveBeenCalledOnce()
    expect(screen.queryByRole("button", { name: /interrupt|打断/i })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.end" }))
    expect(call.close).toHaveBeenCalledOnce()
  })

  it("ends the old call on task navigation and ignores late speech callbacks", async () => {
    const props = { taskId: 42, messages: [], busy: false, waiting: false, disabled: false, onSend: vi.fn().mockResolvedValue(undefined), onInterrupt: vi.fn() }
    const view = render(<VoiceConversation {...props} />)
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.open" }))
    await waitFor(() => expect(screen.getByRole("button", { name: "voiceConversation.start" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.start" }))
    await waitFor(() => expect(calls).toHaveLength(1))
    const old = calls[0]
    view.rerender(<VoiceConversation {...props} taskId={43} />)
    expect(old.close).toHaveBeenCalledOnce()
    act(() => { old.options.transcript("final", "Too late", 0) })
    expect(props.onSend).not.toHaveBeenCalled()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })
})
