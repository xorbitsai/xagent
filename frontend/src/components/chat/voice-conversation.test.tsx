import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const calls = vi.hoisted(() => [] as Array<{
  options: { activity: (active: boolean) => void; transcript: (type: "partial" | "final", text: string, id: number) => void; error: (reason?: "session_timeout" | "authentication_required") => void }
  close: ReturnType<typeof vi.fn>; stopAudio: ReturnType<typeof vi.fn>; speak: ReturnType<typeof vi.fn>
}>)
const auth = vi.hoisted(() => ({ token: "user-token", user: { id: 1 }, session: { sessionId: "login-1" } }))
const preferences = vi.hoisted(() => ({ asr_model_id: "live", tts_model_id: "speaker", voice: "saved-voice" }))
vi.mock("@/hooks/use-voice-preferences", () => ({ useVoicePreferences: () => ({ preferences, loading: false, error: false }) }))
vi.mock("@/contexts/auth-context", () => ({ useAuth: () => auth }))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))
vi.mock("@/lib/voice-audio", async original => ({ ...await original<typeof import("@/lib/voice-audio")>(), VoiceAudio: class {
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
import { VoiceAudioError } from "@/lib/voice-audio"

function Call(props: Omit<React.ComponentProps<typeof VoiceConversation>, "open" | "onOpenChange">) {
  const [open, setOpen] = React.useState(true)
  return <VoiceConversation {...props} open={open} onOpenChange={setOpen} />
}

afterEach(() => { cleanup(); calls.length = 0; preferences.asr_model_id = "live"; vi.clearAllMocks() })

describe("voice conversation call lifecycle", () => {
  it("shows the ASR session deadline without blaming microphone configuration", async () => {
    render(<Call taskId={42} messages={[]} busy={false} waiting={false} disabled={false} onSend={vi.fn()} onInterrupt={vi.fn()} />)
    await waitFor(() => expect(screen.getByRole("button", { name: "voiceConversation.start" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.start" }))
    await waitFor(() => expect(calls).toHaveLength(1))
    act(() => { calls[0].options.error("session_timeout") })
    expect(screen.getByRole("status")).toHaveTextContent("voiceConversation.sessionTimeout")
    expect(calls[0].close).toHaveBeenCalledOnce()
  })

  it.each([false, true])("keeps a TTS failure recoverable except for authentication loss (%s)", async authFailure => {
    const props = { taskId: 42, messages: [], busy: false, waiting: false, disabled: false, onSend: vi.fn().mockResolvedValue(undefined), onInterrupt: vi.fn() }
    const view = render(<Call {...props} />)
    await waitFor(() => expect(screen.getByRole("button", { name: "voiceConversation.start" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.start" }))
    await waitFor(() => expect(calls).toHaveLength(1))
    calls[0].speak.mockRejectedValueOnce(authFailure ? new VoiceAudioError("authentication_required") : new Error("temporary synthesis failure"))
    await act(async () => {
      calls[0].options.transcript("final", "Hello", 0)
      await new Promise(resolve => setTimeout(resolve, 320))
    })
    view.rerender(<Call {...props} messages={[{ id: "answer", role: "assistant", isResult: true, status: "completed", content: "Hello back." }]} />)
    if (authFailure) {
      await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("voiceConversation.authenticationRequired"))
      expect(calls[0].close).toHaveBeenCalledOnce()
    } else {
      await screen.findByText("voiceConversation.playbackFailed")
      expect(calls[0].close).not.toHaveBeenCalled()
      expect(screen.getByRole("button", { name: "voiceConversation.mute" })).toBeEnabled()
    }
  })

  it("selects live models, sends a voice turn, speaks the reply and interrupts without a button", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined)
    const onInterrupt = vi.fn()
    const props = { taskId: 42, messages: [], busy: false, waiting: false, disabled: false, onSend, onInterrupt }
    const view = render(<Call {...props} />)
    await waitFor(() => expect(screen.getByRole("button", { name: "voiceConversation.start" })).toBeEnabled())
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument()
    expect(screen.queryByRole("option", { name: "batch" })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.start" }))
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("voiceConversation.states.listening"))
    const call = calls[0]
    expect(call.options).toMatchObject({ asr: "live", tts: "speaker", voice: "saved-voice" })
    await act(async () => {
      call.options.activity(true)
      call.options.transcript("final", "Hello there.", 0)
      call.options.activity(false)
      await new Promise(resolve => setTimeout(resolve, 320))
    })
    expect(onSend).toHaveBeenCalledWith("Hello there.", expect.any(String))
    view.rerender(<Call {...props} busy messages={[
      { id: "answer", role: "assistant", isResult: true, status: "running", content: "Hello back. Still thinking" },
    ]} />)
    await waitFor(() => expect(call.speak).toHaveBeenCalledWith("Hello back. "))
    const stops = call.stopAudio.mock.calls.length
    act(() => { call.options.activity(true) })
    expect(call.stopAudio).toHaveBeenCalledTimes(stops)
    act(() => { call.options.transcript("partial", "Change that", 1) })
    expect(call.stopAudio).toHaveBeenCalledTimes(stops + 1)
    expect(onInterrupt).toHaveBeenCalledOnce()
    expect(screen.queryByRole("button", { name: /interrupt|打断/i })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.end" }))
    expect(call.close).toHaveBeenCalledOnce()
  })

  it("requires correcting an unavailable saved model rather than silently changing it", async () => {
    preferences.asr_model_id = "removed-model"
    render(<Call taskId={42} messages={[]} busy={false} waiting={false} disabled={false} onSend={vi.fn()} onInterrupt={vi.fn()} />)
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("settings.voice.selectionUnavailable"))
    expect(screen.getByRole("button", { name: "voiceConversation.start" })).toBeDisabled()
    expect(screen.getByRole("link", { name: "settings.voice.openSettings" })).toHaveAttribute("href", "?settings=voice")
  })

  it("shows a rejected task's explanation instead of blaming the microphone", async () => {
    const message = "Your Auto model configuration has no usable candidate models. Review your Auto model settings."
    const onSend = vi.fn().mockRejectedValue(new Error(message))
    render(<Call taskId={null} messages={[]} busy={false} waiting={false} disabled={false} onSend={onSend} onInterrupt={vi.fn()} />)
    await waitFor(() => expect(screen.getByRole("button", { name: "voiceConversation.start" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.start" }))
    await waitFor(() => expect(calls).toHaveLength(1))
    await act(async () => {
      calls[0].options.activity(true)
      calls[0].options.transcript("final", "Hello. Can you hear me?", 0)
      calls[0].options.activity(false)
      await new Promise(resolve => setTimeout(resolve, 320))
    })
    expect(screen.getByRole("status")).toHaveTextContent("voiceConversation.sendFailed")
    expect(screen.getByRole("alert")).toHaveTextContent(message)
    expect(screen.queryByText("voiceConversation.states.error")).not.toBeInTheDocument()
    expect(onSend).toHaveBeenCalledTimes(1)
    expect(calls[0].close).toHaveBeenCalledOnce()
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.start" }))
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("voiceConversation.states.listening"))
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(onSend).toHaveBeenCalledTimes(1)
  })

  it("ends the old call on task navigation and ignores late speech callbacks", async () => {
    const props = { taskId: 42, messages: [], busy: false, waiting: false, disabled: false, onSend: vi.fn().mockResolvedValue(undefined), onInterrupt: vi.fn() }
    const view = render(<Call {...props} />)
    await waitFor(() => expect(screen.getByRole("button", { name: "voiceConversation.start" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.start" }))
    await waitFor(() => expect(calls).toHaveLength(1))
    const old = calls[0]
    view.rerender(<Call {...props} taskId={43} />)
    expect(old.close).toHaveBeenCalledOnce()
    act(() => { old.options.transcript("final", "Too late", 0) })
    expect(props.onSend).not.toHaveBeenCalled()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })
})
