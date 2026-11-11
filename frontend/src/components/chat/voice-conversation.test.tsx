import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const calls = vi.hoisted(() => [] as Array<{
  options: { context?: unknown; vadSilenceThreshold?: number; activity: (active: boolean) => void; transcript: (type: "partial" | "final", text: string, id: number) => void; error: (reason?: "session_timeout" | "authentication_required") => void }
  close: ReturnType<typeof vi.fn>; stopAudio: ReturnType<typeof vi.fn>; speak: ReturnType<typeof vi.fn>; level: ReturnType<typeof vi.fn>
}>)
const auth = vi.hoisted(() => ({ token: "user-token", user: { id: 1 }, session: { sessionId: "login-1" } }))
const preferences = vi.hoisted(() => ({ asr_model_id: "live", tts_model_id: "speaker", voice: "saved-voice" }))
const prefsHook = vi.hoisted(() => ({ error: false, calls: [] as Array<[boolean, boolean | undefined]> }))
vi.mock("@/hooks/use-voice-preferences", () => ({ useVoicePreferences: (enabled: boolean, refreshKey?: boolean) => {
  prefsHook.calls.push([enabled, refreshKey])
  return { preferences, loading: false, error: prefsHook.error }
} }))
vi.mock("@/contexts/auth-context", () => ({ useAuth: () => auth }))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))
vi.mock("@/lib/voice-audio", async original => ({ ...await original<typeof import("@/lib/voice-audio")>(), VoiceAudio: class {
  close = vi.fn(); stopAudio = vi.fn(); speak = vi.fn().mockResolvedValue(undefined); level = vi.fn().mockReturnValue(0)
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
import { VoiceCallBar } from "./voice-call-ui"
import { VoiceCallStoreProvider, useVoiceCallView } from "@/contexts/voice-call-store"
import { VoiceAudioError } from "@/lib/voice-audio"
import { apiRequest } from "@/lib/api-wrapper"

// The call has no window of its own: it publishes its state and the composer
// draws it. This stands in for the composer.
function Composer() {
  const call = useVoiceCallView()
  return call ? <VoiceCallBar call={call} /> : null
}
function Call(props: Omit<React.ComponentProps<typeof VoiceConversation>, "open" | "onOpenChange">) {
  const [open, setOpen] = React.useState(true)
  return <VoiceCallStoreProvider><VoiceConversation {...props} open={open} onOpenChange={setOpen} /><Composer /></VoiceCallStoreProvider>
}
const idle = { taskId: 42, messages: [], busy: false, waiting: false, disabled: false, onSend: vi.fn(), onInterrupt: vi.fn() }

beforeEach(() => { vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null) })
afterEach(() => { cleanup(); calls.length = 0; prefsHook.calls.length = 0; prefsHook.error = false; preferences.asr_model_id = "live"; vi.clearAllMocks(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

describe("voice call lifecycle", () => {
  it("starts as soon as it is opened, with no confirmation step", async () => {
    render(<Call {...idle} />)
    await waitFor(() => expect(calls).toHaveLength(1))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("voiceConversation.states.listening"))
  })

  it("omits ElevenLabs VAD timing for a selected Xinference ASR", async () => {
    vi.mocked(apiRequest).mockResolvedValueOnce(new Response(JSON.stringify([
      { id: 1, model_id: "live", model_name: "gpu-asr", model_provider: "xinference", abilities: ["asr"] },
      { id: 3, model_id: "speaker", model_name: "eleven_flash_v2_5", model_provider: "elevenlabs", abilities: ["tts"] },
    ].map(model => ({ category: "speech", base_url: null, temperature: null, context_window: null, dimension: null, description: null, created_at: null, updated_at: null, is_active: true, is_owner: true, can_edit: true, can_delete: true, is_shared: false, ...model })))))
      .mockResolvedValueOnce(new Response(JSON.stringify({ streaming: true, server_vad: true })))
    render(<Call {...idle} />)
    await waitFor(() => expect(calls).toHaveLength(1))
    expect(calls[0].options.vadSilenceThreshold).toBeUndefined()
  })

  it("shows the ASR session deadline without blaming microphone configuration", async () => {
    render(<Call {...idle} />)
    await waitFor(() => expect(calls).toHaveLength(1))
    act(() => { calls[0].options.error("session_timeout") })
    expect(screen.getByRole("status")).toHaveTextContent("voiceConversation.states.stopped")
    expect(screen.getByRole("alert")).toHaveTextContent("voiceConversation.sessionTimeout")
    expect(calls[0].close).toHaveBeenCalledOnce()
  })

  it.each([false, true])("keeps a TTS failure recoverable except for authentication loss (%s)", async authFailure => {
    const props = { ...idle, onSend: vi.fn().mockResolvedValue(undefined) }
    const view = render(<Call {...props} />)
    await waitFor(() => expect(calls).toHaveLength(1))
    calls[0].speak.mockRejectedValueOnce(authFailure ? new VoiceAudioError("authentication_required") : new Error("temporary synthesis failure"))
    await act(async () => {
      calls[0].options.transcript("final", "Hello", 0)
      await new Promise(resolve => setTimeout(resolve, 320))
    })
    view.rerender(<Call {...props} messages={[{ id: "answer", role: "assistant", isResult: true, status: "completed", content: "Hello back." }]} />)
    if (authFailure) {
      await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("voiceConversation.authenticationRequired"))
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
    const props = { ...idle, onSend, onInterrupt }
    const view = render(<Call {...props} />)
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("voiceConversation.states.listening"))
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument()
    expect(screen.queryByRole("option", { name: "batch" })).not.toBeInTheDocument()
    const call = calls[0]
    expect(call.options).toMatchObject({ asr: "live", tts: "speaker", voice: "saved-voice" })
    expect(call.options.vadSilenceThreshold).toBe(0.5)
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
    expect(screen.queryByTestId("voice-call-bar")).not.toBeInTheDocument()
  })

  it("shows what is being said as it is recognized and clears it once sent", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined)
    render(<Call {...idle} onSend={onSend} />)
    await waitFor(() => expect(calls).toHaveLength(1))
    expect(screen.queryByTestId("voice-caption")).not.toBeInTheDocument()
    act(() => { calls[0].options.activity(true) })
    act(() => { calls[0].options.transcript("partial", "nothing much", 0) })
    expect(screen.getByTestId("voice-caption")).toHaveTextContent("nothing much")
    act(() => { calls[0].options.transcript("final", "nothing much thank you", 0) })
    expect(screen.getByTestId("voice-caption")).toHaveTextContent("nothing much thank you")
    expect(onSend).not.toHaveBeenCalled()
    await act(async () => {
      calls[0].options.activity(false)
      await new Promise(resolve => setTimeout(resolve, 320))
    })
    expect(onSend).toHaveBeenCalledWith("nothing much thank you", expect.any(String))
    await waitFor(() => expect(screen.queryByTestId("voice-caption")).not.toBeInTheDocument())
  })

  it("requires correcting an unavailable saved model rather than silently changing it", async () => {
    preferences.asr_model_id = "removed-model"
    render(<Call {...idle} />)
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("settings.voice.selectionUnavailable"))
    expect(calls).toHaveLength(0)
    expect(screen.getByRole("link", { name: "settings.voice.openSettings" })).toHaveAttribute("href", "?settings=voice")
  })

  it("shows a rejected task's explanation instead of blaming the microphone", async () => {
    const message = "Your Auto model configuration has no usable candidate models. Review your Auto model settings."
    const onSend = vi.fn().mockRejectedValue(new Error(message))
    render(<Call {...idle} taskId={null} onSend={onSend} />)
    await waitFor(() => expect(calls).toHaveLength(1))
    await act(async () => {
      calls[0].options.activity(true)
      calls[0].options.transcript("final", "Hello. Can you hear me?", 0)
      calls[0].options.activity(false)
      await new Promise(resolve => setTimeout(resolve, 320))
    })
    expect(screen.getByRole("status")).toHaveTextContent("voiceConversation.states.stopped")
    expect(screen.getByRole("alert")).toHaveTextContent(message)
    expect(onSend).toHaveBeenCalledTimes(1)
    expect(calls[0].close).toHaveBeenCalledOnce()
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.retry" }))
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("voiceConversation.states.listening"))
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(onSend).toHaveBeenCalledTimes(1)
  })

  it("ends the old call on task navigation and ignores late speech callbacks", async () => {
    const props = { ...idle, onSend: vi.fn().mockResolvedValue(undefined) }
    const view = render(<Call {...props} />)
    await waitFor(() => expect(calls).toHaveLength(1))
    const old = calls[0]
    view.rerender(<Call {...props} taskId={43} />)
    expect(old.close).toHaveBeenCalledOnce()
    act(() => { old.options.transcript("final", "Too late", 0) })
    expect(props.onSend).not.toHaveBeenCalled()
    expect(screen.queryByTestId("voice-call-bar")).not.toBeInTheDocument()
  })

  it("does not start a call while an agent run is in progress", async () => {
    render(<Call {...idle} busy />)
    await waitFor(() => expect(screen.queryByTestId("voice-call-bar")).not.toBeInTheDocument())
    expect(calls).toHaveLength(0)
  })

  it("offers to try again when the models lookup failed, and repeats the lookup", async () => {
    vi.mocked(apiRequest).mockResolvedValueOnce(new Response("unavailable", { status: 500 }))
    render(<Call {...idle} />)
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("voiceConversation.modelsUnavailable"))
    expect(calls).toHaveLength(0)
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.retry" }))
    await waitFor(() => expect(calls).toHaveLength(1))
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("does not offer to try again when only a settings change can help", async () => {
    preferences.asr_model_id = "removed-model"
    render(<Call {...idle} />)
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("settings.voice.selectionUnavailable"))
    expect(screen.queryByRole("button", { name: "voiceConversation.retry" })).not.toBeInTheDocument()
  })

  it("holds a retry until the running task has finished", async () => {
    const view = render(<Call {...idle} />)
    await waitFor(() => expect(calls).toHaveLength(1))
    act(() => { calls[0].options.error() })
    view.rerender(<Call {...idle} busy />)
    expect(screen.getByRole("button", { name: "voiceConversation.retry" })).toBeDisabled()
    view.rerender(<Call {...idle} />)
    expect(screen.getByRole("button", { name: "voiceConversation.retry" })).toBeEnabled()
  })

  it("reports a missing login instead of waiting forever", async () => {
    const token = auth.token
    auth.token = null as unknown as string
    try {
      render(<Call {...idle} />)
      await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("voiceConversation.authenticationRequired"))
      expect(calls).toHaveLength(0)
      expect(screen.queryByRole("button", { name: "voiceConversation.retry" })).not.toBeInTheDocument()
    } finally { auth.token = token }
  })

  describe("audio unlock", () => {
    // The call can only begin after its models and settings have loaded, but a
    // browser only lets audio start from the user's click.
    const contexts: Array<{ resume: ReturnType<typeof vi.fn>; close: ReturnType<typeof vi.fn>; state: string }> = []
    beforeEach(() => {
      contexts.length = 0
      vi.stubGlobal("AudioContext", class {
        state = "running"; resume = vi.fn().mockResolvedValue(undefined); close = vi.fn().mockResolvedValue(undefined)
        constructor() { contexts.push(this) }
      })
    })
    const slowModels = () => {
      let release!: () => void
      vi.mocked(apiRequest).mockImplementationOnce(() => new Promise<Response>(resolve => {
        release = () => resolve(new Response(JSON.stringify([
          { id: 1, model_id: "live", model_name: "scribe_v2_realtime", abilities: ["asr"] },
          { id: 3, model_id: "speaker", model_name: "eleven_flash_v2_5", abilities: ["tts"] },
        ].map(model => ({ category: "speech", model_provider: "elevenlabs", base_url: null, temperature: null, context_window: null, dimension: null, description: null, created_at: null, updated_at: null, is_active: true, is_owner: true, can_edit: true, can_delete: true, is_shared: false, ...model })))))
      }))
      return () => release()
    }

    it("creates and resumes the audio context in the opening click, before the models have loaded", async () => {
      const finishLoading = slowModels()
      const props = { ...idle, onOpenChange: vi.fn() }
      const view = render(<VoiceCallStoreProvider><VoiceConversation {...props} open={false} /><Composer /></VoiceCallStoreProvider>)
      expect(contexts).toHaveLength(0)
      view.rerender(<VoiceCallStoreProvider><VoiceConversation {...props} open /><Composer /></VoiceCallStoreProvider>)
      expect(contexts).toHaveLength(1)
      expect(contexts[0].resume).toHaveBeenCalled()
      expect(calls).toHaveLength(0)
      await act(async () => { finishLoading() })
      await waitFor(() => expect(calls).toHaveLength(1))
      expect(calls[0].options.context).toBe(contexts[0])
      expect(contexts).toHaveLength(1)
    })

    it("closes a context that was never used when the call is dismissed", async () => {
      slowModels()
      const props = { ...idle, onOpenChange: vi.fn() }
      const view = render(<VoiceCallStoreProvider><VoiceConversation {...props} open /><Composer /></VoiceCallStoreProvider>)
      expect(contexts).toHaveLength(1)
      view.rerender(<VoiceCallStoreProvider><VoiceConversation {...props} open={false} /><Composer /></VoiceCallStoreProvider>)
      expect(contexts[0].close).toHaveBeenCalledOnce()
    })

    it("unlocks a fresh context for a retry, which is a click too", async () => {
      render(<Call {...idle} />)
      await waitFor(() => expect(calls).toHaveLength(1))
      act(() => { calls[0].options.error() })
      expect(contexts).toHaveLength(1)
      fireEvent.click(screen.getByRole("button", { name: "voiceConversation.retry" }))
      expect(contexts).toHaveLength(2)
      await waitFor(() => expect(calls).toHaveLength(2))
      expect(calls[1].options.context).toBe(contexts[1])
    })
  })

  it("looks the models and settings up when a call is opened, not ahead of time", async () => {
    const props = { ...idle, onOpenChange: vi.fn() }
    const view = render(<VoiceCallStoreProvider><VoiceConversation {...props} open={false} /><Composer /></VoiceCallStoreProvider>)
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 30)) })
    expect(apiRequest).not.toHaveBeenCalled()
    expect(prefsHook.calls.every(([enabled]) => enabled === false)).toBe(true)
    view.rerender(<VoiceCallStoreProvider><VoiceConversation {...props} open /><Composer /></VoiceCallStoreProvider>)
    await waitFor(() => expect(calls).toHaveLength(1))
    expect(apiRequest).toHaveBeenCalled()
    expect(prefsHook.calls.some(([enabled]) => enabled === true)).toBe(true)
    // Closed and opened again: settings changed in between are read afresh.
    view.rerender(<VoiceCallStoreProvider><VoiceConversation {...props} open={false} /><Composer /></VoiceCallStoreProvider>)
    expect(prefsHook.calls[prefsHook.calls.length - 1][0]).toBe(false)
  })

  it("repeats a failed settings lookup when trying again", async () => {
    prefsHook.error = true
    render(<Call {...idle} />)
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("settings.voice.loadFailed"))
    expect(calls).toHaveLength(0)
    prefsHook.error = false
    fireEvent.click(screen.getByRole("button", { name: "voiceConversation.retry" }))
    await waitFor(() => expect(calls).toHaveLength(1))
    // The lookup was asked to refresh.
    expect(prefsHook.calls.some(([, refresh]) => refresh === true)).toBe(true)
  })

  it("keeps what was said while a message was being sent", async () => {
    let finishSending!: () => void
    const onSend = vi.fn(() => new Promise<void>(resolve => { finishSending = resolve }))
    render(<Call {...idle} onSend={onSend} />)
    await waitFor(() => expect(calls).toHaveLength(1))
    await act(async () => {
      calls[0].options.activity(true)
      calls[0].options.transcript("final", "First thing.", 0)
      calls[0].options.activity(false)
      await new Promise(resolve => setTimeout(resolve, 320))
    })
    expect(onSend).toHaveBeenCalledWith("First thing.", expect.any(String))
    // The user carries on talking while the first message is still on its way.
    act(() => { calls[0].options.transcript("partial", "And another thing", 1) })
    await act(async () => { finishSending() })
    expect(screen.getByTestId("voice-caption")).toHaveTextContent("And another thing")
  })

  it("keeps showing what is said when only playback failed", async () => {
    const props = { ...idle, onSend: vi.fn().mockResolvedValue(undefined) }
    const view = render(<Call {...props} />)
    await waitFor(() => expect(calls).toHaveLength(1))
    calls[0].speak.mockRejectedValueOnce(new Error("temporary synthesis failure"))
    await act(async () => {
      calls[0].options.transcript("final", "Hello", 0)
      await new Promise(resolve => setTimeout(resolve, 320))
    })
    view.rerender(<Call {...props} messages={[{ id: "answer", role: "assistant", isResult: true, status: "completed", content: "Hello back." }]} />)
    await screen.findByTestId("voice-note")
    expect(screen.getByTestId("voice-note")).toHaveTextContent("voiceConversation.playbackFailed")
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    act(() => { calls[0].options.transcript("partial", "Still talking", 1) })
    expect(screen.getByTestId("voice-caption")).toHaveTextContent("Still talking")
    expect(screen.getByTestId("voice-note")).toBeInTheDocument()
  })

  it("does not offer another attempt when the login is gone", async () => {
    render(<Call {...idle} />)
    await waitFor(() => expect(calls).toHaveLength(1))
    act(() => { calls[0].options.error("authentication_required") })
    expect(screen.getByRole("alert")).toHaveTextContent("voiceConversation.authenticationRequired")
    expect(screen.queryByRole("button", { name: "voiceConversation.retry" })).not.toBeInTheDocument()
  })
})
