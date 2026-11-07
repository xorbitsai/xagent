import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const runtime = vi.hoisted(() => ({
  path: "/task", search: "", login: "login-1",
  state: { taskId: null as number | null, messages: [] as Array<{
    id: string; role: string; isResult: boolean; status: string; content: string
  }>, isProcessing: false, currentTask: null as { status: string; waitingRequestId?: string } | null },
  sendMessage: vi.fn(), pauseTask: vi.fn(), isConnected: false, isMessageDeliveryPending: false,
  isConversationResetPending: false, isSessionInteractionLocked: false,
  voiceInputEnabled: true, taskControlsEnabled: true,
}))
const calls = vi.hoisted(() => [] as Array<{
  options: { activity: (speaking: boolean) => void; transcript: (type: "final", text: string, id: number) => void }
  close: ReturnType<typeof vi.fn>; speak: ReturnType<typeof vi.fn>
}>)
vi.mock("next/navigation", () => ({ usePathname: () => runtime.path, useSearchParams: () => new URLSearchParams(runtime.search) }))
vi.mock("@/contexts/app-context-chat", () => ({ useApp: () => runtime }))
vi.mock("@/contexts/auth-context", () => ({ useAuth: () => ({ token: "test-token", user: { id: 1 }, session: { sessionId: runtime.login } }) }))
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
    { id: 2, model_id: "speaker", model_name: "eleven_flash_v2_5", abilities: ["tts"] },
  ].map(model => ({
    category: "speech", model_provider: "elevenlabs", base_url: null, temperature: null,
    context_window: null, dimension: null, description: null, created_at: null, updated_at: null,
    is_active: true, is_owner: true, can_edit: true, can_delete: true, is_shared: false, ...model,
  }))), { status: 200 })),
}))

import { TaskVoiceProvider, useTaskVoice } from "./task-voice-context"

function Entry() {
  const open = useTaskVoice()
  return <button onClick={() => open?.({ model: "chosen-model", agentId: 7, metadata: { source: "composer" } })}>home-voice</button>
}
const tree = () => <TaskVoiceProvider><Entry /></TaskVoiceProvider>
async function openCall() {
  const view = render(tree())
  fireEvent.click(screen.getByRole("button", { name: "home-voice" }))
  await waitFor(() => expect(screen.getByRole("button", { name: "voiceConversation.start" })).toBeEnabled())
  fireEvent.click(screen.getByRole("button", { name: "voiceConversation.start" }))
  await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("voiceConversation.states.listening"))
  return view
}
async function speak() {
  await act(async () => {
    calls[0].options.activity(true)
    calls[0].options.transcript("final", "Hello there.", 1)
    calls[0].options.activity(false)
    await new Promise(resolve => setTimeout(resolve, 320))
  })
}

beforeEach(() => {
  runtime.path = "/task"; runtime.search = ""; runtime.login = "login-1"
  runtime.state = { taskId: null, messages: [], isProcessing: false, currentTask: null }
  runtime.isConnected = false; runtime.isMessageDeliveryPending = false
  runtime.sendMessage.mockReset(); runtime.pauseTask.mockReset()
})
afterEach(() => { cleanup(); calls.length = 0; vi.clearAllMocks() })

describe("home voice task creation", () => {
  it("retains creation ownership after send resolves but before task navigation and socket readiness", async () => {
    runtime.sendMessage.mockImplementation(async (_text, _config, _files, options) => { options.onCreatedTask(42) })
    const view = await openCall()
    await speak()
    runtime.state.taskId = 42
    view.rerender(tree())
    expect(calls[0].close).not.toHaveBeenCalled()
    runtime.path = "/task/42"
    view.rerender(tree())
    expect(calls[0].close).not.toHaveBeenCalled()
    runtime.isConnected = true
    view.rerender(tree())
    expect(calls[0].close).not.toHaveBeenCalled()
  })

  it("preserves composer metadata when answering a waiting request", async () => {
    runtime.path = "/task/42"; runtime.state.taskId = 42; runtime.isConnected = true
    runtime.state.currentTask = { status: "waiting_for_user", waitingRequestId: "request-1" }
    runtime.sendMessage.mockResolvedValue(undefined)
    await openCall()
    await speak()
    expect(runtime.sendMessage.mock.calls[0][1].metadata).toEqual({ source: "composer", request_id: "request-1" })
  })

  it("creates on the first utterance and keeps microphone/replies across its own task navigation", async () => {
    let accepted!: () => void
    runtime.sendMessage.mockImplementation(() => new Promise<void>(resolve => { accepted = resolve }))
    const view = await openCall()
    expect(runtime.sendMessage).not.toHaveBeenCalled()
    await speak()
    expect(runtime.sendMessage).toHaveBeenCalledWith("Hello there.", expect.objectContaining({
      model: "chosen-model", agentId: 7, clientMessageId: expect.any(String),
    }), [], expect.objectContaining({ canAdoptCreatedTask: expect.any(Function) }))
    runtime.sendMessage.mock.calls[0][3].onCreatedTask(42)
    runtime.state.taskId = 42
    runtime.path = "/task/42"
    runtime.isMessageDeliveryPending = true
    view.rerender(tree())
    expect(calls[0].close).not.toHaveBeenCalled()
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    runtime.isConnected = true; runtime.isMessageDeliveryPending = false
    await act(async () => { accepted() })
    runtime.state.messages = [{ id: "reply", role: "assistant", isResult: true, status: "completed", content: "Welcome back." }]
    view.rerender(tree())
    await waitFor(() => expect(calls[0].speak).toHaveBeenCalledWith("Welcome back."))
    expect(calls).toHaveLength(1)
  })

  it.each(["draft", "task", "login", "unmount"])("revokes a pending creation on %s changes", async change => {
    runtime.sendMessage.mockImplementation(() => new Promise<void>(() => {}))
    const view = await openCall()
    await speak()
    const canAdopt = runtime.sendMessage.mock.calls[0][3].canAdoptCreatedTask
    expect(canAdopt()).toBe(true)
    if (change === "draft") runtime.search = "new=another"
    if (change === "task") { runtime.path = "/task/99"; runtime.state.taskId = 99 }
    if (change === "login") runtime.login = "login-2"
    if (change === "unmount") view.unmount()
    else view.rerender(tree())
    expect(canAdopt()).toBe(false)
    expect(calls[0].close).toHaveBeenCalledOnce()
  })
})
