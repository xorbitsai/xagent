import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { ModelWithAccess } from "@/lib/models"

const auth = vi.hoisted(() => ({ token: "token-a", user: { id: 1 }, session: { sessionId: "login-a" } }))
const request = vi.hoisted(() => vi.fn())
vi.mock("@/contexts/auth-context", () => ({ useAuth: () => auth }))
vi.mock("@/lib/api-wrapper", async importOriginal => ({ ...await importOriginal<typeof import("@/lib/api-wrapper")>(), apiRequest: request }))
import { useVoiceModels, voiceModelOptions, voiceModelReason } from "./use-voice-models"

const model = (override: Partial<ModelWithAccess> = {}): ModelWithAccess => ({
  id: 1, model_id: "live", category: "speech", model_provider: "elevenlabs", model_name: "scribe_v2_realtime",
  base_url: null, temperature: null, context_window: null, dimension: null, description: null,
  created_at: null, updated_at: null, is_active: true, is_owner: false, can_edit: false,
  can_delete: false, is_shared: true, abilities: ["asr"], ...override,
})
const asr = model()
const tts = model({ id: 2, model_id: "speaker", model_name: "eleven_v3", abilities: ["tts"] })
const response = (models: ModelWithAccess[]) => new Response(JSON.stringify(models), { status: 200 })
beforeEach(() => { request.mockReset(); auth.token = "token-a"; auth.session.sessionId = "login-a" })
afterEach(() => { vi.clearAllMocks(); vi.restoreAllMocks() })

describe("voice model requirements", () => {
  it("requires active streaming ASR/server VAD and TTS, including accessible shared models", () => {
    const result = voiceModelOptions([asr, tts,
      model({ model_id: "batch", model_name: "scribe_v2" }),
      model({ model_id: "other", model_provider: "unsupported" }),
      model({ ...tts, model_id: "inactive", is_active: false }),
      model({ model_id: "wrong-category", category: "llm" }),
    ])
    expect(result).toEqual({ asr: [asr], tts: [tts] })
  })
  it.each([
    [[], [], true, "needsAsrAndTts"], [[asr], [], true, "needsTts"],
    [[], [tts], true, "needsAsr"], [[asr], [tts], false, "needsLlm"],
    [[asr], [tts], true, undefined],
  ])("reports the missing requirement for %j / %j", (asr, tts, llm, reason) => {
    expect(voiceModelReason({ asr: asr as ModelWithAccess[], tts: tts as ModelWithAccess[], loading: false, error: false }, Boolean(llm))).toBe(reason)
  })
  it("keeps network failures distinct from missing configuration", () => {
    expect(voiceModelReason({ asr: [], tts: [], loading: true, error: false })).toBe("checkingModels")
    expect(voiceModelReason({ asr: [], tts: [], loading: false, error: true })).toBe("modelsUnavailable")
  })
  it("does not fetch without capability opt-in, and refreshes when returning to the app", async () => {
    request.mockImplementation(async () => response([asr, tts]))
    const view = renderHook(({ enabled }) => useVoiceModels(enabled), { initialProps: { enabled: false } })
    expect(request).not.toHaveBeenCalled()
    view.rerender({ enabled: true })
    await waitFor(() => expect(view.result.current.asr.map(model => model.model_id)).toEqual(["live"]))
    request.mockImplementation(async () => response([asr]))
    act(() => { window.dispatchEvent(new Event("focus")) })
    await waitFor(() => expect(voiceModelReason(view.result.current)).toBe("needsTts"))
    request.mockImplementation(async () => new Response(null, { status: 503 }))
    act(() => { window.dispatchEvent(new Event("focus")) })
    await waitFor(() => expect(voiceModelReason(view.result.current)).toBe("modelsUnavailable"))
    expect(view.result.current.asr.map(model => model.model_id)).toEqual(["live"])
    view.unmount()
  })
  it("never restores a previous login's models after a late response", async () => {
    let release!: (value: Response) => void
    request.mockImplementationOnce(() => new Promise<Response>(resolve => { release = resolve }))
    request.mockImplementationOnce(async () => response([]))
    const view = renderHook(() => useVoiceModels(true))
    auth.token = "token-b"; auth.session.sessionId = "login-b"
    view.rerender()
    await waitFor(() => expect(view.result.current.loading).toBe(false))
    await act(async () => { release(response([asr, tts])) })
    expect(voiceModelReason(view.result.current)).toBe("needsAsrAndTts")
    view.unmount()
  })
  it("keeps the current owner's models during background revalidation", async () => {
    request.mockResolvedValueOnce(response([asr, tts]))
    const view = renderHook(() => useVoiceModels(true))
    await waitFor(() => expect(view.result.current.tts).toHaveLength(1))
    let release!: (response: Response) => void
    request.mockImplementationOnce(() => new Promise<Response>(resolve => { release = resolve }))
    act(() => { window.dispatchEvent(new Event("focus")) })
    expect(view.result.current.loading).toBe(false)
    expect(view.result.current.tts.map(model => model.model_id)).toEqual(["speaker"])
    await act(async () => { release(response([asr])) })
    await waitFor(() => expect(view.result.current.tts).toHaveLength(0))
    view.unmount()
  })
})

it("uses running Xinference capabilities for arbitrary model UIDs", () => {
  const live = model({ model_id: "gpu-asr", model_provider: "xinference", model_name: "arbitrary-asr" })
  const speaker = model({ ...tts, model_id: "gpu-tts", model_provider: "xinference", model_name: "arbitrary-tts" })
  expect(voiceModelOptions([live, speaker])).toEqual({ asr: [], tts: [] })
  expect(voiceModelOptions([live, speaker], { "gpu-asr": { asr: true }, "gpu-tts": { tts: true } })).toEqual({ asr: [live], tts: [speaker] })
})

it("requires both streaming and server VAD from Xinference", async () => {
  const live = model({ model_id: "gpu-asr", model_provider: "xinference", model_name: "arbitrary-asr" })
  request.mockImplementation(async (url: string) => url.includes("/capabilities")
    ? new Response(JSON.stringify({ streaming: true, server_vad: false }), { status: 200 })
    : response([live]))
  const view = renderHook(() => useVoiceModels(true))
  await waitFor(() => expect(view.result.current.loading).toBe(false))
  expect(view.result.current.asr).toEqual([])
  view.unmount()
})

it("enables Xinference ASR and TTS after positive capability probes", async () => {
  const live = model({ model_id: "gpu-asr", model_provider: "xinference" })
  const speaker = model({ ...tts, model_id: "gpu-tts", model_provider: "xinference" })
  request.mockImplementation(async (url: string) => url.includes("/capabilities")
    ? new Response(JSON.stringify({ streaming: true, server_vad: true }), { status: 200 })
    : response([live, speaker]))
  const view = renderHook(() => useVoiceModels(true))
  await waitFor(() => expect(view.result.current.loading).toBe(false))
  expect(view.result.current.asr).toMatchObject([live])
  expect(view.result.current.tts).toMatchObject([speaker])
  expect(voiceModelReason(view.result.current)).toBeUndefined()
  view.unmount()
})

it("publishes available models during a slow probe and deduplicates tab return events", async () => {
  const live = model({ model_id: "gpu-asr", model_provider: "xinference" })
  let release!: (value: Response) => void
  request.mockImplementation((url: string) => url.includes("/capabilities")
    ? new Promise<Response>(resolve => { release = resolve })
    : Promise.resolve(response([asr, tts, live])))
  const view = renderHook(() => useVoiceModels(true))
  await waitFor(() => expect(view.result.current.tts).toMatchObject([tts]))
  expect(view.result.current.loading).toBe(false)
  expect(view.result.current.asr).toMatchObject([asr])
  act(() => {
    window.dispatchEvent(new Event("focus"))
    document.dispatchEvent(new Event("visibilitychange"))
  })
  expect(request).toHaveBeenCalledTimes(2)
  await act(async () => { release(new Response(JSON.stringify({ streaming: true, server_vad: true }))) })
  await waitFor(() => expect(view.result.current.asr).toMatchObject([asr, live]))
  expect(request).toHaveBeenCalledTimes(2)
  view.unmount()
})

it("caches probes briefly, retains prior capability on failure, and accepts a later negative result", async () => {
  let now = 1000
  vi.spyOn(Date, "now").mockImplementation(() => now)
  const live = model({ model_id: "gpu-asr", model_provider: "xinference" })
  let status = 200
  let streaming = true
  request.mockImplementation(async (url: string) => url.includes("/capabilities")
    ? new Response(JSON.stringify({ streaming, server_vad: true }), { status })
    : response([live, tts]))
  const view = renderHook(() => useVoiceModels(true))
  await waitFor(() => expect(view.result.current.asr).toMatchObject([live]))
  act(() => { window.dispatchEvent(new Event("focus")) })
  await waitFor(() => expect(request).toHaveBeenCalledTimes(3))
  now += 31000; status = 503
  await act(async () => { window.dispatchEvent(new Event("focus")) })
  await waitFor(() => expect(request).toHaveBeenCalledTimes(5))
  expect(view.result.current.asr).toMatchObject([live])
  expect(view.result.current.error).toBe(false)
  status = 200; streaming = false
  await act(async () => { window.dispatchEvent(new Event("focus")) })
  await waitFor(() => expect(view.result.current.asr).toEqual([]))
  expect(voiceModelReason(view.result.current)).toBe("needsAsr")
  view.unmount()
})

it("invalidates cached capabilities when the model endpoint changes", async () => {
  let live = model({ model_id: "gpu-asr", model_provider: "xinference" })
  request.mockImplementation(async (url: string) => url.includes("/capabilities")
    ? new Response(JSON.stringify({ streaming: !live.base_url, server_vad: true }))
    : response([live, tts]))
  const view = renderHook(() => useVoiceModels(true))
  await waitFor(() => expect(view.result.current.asr).toMatchObject([live]))
  live = { ...live, base_url: "https://other.example" }
  await act(async () => { window.dispatchEvent(new Event("focus")) })
  await waitFor(() => expect(view.result.current.asr).toEqual([]))
  expect(request.mock.calls.filter(([url]) => url.includes("/capabilities"))).toHaveLength(2)
  view.unmount()
})
