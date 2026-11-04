import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

const request = vi.hoisted(() => vi.fn())
const save = vi.hoisted(() => vi.fn())
const saved = vi.hoisted(() => ({ preferences: { asr_model_id: "live", tts_model_id: "speaker", voice: "voice-one" }, loading: false, error: false }))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))
vi.mock("@/hooks/use-voice-preferences", () => ({ useVoicePreferences: () => saved }))
vi.mock("@/hooks/use-voice-models", async original => ({
  ...await original<typeof import("@/hooks/use-voice-models")>(),
  useVoiceModels: () => ({ asr: [{ id: 1, model_id: "live", model_name: "scribe_v2_realtime" }], tts: [{ id: 2, model_id: "speaker", model_name: "eleven_v3" }, { id: 3, model_id: "other", model_name: "eleven_flash_v2_5" }], loading: false, error: false }),
}))
vi.mock("@/lib/api-wrapper", () => ({ apiRequest: request }))
vi.mock("@/lib/user-preferences", () => ({ updateUserPreferences: save }))
import { VoiceSettings } from "./voice-settings"

beforeEach(() => {
  saved.error = false
  saved.preferences = { asr_model_id: "live", tts_model_id: "speaker", voice: "voice-one" }
  request.mockReset().mockResolvedValue(new Response(JSON.stringify([{ voice_id: "voice-one", name: "Rachel" }, { voice_id: "voice-two", name: "Adam" }])))
  save.mockReset().mockResolvedValue({ ok: true })
})
afterEach(cleanup)

it("loads named voices and saves a separate account preference", async () => {
  render(<VoiceSettings />)
  await waitFor(() => expect(screen.getByRole("option", { name: "Adam" })).toBeInTheDocument())
  expect(screen.getByLabelText("settings.voice.speaker")).toHaveValue("voice-one")
  expect(request).toHaveBeenCalledWith(expect.stringContaining("/speech/voices?model_id=speaker"))
  fireEvent.change(screen.getByLabelText("settings.voice.speaker"), { target: { value: "voice-two" } })
  fireEvent.click(screen.getByRole("button", { name: "common.save" }))
  await waitFor(() => expect(save).toHaveBeenCalledWith({ voice_conversation: { asr_model_id: "live", tts_model_id: "speaker", voice: "voice-two" } }))
  expect(await screen.findByText("settings.voice.saved")).toBeInTheDocument()
})

it("resets the provider voice when changing the speaking model and ignores the old list", async () => {
  let release!: (response: Response) => void
  request.mockReset().mockImplementationOnce(() => new Promise<Response>(resolve => { release = resolve }))
    .mockResolvedValueOnce(new Response(JSON.stringify([{ voice_id: "other-voice", name: "Other voice" }])))
  render(<VoiceSettings />)
  await waitFor(() => expect(screen.getByLabelText("voiceConversation.speechModel")).toHaveValue("speaker"))
  fireEvent.change(screen.getByLabelText("voiceConversation.speechModel"), { target: { value: "other" } })
  await screen.findByRole("option", { name: "Other voice" })
  release(new Response(JSON.stringify([{ voice_id: "stale", name: "Old voice" }])))
  await waitFor(() => expect(screen.getByLabelText("settings.voice.speaker")).toHaveValue(""))
  expect(screen.queryByRole("option", { name: "Old voice" })).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole("button", { name: "common.save" }))
  await waitFor(() => expect(save).toHaveBeenCalledWith({ voice_conversation: { asr_model_id: "live", tts_model_id: "other", voice: null } }))
})

it("does not overwrite unknown preferences after a load failure", () => {
  saved.error = true
  render(<VoiceSettings />)
  expect(screen.getByRole("button", { name: "common.save" })).toBeDisabled()
  expect(screen.getByRole("alert")).toHaveTextContent("settings.voice.loadFailed")
})

it("shows failed saves and keeps the model default available if voice listing fails", async () => {
  request.mockReset().mockResolvedValue(new Response(null, { status: 502 }))
  save.mockResolvedValue({ ok: false })
  render(<VoiceSettings />)
  await screen.findByText("settings.voice.voicesFailed")
  fireEvent.change(screen.getByLabelText("settings.voice.speaker"), { target: { value: "" } })
  fireEvent.click(screen.getByRole("button", { name: "common.save" }))
  expect(await screen.findByText("settings.voice.saveFailed")).toBeInTheDocument()
})

it("preserves an unchanged saved voice when listing fails while changing the listening model", async () => {
  request.mockReset().mockResolvedValue(new Response(null, { status: 502 }))
  render(<VoiceSettings />)
  await screen.findByText("settings.voice.voicesFailed")
  expect(screen.getByLabelText("settings.voice.speaker")).toHaveValue("voice-one")
  fireEvent.change(screen.getByLabelText("voiceConversation.transcriptionModel"), { target: { value: "" } })
  expect(screen.getByRole("button", { name: "common.save" })).toBeEnabled()
  fireEvent.click(screen.getByRole("button", { name: "common.save" }))
  await waitFor(() => expect(save).toHaveBeenCalledWith({ voice_conversation: { asr_model_id: null, tts_model_id: "speaker", voice: "voice-one" } }))
})
