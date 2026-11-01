import { act, renderHook, waitFor } from "@testing-library/react"
import { expect, it, vi } from "vitest"
const auth = vi.hoisted(() => ({ token: "token", user: { id: 1 }, session: { sessionId: "first" } }))
const fetchPreferences = vi.hoisted(() => vi.fn())
vi.mock("@/contexts/auth-context", () => ({ useAuth: () => auth }))
vi.mock("@/lib/user-preferences", () => ({ fetchUserPreferences: fetchPreferences }))
import { useVoicePreferences } from "./use-voice-preferences"

it("loads only when opened and discards the previous account's late preferences", async () => {
  let release!: (value: unknown) => void
  fetchPreferences.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    .mockResolvedValueOnce({ voice_conversation: { tts_model_id: "new-speaker" } })
  const view = renderHook(({ open }) => useVoicePreferences(open), { initialProps: { open: false } })
  expect(fetchPreferences).not.toHaveBeenCalled()
  view.rerender({ open: true })
  auth.user.id = 2; auth.session.sessionId = "second"
  view.rerender({ open: true })
  await waitFor(() => expect(view.result.current.preferences).toEqual({ tts_model_id: "new-speaker" }))
  await act(async () => { release({ voice_conversation: { tts_model_id: "old-speaker" } }) })
  expect(view.result.current.preferences).toEqual({ tts_model_id: "new-speaker" })
  view.unmount()
})
