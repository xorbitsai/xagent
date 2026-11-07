import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))
import { VoiceModeButton } from "./voice-mode-button"
afterEach(cleanup)
it("keeps a disabled entry discoverable by mouse and keyboard with its exact requirement", async () => {
  const open = vi.fn()
  render(<VoiceModeButton reason="needsAsr" onClick={open} />)
  const button = screen.getByRole("button", { name: "voiceConversation.open" })
  expect(button).toBeDisabled()
  expect(button).toHaveAccessibleDescription("voiceConversation.needsAsr")
  expect(button.parentElement).toHaveAttribute("title", "voiceConversation.needsAsr")
  expect(button.parentElement).toHaveAttribute("tabindex", "0")
  fireEvent.click(button)
  expect(open).not.toHaveBeenCalled()
  fireEvent.focus(button.parentElement!)
  expect(await screen.findByRole("tooltip")).toHaveTextContent("voiceConversation.needsAsr")
})
