/// <reference types="@testing-library/jest-dom/vitest" />

import React from "react"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api-wrapper")>()),
  apiRequest: apiRequestMock,
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({ openFilePreview: vi.fn() }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: { id: "guest", is_admin: false } }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

vi.mock("@/components/config-dialog", () => ({
  ConfigDialog: ({ trigger }: { trigger: React.ReactNode }) => trigger,
}))

vi.mock("@/components/voice-input-controller", () => ({
  useVoiceInputControls: () => ({ isListening: false, toggle: vi.fn(), supported: false }),
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn() } }))

import { ChatInput } from "./ChatInput"
import {
  FileAccessProvider,
  createPublicFileAccessPolicy,
} from "@/contexts/file-access-context"

describe("ChatInput public file access", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    localStorage.setItem("auth_cache", JSON.stringify({
      token: "personal-access-token",
      refreshToken: "personal-refresh-token",
      user: { id: "personal-user" },
    }))
    vi.useFakeTimers()
  })

  afterEach(() => {
    cleanup()
    localStorage.clear()
    vi.useRealTimers()
  })

  it("keeps public upload enabled while fail-closing @ file mentions against ambient auth", async () => {
    apiRequestMock.mockResolvedValue(new Response(JSON.stringify({
      items: [{ file_id: "private-file-id", filename: "private-report.pdf" }],
    }), {
      headers: { "Content-Type": "application/json" },
    }))

    const { container } = render(
      <FileAccessProvider policy={createPublicFileAccessPolicy("guest-token")}>
        <ChatInput hideConfig readOnlyConfig onSend={vi.fn()} />
      </FileAccessProvider>,
    )

    expect(document.querySelector('input[type="file"]')).not.toBeNull()
    const editor = screen.getByRole("textbox")
    editor.textContent = "@personal-file"
    const textNode = editor.firstChild
    const range = document.createRange()
    range.setStart(textNode!, "@personal-file".length)
    range.collapse(true)
    const selection = window.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range)

    fireEvent.input(editor)
    await act(async () => {
      vi.advanceTimersByTime(200)
      await Promise.resolve()
    })

    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(container).not.toHaveTextContent("private-report.pdf")
    expect(container.innerHTML).not.toContain("private-file-id")
    expect(localStorage.getItem("auth_cache")).toContain("personal-access-token")
  })
})
