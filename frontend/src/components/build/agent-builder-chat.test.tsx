import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return {
    ...actual,
    apiRequest: apiRequestMock,
  }
})

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
  getUploadApiUrl: () => "http://api.local",
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token" }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string>) =>
      vars?.appName ? `${key}:${vars.appName}` : key,
  }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("sonner", () => ({
  toast: {
    error: toastErrorMock,
  },
}))

vi.mock("@/components/chat/ChatInput", () => ({
  ChatInput: ({
    onSend,
    files = [],
    onFilesChange,
  }: {
    onSend?: (message: string) => void | Promise<void>
    files?: File[]
    onFilesChange?: (files: File[]) => void
  }) => (
    <div>
      <button
        type="button"
        onClick={() =>
          onFilesChange?.([
            ...files,
            new File(["chat-input"], "chat-input.txt", { type: "text/plain" }),
          ])
        }
      >
        attach-chat-input-file
      </button>
      <button type="button" onClick={() => onSend?.("chat input message")}>
        send-chat-input
      </button>
    </div>
  ),
}))

vi.mock("@/components/chat/ChatMessage", () => ({
  ChatMessage: ({ onSendInteraction }: { onSendInteraction?: (text: string, files?: File[]) => void }) => (
    onSendInteraction ? (
      <button
        type="button"
        onClick={() => onSendInteraction("upload this", [
          new File(["data"], "data.txt", { type: "text/plain" }),
        ])}
      >
        send-file-interaction
      </button>
    ) : (
      <div>message</div>
    )
  ),
}))

vi.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
    ({ children, ...props }, ref) => (
      <div ref={ref} {...props}>
        {children}
      </div>
    )
  ),
}))

vi.mock("@/components/file/file-attachment", () => ({
  FileAttachment: () => <div>attachment</div>,
}))

vi.mock("lucide-react", () => ({
  Bot: (props: React.SVGProps<SVGSVGElement>) => <svg {...props} />,
}))

import { AgentBuilderChat, type AgentConfig } from "./agent-builder-chat"

const agentConfig: AgentConfig = {
  name: "Demo",
  description: "Demo",
  instructions: "Help",
  executionMode: "balanced",
  suggestedPrompts: [],
  selectedToolCategories: [],
  modelConfig: {
    general: null,
    small_fast: null,
    visual: null,
    compact: null,
  },
}

describe("AgentBuilderChat", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("shows backend upload error details when file upload is unavailable", async () => {
    apiRequestMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "Startup file storage sync failed" }), {
        status: 503,
        statusText: "Service Unavailable",
        headers: { "Content-Type": "application/json" },
      })
    )

    render(
      <AgentBuilderChat
        agentConfig={agentConfig}
        onUpdateConfig={vi.fn()}
      />
    )

    fireEvent.click(await screen.findByText("send-file-interaction"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "Startup file storage sync failed"
      )
    })
  })
})
