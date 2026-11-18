import React from "react"
import { renderToString } from "react-dom/server"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("./use-widget-session", async () => ({
  ...await vi.importActual<typeof import("./use-widget-session")>("./use-widget-session"),
  useWidgetSession: () => ({
    status: "connecting",
    session: null,
    agent: null,
    terminalCode: null,
    isAbsoluteExpiryWarningVisible: false,
    handleSessionExpired: () => undefined,
    handleConnectionClose: () => "handled" as const,
    handleConnectionFailure: () => undefined,
    handleConnectionOpen: () => undefined,
    ensureFreshSession: () => Promise.resolve(null),
    refreshSession: () => Promise.resolve(null),
    waitForOpenConnection: () => Promise.resolve(),
  }),
}))
vi.mock("@/contexts/app-context-chat", () => ({
  AppProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useApp: () => ({
    state: { taskId: null, currentTask: null, isProcessing: false, messages: [] },
    filesDisabled: true,
    voiceInputEnabled: false,
    isConnected: false,
    isSendInFlight: false,
    canStopTask: false,
    stopTask: () => undefined,
    sendMessage: () => undefined,
    startNewConversation: () => Promise.resolve(),
    isConversationResetPending: false,
    isMessageDeliveryPending: false,
    sessionConversationState: "unbound",
    sessionConversationAttachment: "attached",
    isSessionInteractionLocked: false,
  }),
}))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))

import { SessionAgentChatPage } from "./session-agent-chat-page"

describe("SessionAgentChatPage rendered on the server", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  // The standalone Node server renders this page for /widget/chat/session,
  // where no window exists.
  it("renders without a window", () => {
    vi.stubGlobal("window", undefined)

    expect(() => renderToString(<SessionAgentChatPage />)).not.toThrow()
  })
})
