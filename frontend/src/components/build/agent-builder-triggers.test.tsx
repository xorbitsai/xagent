import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
// Stable references across renders — an inline `t: (key) => key` (or a fresh
// `apps: []`/`dispatch: vi.fn()`) recreated on every useI18n()/useMcpApps()/
// useApp() call defeats every useCallback/useMemo keyed on it throughout the
// real component tree (e.g. AgentTriggersDialog's loadRunsFor depends on
// `t`), making dependent effects re-run on every render instead of only when
// something real changed.
const translateMock = vi.hoisted(() => (key: string) => key)
const mcpAppsMock = vi.hoisted(() => ({ apps: [] as unknown[], getAppIcon: () => null }))
const appContextMock = vi.hoisted(() => ({
  state: {
    messages: [],
    traceEvents: [],
    currentTask: null,
    isProcessing: false,
    isHistoryLoading: false,
    taskId: null,
    filePreview: { isOpen: false },
    dagExecution: null,
    steps: [],
  },
  setTaskId: vi.fn(),
  sendMessage: vi.fn(),
  dispatch: vi.fn(),
  closeFilePreview: vi.fn(),
  pauseTask: vi.fn(),
  resumeTask: vi.fn(),
  openFilePreview: vi.fn(),
  requestStatus: vi.fn(),
}))

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return {
    ...actual,
    apiRequest: apiRequestMock,
  }
})

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
    getUploadApiUrl: () => "http://api.local",
    getWsUrl: () => "ws://api.local",
  }
})

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => appContextMock,
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token" }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: "en",
    t: translateMock,
  }),
}))

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => mcpAppsMock,
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => ({ get: () => null }),
}))

vi.mock("@/components/layout/resizable-three-column-layout", () => ({
  ResizableThreeColumnLayout: ({ middlePanel, rightPanel }: { middlePanel: React.ReactNode; rightPanel: React.ReactNode }) => (
    <div>
      <div data-testid="middle-panel">{middlePanel}</div>
      <div data-testid="right-panel">{rightPanel}</div>
    </div>
  ),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => null,
}))

vi.mock("@/components/build/agent-builder-chat", () => ({
  AgentBuilderChat: () => null,
}))

vi.mock("@/components/kb/knowledge-base-creation-dialog", () => ({
  KnowledgeBaseCreationDialog: () => null,
}))

vi.mock("@/components/mcp/connect-mcp-dialog", () => ({
  ConnectMcpDialog: () => null,
}))

vi.mock("@/components/chat/FileMentionDropdown", () => ({
  FileMentionDropdown: () => null,
}))

vi.mock("@/hooks/use-file-mention", () => ({
  useFileMention: () => ({
    checkTrigger: vi.fn(),
    isOpen: false,
    items: [],
    selectedIndex: 0,
    selectItem: vi.fn(),
    close: vi.fn(),
  }),
}))

vi.mock("@/components/ui/multi-select", () => ({
  MultiSelect: () => null,
}))

vi.mock("@/components/ui/select", () => ({
  Select: () => null,
}))

vi.mock("@/components/build/build-file-preview-sheet", () => ({
  BuildFilePreviewSheet: () => null,
}))

import { AgentBuilder } from "./agent-builder"
import type { AgentTrigger } from "@/lib/agent-triggers-api"

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

function makeTrigger(overrides: Partial<AgentTrigger> & { id: number }): AgentTrigger {
  return {
    user_id: 1,
    agent_id: 42,
    type: "webhook",
    name: "Trigger",
    enabled: true,
    config: {},
    prompt_template: null,
    webhook_token: "tok",
    webhook_secret: null,
    next_run_at: null,
    last_run_at: null,
    last_error: null,
    created_at: null,
    updated_at: null,
    ...overrides,
  }
}

const TRIGGERS_URL = "http://api.local/api/agents/42/triggers"
const GMAIL_ACCOUNTS_URL = "http://api.local/api/cloud/accounts?provider=gmail"

describe("AgentBuilder trigger summary cards", () => {
  const originalWebSocket = globalThis.WebSocket

  beforeEach(() => {
    apiRequestMock.mockReset()
    globalThis.WebSocket = vi.fn() as unknown as typeof WebSocket

    let triggers = [makeTrigger({ id: 9, name: "API / Webhook" })]

    apiRequestMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === "http://api.local/api/agents/42") {
        return Promise.resolve(
          jsonResponse({
            id: 42,
            name: "Trigger agent",
            description: "",
            instructions: "",
            execution_mode: "balanced",
            suggested_prompts: [],
            visibility: "team",
            team_id: null,
            knowledge_bases: [],
            skills: [],
            tool_categories: [],
            can_edit: true,
          }),
        )
      }
      if (url === TRIGGERS_URL) {
        return Promise.resolve(jsonResponse(triggers))
      }
      if (url === `${TRIGGERS_URL}/9` && init?.method === "PATCH") {
        triggers = triggers.map((item) => ({ ...item, enabled: false }))
        return Promise.resolve(jsonResponse(triggers[0]))
      }
      if (url.endsWith("/api/kb/collections")) {
        return Promise.resolve(jsonResponse({ collections: [] }))
      }
      if (url.endsWith("/api/tools/available")) {
        return Promise.resolve(jsonResponse({ tools: [] }))
      }
      if (url.endsWith("/api/skills/") || url.endsWith("/api/mcp/servers")) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url.endsWith("/api/models/?category=llm") || url.endsWith("/api/models/user-default")) {
        return Promise.resolve(jsonResponse([]))
      }
      return Promise.resolve(jsonResponse({}))
    })
  })

  afterEach(() => {
    cleanup()
    globalThis.WebSocket = originalWebSocket
  })

  it("disables the trigger type in place when its card switch is toggled off", async () => {
    render(<AgentBuilder agentId="42" />)

    // The webhook summary card shows up once the trigger list loads.
    expect(await screen.findByText("triggers.cards.webhook.title")).toBeInTheDocument()

    const cardSwitch = screen
      .getAllByRole("switch")
      .find((el) => el.getAttribute("aria-checked") === "true")
    expect(cardSwitch).toBeDefined()
    fireEvent.click(cardSwitch!)

    // Toggling off patches the trigger directly instead of opening the dialog.
    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        `${TRIGGERS_URL}/9`,
        expect.objectContaining({
          method: "PATCH",
          body: JSON.stringify({ enabled: false }),
        }),
      )
    })
    expect(screen.queryByText("triggers.subtitle")).not.toBeInTheDocument()

    // The card disappears after the refreshed summary reports 0 enabled.
    await waitFor(() => {
      expect(screen.queryByText("triggers.cards.webhook.title")).not.toBeInTheDocument()
    })
  })

  it("resyncs the summary via refreshTriggerSummary when a batch disable partially fails", async () => {
    let triggers = [
      makeTrigger({ id: 9, name: "Hook A" }),
      makeTrigger({ id: 10, name: "Hook B" }),
    ]
    let getCallsAfterFailure = 0
    let patchAttempted = false

    apiRequestMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === "http://api.local/api/agents/42") {
        return Promise.resolve(
          jsonResponse({
            id: 42,
            name: "Trigger agent",
            description: "",
            instructions: "",
            execution_mode: "balanced",
            suggested_prompts: [],
            visibility: "team",
            team_id: null,
            knowledge_bases: [],
            skills: [],
            tool_categories: [],
            can_edit: true,
          }),
        )
      }
      if (url === TRIGGERS_URL && (!init?.method || init.method === "GET")) {
        if (patchAttempted) getCallsAfterFailure += 1
        return Promise.resolve(jsonResponse(triggers))
      }
      if (url === `${TRIGGERS_URL}/9` && init?.method === "PATCH") {
        triggers = triggers.map((item) => (item.id === 9 ? { ...item, enabled: false } : item))
        return Promise.resolve(jsonResponse(triggers[0]))
      }
      if (url === `${TRIGGERS_URL}/10` && init?.method === "PATCH") {
        patchAttempted = true
        return Promise.reject(new Error("boom"))
      }
      if (url.endsWith("/api/kb/collections")) {
        return Promise.resolve(jsonResponse({ collections: [] }))
      }
      if (url.endsWith("/api/tools/available")) {
        return Promise.resolve(jsonResponse({ tools: [] }))
      }
      if (url.endsWith("/api/skills/") || url.endsWith("/api/mcp/servers")) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url.endsWith("/api/models/?category=llm") || url.endsWith("/api/models/user-default")) {
        return Promise.resolve(jsonResponse([]))
      }
      return Promise.resolve(jsonResponse({}))
    })

    render(<AgentBuilder agentId="42" />)

    const cardSwitch = (
      await screen.findAllByRole("switch")
    ).find((el) => el.getAttribute("aria-checked") === "true")
    expect(cardSwitch).toBeDefined()
    fireEvent.click(cardSwitch!)

    // One PATCH in the batch rejected: disableTriggerType's catch resyncs via
    // refreshTriggerSummary (a fresh GET) instead of trusting the optimistic
    // merge, which would otherwise wrongly report both hooks disabled.
    await waitFor(() => {
      expect(getCallsAfterFailure).toBeGreaterThan(0)
    })
  })
})

describe("AgentBuilder trigger summary cards (agent not created yet)", () => {
  const originalWebSocket = globalThis.WebSocket

  beforeEach(() => {
    apiRequestMock.mockReset()
    globalThis.WebSocket = vi.fn() as unknown as typeof WebSocket

    apiRequestMock.mockImplementation((url: string) => {
      if (url === GMAIL_ACCOUNTS_URL) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url.endsWith("/api/kb/collections")) {
        return Promise.resolve(jsonResponse({ collections: [] }))
      }
      if (url.endsWith("/api/tools/available")) {
        return Promise.resolve(jsonResponse({ tools: [] }))
      }
      if (url.endsWith("/api/skills/") || url.endsWith("/api/mcp/servers")) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url.endsWith("/api/models/?category=llm") || url.endsWith("/api/models/user-default")) {
        return Promise.resolve(jsonResponse([]))
      }
      return Promise.resolve(jsonResponse({}))
    })
  })

  afterEach(() => {
    cleanup()
    globalThis.WebSocket = originalWebSocket
  })

  it("disables a staged trigger type in place, without any network call, before the agent exists", async () => {
    render(<AgentBuilder />)

    // Stage an enabled webhook trigger via the dialog (no agentId yet, so
    // creation only touches the parent-owned staged list, no API call).
    fireEvent.click(await screen.findByText("triggers.builder.open"))
    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    await screen.findByLabelText("triggers.form.name")
    const [detailSwitch] = screen.getAllByRole("switch")
    fireEvent.click(detailSwitch)
    await waitFor(() => {
      expect(detailSwitch).toHaveAttribute("aria-checked", "true")
    })
    fireEvent.click(screen.getByRole("button", { name: "common.done" }))

    // The summary card appears once the staged trigger is enabled.
    expect(await screen.findByText("triggers.cards.webhook.title")).toBeInTheDocument()
    apiRequestMock.mockClear()

    const cardSwitch = screen
      .getAllByRole("switch")
      .find((el) => el.getAttribute("aria-checked") === "true")
    expect(cardSwitch).toBeDefined()
    fireEvent.click(cardSwitch!)

    // disableTriggerType's `!localAgentId` branch patches stagedTriggers
    // directly — no PATCH/GET to any trigger endpoint.
    await waitFor(() => {
      expect(screen.queryByText("triggers.cards.webhook.title")).not.toBeInTheDocument()
    })
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      expect.stringContaining("/triggers"),
      expect.anything(),
    )
  })
})
