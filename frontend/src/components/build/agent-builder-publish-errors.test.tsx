import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// Issue #969: the non-update builder actions (publish, unpublish, publish from
// the creation success dialog, optimize instructions) must never pass a raw
// `detail` payload to toast.error. sonner is mocked here, so each case asserts
// two things: the exact string handed to the toaster is displayable, and the
// builder is still mounted afterwards.

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return { ...actual, apiRequest: apiRequestMock }
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
  useApp: () => ({
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
  }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token", user: { id: "1", is_admin: false } }),
}))

// The i18n return value must be referentially stable: AgentSshBindings keys a
// fetch effect on `t`, so a per-render `t` identity turns that effect into an
// unbounded fetch/render loop under jsdom.
vi.mock("@/contexts/i18n-context", () => {
  const i18n = {
    locale: "en",
    t: (key: string, vars?: Record<string, string>) =>
      vars?.appName ? `${key}:${vars.appName}` : key,
  }
  return { useI18n: () => i18n }
})

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({ apps: [], getAppIcon: () => null }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("sonner", () => ({ toast: { error: toastErrorMock, success: vi.fn() } }))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => ({ get: () => null }),
}))

vi.mock("@/components/layout/resizable-three-column-layout", () => ({
  ResizableThreeColumnLayout: ({ middlePanel }: { middlePanel: React.ReactNode }) => (
    <div>{middlePanel}</div>
  ),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => null,
}))

vi.mock("@/components/build/agent-builder-chat", () => ({ AgentBuilderChat: () => null }))
vi.mock("@/components/kb/knowledge-base-creation-dialog", () => ({
  KnowledgeBaseCreationDialog: () => null,
}))
vi.mock("@/components/mcp/connect-mcp-dialog", () => ({
  ConnectMcpDialog: () => null,
}))
vi.mock("@/components/chat/FileMentionDropdown", () => ({ FileMentionDropdown: () => null }))
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
  MultiSelect: () => <div data-testid="multi-select" />,
}))
// The model Select is the only place `modelConfig.general` reaches the DOM
// (agent-builder.tsx renders it at the `models.length > 0` branch of the config
// form). Mirroring the value onto data-value gives the create flow a readiness
// signal it can wait on instead of retrying clicks.
vi.mock("@/components/ui/select", () => ({
  Select: ({ value }: { value?: string }) => (
    <div data-testid="model-select" data-value={value ?? ""} />
  ),
}))
vi.mock("@/components/build/build-file-preview-sheet", () => ({
  BuildFilePreviewSheet: () => null,
}))

import { AgentBuilder } from "./agent-builder"

const AGENT_ID = "5"

function agentResponse(status: "draft" | "published") {
  return {
    id: Number(AGENT_ID),
    user_id: 1,
    team_id: null,
    name: "Existing Agent",
    description: "",
    instructions: "You are an existing agent.",
    execution_mode: "balanced",
    models: { general: "10" },
    knowledge_bases: [],
    skills: [],
    tool_categories: ["basic"],
    suggested_prompts: [],
    logo_url: null,
    status,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    widget_enabled: false,
    allowed_domains: [],
    share_enabled: false,
    share_updated_at: null,
    can_edit: true,
  }
}

// Error payload matrix shared by every action: each case must surface
// `expected` (with `FALLBACK` replaced by the action-specific i18n key)
// instead of handing the raw payload to the toaster.
const FALLBACK = "__ACTION_FALLBACK__"

type ErrorCase = {
  name: string
  body: string | null
  expected: string
  // Default to a FastAPI validation failure; a case overrides these when the
  // transport shape itself is part of the fixture.
  status?: number
  contentType?: string
}

const ERROR_CASES: ErrorCase[] = [
  {
    name: "a plain string detail",
    body: JSON.stringify({ detail: " Action failed with string detail " }),
    expected: "Action failed with string detail",
  },
  {
    name: "a structured detail message",
    body: JSON.stringify({ detail: { message: "Action failed", context: [] } }),
    expected: "Action failed",
  },
  {
    name: "a structured detail msg",
    body: JSON.stringify({ detail: { msg: "  Action failed with detail msg  " } }),
    expected: "Action failed with detail msg",
  },
  {
    name: "a detail msg alongside a top-level message",
    body: JSON.stringify({
      detail: { msg: "Detail wins" },
      message: "Top-level loses",
    }),
    expected: "Detail wins",
  },
  {
    name: "a detail object without a readable message",
    body: JSON.stringify({ detail: { code: 123 } }),
    expected: FALLBACK,
  },
  {
    name: "FastAPI validation detail messages",
    body: JSON.stringify({
      detail: [
        { msg: " Field is required " },
        " Invalid value ",
        { message: " Unsupported option " },
        { msg: " " },
      ],
    }),
    expected: "Field is required; Invalid value; Unsupported option",
  },
  {
    name: "a detail array without readable entries",
    body: JSON.stringify({ detail: [1, true, null, { msg: " " }] }),
    expected: FALLBACK,
  },
  {
    name: "a bare top-level message",
    body: JSON.stringify({ message: "  Top-level failure  " }),
    expected: "Top-level failure",
  },
  {
    name: "an empty response body",
    body: null,
    expected: FALLBACK,
  },
  {
    name: "a non-JSON response body",
    body: "<html>Bad Gateway</html>",
    expected: FALLBACK,
    status: 502,
    contentType: "text/html",
  },
]

// The one model the create flow resolves to, both as the available-model list
// entry and as the user's general default.
const DEFAULT_MODEL = {
  id: 10,
  model_id: "test/model",
  model_name: "Test Model",
  model_provider: "test",
  category: "llm",
}

function jsonResponse(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), { status })
}

function errorResponse(errorCase: ErrorCase) {
  return new Response(errorCase.body, {
    status: errorCase.status ?? 422,
    headers: { "Content-Type": errorCase.contentType ?? "application/json" },
  })
}

type ApiOverride = (
  url: string,
  opts?: { method?: string }
) => Promise<Response> | null

// Mount-time API surface every suite needs to resolve before the builder
// settles. `overrides` is consulted first and returns null to fall through, so
// a suite only states what it actually changes.
function installBaseApiMocks(overrides: ApiOverride) {
  apiRequestMock.mockImplementation((url: string, opts?: { method?: string }) => {
    const override = overrides(url, opts)
    if (override) return override
    if (url.endsWith("/api/kb/collections"))
      return Promise.resolve(jsonResponse({ collections: [] }))
    if (url.endsWith("/api/skills/")) return Promise.resolve(jsonResponse([]))
    if (url.endsWith("/api/tools/available"))
      return Promise.resolve(jsonResponse({ tools: [] }))
    if (url.endsWith("/api/models/?category=llm"))
      return Promise.resolve(jsonResponse([]))
    if (url.includes(`/api/agents/${AGENT_ID}/triggers`))
      return Promise.resolve(jsonResponse([]))
    if (url.includes("/api/mcp/servers")) return Promise.resolve(jsonResponse([]))
    return Promise.resolve(jsonResponse({}))
  })
}

// The transport-failure suite injects its rejection by wrapping the installed
// implementation, so it needs a failing path that never matches.
const NO_FAILING_RESPONSE: ErrorCase = {
  name: "unused",
  body: null,
  expected: FALLBACK,
}

function installEditModeApi(
  status: "draft" | "published",
  failingPath: string,
  failingCase: ErrorCase
) {
  installBaseApiMocks((url, opts) => {
    if (opts?.method === "POST" && url.endsWith(failingPath))
      return Promise.resolve(errorResponse(failingCase))
    if (url.endsWith("/api/models/user-default"))
      return Promise.resolve(jsonResponse([]))
    if (url.endsWith(`/api/agents/${AGENT_ID}`))
      return Promise.resolve(jsonResponse(agentResponse(status)))
    return null
  })
}

// Create-mode mock for the success-dialog publish path: agent creation
// succeeds (which opens the dialog), publish fails with the payload under test.
// The model list and the user default both resolve to DEFAULT_MODEL, so the
// builder reaches a creatable state without any model interaction.
function installCreateModeApi(failingCase: ErrorCase) {
  installBaseApiMocks((url, opts) => {
    if (opts?.method === "POST" && url.endsWith(`/api/agents/${AGENT_ID}/publish`))
      return Promise.resolve(errorResponse(failingCase))
    if (opts?.method === "POST" && url.endsWith("/api/agents"))
      return Promise.resolve(jsonResponse(agentResponse("draft")))
    if (url.endsWith("/api/models/?category=llm"))
      return Promise.resolve(jsonResponse([DEFAULT_MODEL]))
    if (url.endsWith("/api/models/user-default"))
      return Promise.resolve(
        jsonResponse([{ config_type: "general", model: { id: DEFAULT_MODEL.id } }])
      )
    return null
  })
}

async function waitForLoadedBuilder() {
  await waitFor(() =>
    expect(
      screen.getByPlaceholderText("builds.configForm.name.placeholder")
    ).toHaveValue("Existing Agent")
  )
}

// Exactly one error toast per failure: a second call would mean the action ran
// twice, or that a validation toast leaked in before the action under test.
async function expectToast(expected: string) {
  await waitFor(() => {
    expect(toastErrorMock).toHaveBeenCalledTimes(1)
    expect(toastErrorMock.mock.calls.at(-1)?.[0]).toBe(expected)
  })
}

beforeEach(() => {
  apiRequestMock.mockReset()
  toastErrorMock.mockReset()
  globalThis.WebSocket = vi.fn() as unknown as typeof WebSocket
})

afterEach(() => cleanup())

// The three edit-mode actions differ only in which agent status they load,
// which POST fails, which control triggers them and which localized fallback
// they own.
const EDIT_MODE_ACTIONS = [
  {
    actionName: "publish",
    status: "draft",
    failingPath: `/api/agents/${AGENT_ID}/publish`,
    clickText: "builds.editor.header.publish",
    fallbackKey: "builds.publication.publishFailed",
  },
  {
    actionName: "unpublish",
    status: "published",
    failingPath: `/api/agents/${AGENT_ID}/unpublish`,
    clickText: "builds.editor.header.unpublish",
    fallbackKey: "builds.publication.unpublishFailed",
  },
  {
    actionName: "optimize instructions",
    status: "draft",
    failingPath: "/api/agents/optimize-instructions",
    clickText: "builds.configForm.instructions.optimize",
    fallbackKey: "builds.configForm.instructions.optimizeError",
  },
] as const

describe.each(EDIT_MODE_ACTIONS)(
  "AgentBuilder $actionName error handling (issue #969)",
  ({ status, failingPath, clickText, fallbackKey }) => {
    it.each(ERROR_CASES)(
      "surfaces a displayable message for $name",
      async (errorCase) => {
        installEditModeApi(status, failingPath, errorCase)
        render(<AgentBuilder agentId={AGENT_ID} />)
        await waitForLoadedBuilder()

        fireEvent.click(screen.getByText(clickText))

        await expectToast(
          errorCase.expected === FALLBACK ? fallbackKey : errorCase.expected
        )
        expect(screen.getByDisplayValue("Existing Agent")).toBeInTheDocument()
      }
    )
  }
)

describe("AgentBuilder network failure handling (issue #969)", () => {
  it("uses the generic fallback when the publish request itself rejects", async () => {
    // A transport-level rejection must not be swallowed by the response-body
    // parsing path: it hits the outer catch and shows the generic fallback.
    installEditModeApi("draft", "__no_failing_path__", NO_FAILING_RESPONSE)
    const base = apiRequestMock.getMockImplementation()!
    apiRequestMock.mockImplementation((url: string, opts?: { method?: string }) => {
      if (opts?.method === "POST" && url.endsWith(`/api/agents/${AGENT_ID}/publish`))
        return Promise.reject(new TypeError("network down"))
      return base(url, opts)
    })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await waitForLoadedBuilder()

    fireEvent.click(screen.getByText("builds.editor.header.publish"))

    await expectToast("builds.editor.error.unknown")
    expect(screen.getByDisplayValue("Existing Agent")).toBeInTheDocument()
  })
})

// Kept separate from the edit-mode table: this path has to create the agent
// first, so it shares no setup with the three header actions.
describe("AgentBuilder success-dialog publish error handling (issue #969)", () => {
  it.each(ERROR_CASES)(
    "surfaces a displayable message for $name",
    async (errorCase) => {
      installCreateModeApi(errorCase)
      render(<AgentBuilder />)

      const nameInput = await screen.findByPlaceholderText(
        "builds.configForm.name.placeholder"
      )
      fireEvent.change(nameInput, { target: { value: "New Agent" } })

      // Instructions live in a contentEditable div, not a form control.
      const editor = document.querySelector("[contenteditable]") as HTMLElement
      expect(editor).toBeTruthy()
      editor.textContent = "You are a new agent."
      fireEvent.input(editor)

      // Creation is rejected while the general model is unset, and that value
      // only lands once /api/models/user-default has been applied. The model
      // Select mirrors it into data-value, so waiting for the id here ensures
      // the state is committed before the one create click below.
      await waitFor(() =>
        expect(screen.getByTestId("model-select")).toHaveAttribute(
          "data-value",
          String(DEFAULT_MODEL.id)
        )
      )

      fireEvent.click(screen.getByText("builds.editor.header.create"))

      // After creation the header keeps its own publish button, so scope the
      // click to the success dialog.
      const dialog = await screen.findByRole("dialog")
      fireEvent.click(within(dialog).getByText("builds.editor.header.publish"))

      await expectToast(
        errorCase.expected === FALLBACK
          ? "builds.publication.publishFailed"
          : errorCase.expected
      )
      // The background form is aria-hidden behind the dialog overlay, so
      // assert survival via the dialog staying mounted after the failure.
      expect(screen.getByRole("dialog")).toBeInTheDocument()
    }
  )
})
