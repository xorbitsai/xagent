import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// Pasting into the instructions editor must clip only the pasted text to the
// room left after removing the replaced selection, so the user's existing text
// is never removed, and must tell the user, once per action, how many
// characters were dropped.

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastWarningMock = vi.hoisted(() => vi.fn())

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

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: "en",
    t: (key: string, vars?: Record<string, string>) =>
      vars ? `${key}:${JSON.stringify(vars)}` : key,
  }),
}))

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({ apps: [], getAppIcon: () => null }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("sonner", () => ({
  toast: { error: vi.fn(), success: vi.fn(), warning: toastWarningMock },
}))

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
  MultiSelect: () => null,
}))
vi.mock("@/components/ui/select", () => ({ Select: () => null }))
vi.mock("@/components/build/build-file-preview-sheet", () => ({
  BuildFilePreviewSheet: () => null,
}))

import { AgentBuilder } from "./agent-builder"

const AGENT_ID = "7"
const MAX = 10000

function agentResponse(instructions: string) {
  return {
    id: Number(AGENT_ID),
    user_id: 1,
    team_id: null,
    name: "Paste Test Agent",
    description: "",
    instructions,
    execution_mode: "balanced",
    models: { general: "10" },
    knowledge_bases: [],
    skills: [],
    tool_categories: [],
    suggested_prompts: [],
    logo_url: null,
    status: "draft",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    widget_enabled: false,
    allowed_domains: [],
    share_enabled: false,
    share_updated_at: null,
    can_edit: true,
  }
}

function installApi(instructions: string) {
  apiRequestMock.mockImplementation((url: string) => {
    if (url.endsWith("/api/kb/collections"))
      return Promise.resolve(new Response(JSON.stringify({ collections: [] }), { status: 200 }))
    if (url.endsWith("/api/skills/"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith("/api/tools/available?include_usage=false"))
      return Promise.resolve(new Response(JSON.stringify({ tools: [] }), { status: 200 }))
    if (url.endsWith("/api/models/?category=llm"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith("/api/models/user-default"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.includes(`/api/agents/${AGENT_ID}/triggers`))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith(`/api/agents/${AGENT_ID}`))
      return Promise.resolve(new Response(JSON.stringify(agentResponse(instructions)), { status: 200 }))
    if (url.includes("/api/mcp/servers"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }))
  })
}

async function renderWithInstructions(existing: string) {
  installApi(existing)
  const { container } = render(<AgentBuilder agentId={AGENT_ID} />)
  const editor = container.querySelector('[role="textbox"][aria-multiline="true"]') as HTMLDivElement
  await waitFor(() => expect(editor.textContent).toBe(existing))
  return editor
}

// Focus first, as a user pasting into the editor already has it focused.
// (jsdom collapses the selection when an unfocused editable is focused.)
function selectAll(editor: HTMLElement) {
  editor.focus()
  const range = document.createRange()
  range.selectNodeContents(editor)
  const selection = window.getSelection()!
  selection.removeAllRanges()
  selection.addRange(range)
}

// Collapsed caret at `offset` inside the editor's single text node. Focus
// first for the same reason as selectAll.
function placeCaret(editor: HTMLElement, offset: number) {
  editor.focus()
  const range = document.createRange()
  range.setStart(editor.firstChild!, offset)
  range.collapse(true)
  const selection = window.getSelection()!
  selection.removeAllRanges()
  selection.addRange(range)
}

function expectSingleTruncationToast(dropped: number) {
  expect(toastWarningMock).toHaveBeenCalledTimes(1)
  expect(toastWarningMock.mock.calls[0][0]).toBe(
    `builds.configForm.instructions.truncatedToast:${JSON.stringify({
      max: MAX.toLocaleString(),
      dropped: dropped.toLocaleString(),
    })}`
  )
}

function paste(editor: HTMLElement, text: string) {
  fireEvent.paste(editor, { clipboardData: { getData: () => text } })
}

const banner = () =>
  screen.queryByText(
    `builds.configForm.instructions.maxLengthExceeded:${JSON.stringify({
      max: MAX.toLocaleString(),
    })}`
  )

beforeEach(() => {
  apiRequestMock.mockReset()
  toastWarningMock.mockReset()
  vi.stubGlobal("WebSocket", vi.fn())
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe("AgentBuilder instructions paste", () => {
  it("keeps the whole pasted text when it replaces a selection and fits the limit", async () => {
    const editor = await renderWithInstructions("a".repeat(5112))
    selectAll(editor)

    paste(editor, "b".repeat(6164))

    expect(editor.textContent).toBe("b".repeat(6164))
    expect(banner()).toBeNull()
    expect(toastWarningMock).not.toHaveBeenCalled()
  })

  it("truncates an over-limit paste to the limit and reports how much was dropped", async () => {
    const editor = await renderWithInstructions("a".repeat(5112))
    selectAll(editor)

    paste(editor, "b".repeat(12000))

    expect(editor.textContent).toBe("b".repeat(MAX))
    expect(banner()).toBeInTheDocument()
    expectSingleTruncationToast(2000)
  })

  it("replaces a full editor when pasting over a selection", async () => {
    const editor = await renderWithInstructions("a".repeat(MAX))
    selectAll(editor)

    paste(editor, "short prompt")

    expect(editor.textContent).toBe("short prompt")
    expect(banner()).toBeNull()
    expect(toastWarningMock).not.toHaveBeenCalled()
  })

  it("clips a paste at a mid-document caret and keeps the existing trailing text", async () => {
    const editor = await renderWithInstructions("x".repeat(1000) + "y".repeat(8000))
    placeCaret(editor, 1000)

    paste(editor, "b".repeat(3000))

    expect(editor.textContent).toBe("x".repeat(1000) + "b".repeat(1000) + "y".repeat(8000))
    expect(banner()).toBeInTheDocument()
    expectSingleTruncationToast(2000)
  })

  it("clips a paste at the end caret to the room left", async () => {
    const editor = await renderWithInstructions("x".repeat(9000))
    placeCaret(editor, 9000)

    paste(editor, "b".repeat(3000))

    expect(editor.textContent).toBe("x".repeat(9000) + "b".repeat(1000))
    expectSingleTruncationToast(2000)
  })

  it("clips a paste when the editor has no selection range", async () => {
    const editor = await renderWithInstructions("x".repeat(9000))
    editor.focus()
    window.getSelection()!.removeAllRanges()
    expect(window.getSelection()!.rangeCount).toBe(0)

    paste(editor, "b".repeat(3000))

    expect(editor.textContent).toBe("x".repeat(9000) + "b".repeat(1000))
    expectSingleTruncationToast(2000)
  })

  it("inserts nothing when pasting into a full editor at a mid-document caret", async () => {
    const editor = await renderWithInstructions("a".repeat(MAX))
    placeCaret(editor, 5000)

    paste(editor, "bbbbb")

    expect(editor.textContent).toBe("a".repeat(MAX))
    expectSingleTruncationToast(5)
  })

  it("shows one toast with the total when a paste lands in content that was already over the limit", async () => {
    const editor = await renderWithInstructions("a".repeat(MAX + 5))
    placeCaret(editor, 5000)

    paste(editor, "bbb")

    expect(editor.textContent).toBe("a".repeat(MAX))
    expectSingleTruncationToast(8)
  })

  it("truncates typing past the limit and toasts once", async () => {
    const editor = await renderWithInstructions("a".repeat(MAX))
    editor.textContent = "a".repeat(MAX) + "z"

    fireEvent.input(editor)

    expect(editor.textContent).toBe("a".repeat(MAX))
    expectSingleTruncationToast(1)
  })
})
