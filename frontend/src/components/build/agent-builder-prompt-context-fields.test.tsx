import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const toastErrorMock = vi.hoisted(() => vi.fn())
// Every render's props, oldest first: the real chat socket keeps calling the first one it saw.
const chatProps = vi.hoisted(
  () => [] as { onUpdateConfig: (updates: { id?: number; created?: boolean }) => void }[],
)

const apiRequestMock = vi.hoisted(() => vi.fn())

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
      vars?.appName ? `${key}:${vars.appName}` : key,
  }),
}))

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
  ResizableThreeColumnLayout: ({
    leftPanel,
    middlePanel,
  }: {
    leftPanel: React.ReactNode
    middlePanel: React.ReactNode
  }) => (
    <div>
      {leftPanel}
      {middlePanel}
    </div>
  ),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => null,
}))

vi.mock("@/components/build/agent-builder-chat", () => ({
  AgentBuilderChat: (props: (typeof chatProps)[number]) => {
    chatProps.push(props)
    return null
  },
}))
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
const FIELDS = "builds.configForm.promptContextFields"

type Declaration = { name: string; max_length: number; description?: string | null }

const DEFAULT_MODEL = {
  id: 10,
  model_id: "test/model",
  model_name: "Test Model",
  model_provider: "test",
  category: "llm",
}

function agentResponse(fields: Declaration[] | undefined, canEdit = true, id = Number(AGENT_ID)) {
  return {
    id,
    user_id: 1,
    team_id: null,
    name: "Fields Agent",
    description: "",
    instructions: "Use customer.tier in the reference data.",
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
    can_edit: canEdit,
    ...(fields === undefined ? {} : { prompt_context_fields: fields }),
  }
}

function installApi(
  fields: Declaration[] | undefined,
  putResponse?: () => Response,
  canEdit = true,
  { otherAgents = {} }: { otherAgents?: Record<number, Declaration[]> } = {},
) {
  apiRequestMock.mockImplementation((url: string, opts?: { method?: string; body?: string }) => {
    if (opts?.method === "POST" && url.endsWith("/api/agents")) {
      const body = JSON.parse(opts.body || "{}")
      return Promise.resolve(
        new Response(JSON.stringify(agentResponse(body.prompt_context_fields)), { status: 200 })
      )
    }
    if (opts?.method === "PUT") {
      if (putResponse) return Promise.resolve(putResponse())
      const body = JSON.parse(opts.body || "{}")
      return Promise.resolve(
        new Response(JSON.stringify(agentResponse(body.prompt_context_fields)), { status: 200 })
      )
    }
    if (url.endsWith("/api/kb/collections"))
      return Promise.resolve(new Response(JSON.stringify({ collections: [] }), { status: 200 }))
    if (url.endsWith("/api/skills/"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith("/api/tools/available?include_usage=false"))
      return Promise.resolve(new Response(JSON.stringify({ tools: [] }), { status: 200 }))
    if (url.endsWith("/api/models/?category=llm"))
      return Promise.resolve(new Response(JSON.stringify([DEFAULT_MODEL]), { status: 200 }))
    if (url.endsWith("/api/models/user-default"))
      return Promise.resolve(
        new Response(
          JSON.stringify([{ config_type: "general", model: { id: DEFAULT_MODEL.id } }]),
          { status: 200 },
        )
      )
    if (/\/api\/agents\/\d+\/triggers/.test(url))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    const agentMatch = url.match(/\/api\/agents\/(\d+)$/)
    if (agentMatch) {
      const id = Number(agentMatch[1])
      return Promise.resolve(
        new Response(JSON.stringify(agentResponse(otherAgents[id] ?? fields, canEdit, id)), { status: 200 })
      )
    }
    if (url.includes("/api/mcp/servers"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }))
  })
}

const updateButton = () => screen.getByText("builds.editor.header.update")
const nameInputs = () => screen.queryAllByLabelText(`${FIELDS}.nameLabel`) as HTMLInputElement[]
const lengthInputs = () => screen.queryAllByLabelText(`${FIELDS}.maxLengthLabel`) as HTMLInputElement[]
const descriptionInputs = () =>
  screen.queryAllByLabelText(`${FIELDS}.descriptionLabel`) as HTMLInputElement[]
const addButton = () => screen.getByRole("button", { name: `${FIELDS}.add` })
const deleteButtons = () => screen.queryAllByRole("button", { name: `${FIELDS}.delete` })
const callsWith = (method: string) =>
  apiRequestMock.mock.calls.filter(([, opts]) => (opts as { method?: string } | undefined)?.method === method)
const putCalls = () => callsWith("PUT")
const postCalls = () => callsWith("POST").filter(([url]) => String(url).endsWith("/api/agents"))
const lastPutBody = () => JSON.parse((putCalls().at(-1)![1] as { body: string }).body)

async function renderNewAgent() {
  render(<AgentBuilder />)
  const nameInput = await screen.findByPlaceholderText("builds.configForm.name.placeholder")
  fireEvent.change(nameInput, { target: { value: "New Agent" } })
  const editor = document.querySelector("[contenteditable]") as HTMLElement
  editor.textContent = "Use customer.tier in the reference data."
  fireEvent.input(editor)
  await waitFor(() =>
    expect(screen.getByTestId("model-select")).toHaveAttribute("data-value", String(DEFAULT_MODEL.id))
  )
}

async function renderLoaded() {
  render(<AgentBuilder agentId={AGENT_ID} />)
  await waitFor(() => expect(screen.getByDisplayValue("Fields Agent")).toBeInTheDocument())
}

beforeEach(() => {
  apiRequestMock.mockReset()
  toastErrorMock.mockReset()
  chatProps.length = 0
  ;(globalThis as unknown as { WebSocket: unknown }).WebSocket = vi.fn()
})

afterEach(() => cleanup())

describe("AgentBuilder prompt context fields", () => {
  it("shows no rows and stays clean for an agent without declarations", async () => {
    installApi(undefined)
    await renderLoaded()

    expect(nameInputs()).toHaveLength(0)
    expect(updateButton()).toBeDisabled()
  })

  it("loads saved declarations without marking the agent dirty", async () => {
    installApi([{ name: "customer.tier", max_length: 32 }])
    await renderLoaded()

    expect(nameInputs().map((input) => input.value)).toEqual(["customer.tier"])
    expect(lengthInputs().map((input) => input.value)).toEqual(["32"])
    expect(updateButton()).toBeDisabled()
  })

  it("stays clean when an edit spells the stored value differently", async () => {
    installApi([{ name: "customer.tier", max_length: 7, description: "Membership level" }])
    await renderLoaded()

    fireEvent.change(lengthInputs()[0], { target: { value: "007" } })
    fireEvent.change(descriptionInputs()[0], { target: { value: " Membership level " } })
    expect(updateButton()).toBeDisabled()

    fireEvent.change(lengthInputs()[0], { target: { value: "7.0" } })
    expect(updateButton()).not.toBeDisabled()
  })

  it("adds a field and saves it", async () => {
    installApi([])
    await renderLoaded()

    fireEvent.click(addButton())
    expect(lengthInputs()[0].value).toBe("256")
    fireEvent.change(nameInputs()[0], { target: { value: " order_id " } })
    fireEvent.change(lengthInputs()[0], { target: { value: "64" } })
    expect(updateButton()).not.toBeDisabled()

    fireEvent.click(updateButton())

    await waitFor(() => expect(putCalls()).toHaveLength(1))
    expect(lastPutBody().prompt_context_fields).toEqual([{ name: "order_id", max_length: 64, description: null }])
    await waitFor(() => expect(updateButton()).toBeDisabled())
    expect(nameInputs()[0].value).toBe("order_id")
  })

  it("edits and removes declarations, saving the full remaining list", async () => {
    installApi([
      { name: "customer.tier", max_length: 32 },
      { name: "order_id", max_length: 64 },
    ])
    await renderLoaded()

    fireEvent.change(lengthInputs()[0], { target: { value: "48" } })
    fireEvent.click(deleteButtons()[1])
    expect(nameInputs()).toHaveLength(1)

    fireEvent.click(updateButton())

    await waitFor(() => expect(putCalls()).toHaveLength(1))
    expect(lastPutBody().prompt_context_fields).toEqual([{ name: "customer.tier", max_length: 48, description: null }])
  })

  it("saves an empty list when every declaration is removed", async () => {
    installApi([{ name: "customer.tier", max_length: 32 }])
    await renderLoaded()

    fireEvent.click(deleteButtons()[0])
    fireEvent.click(updateButton())

    await waitFor(() => expect(putCalls()).toHaveLength(1))
    expect(lastPutBody().prompt_context_fields).toEqual([])
  })

  it("loads a description and saves an edited one", async () => {
    installApi([{ name: "customer.tier", max_length: 32, description: "Membership level" }])
    await renderLoaded()

    expect(descriptionInputs().map((input) => input.value)).toEqual(["Membership level"])
    expect(updateButton()).toBeDisabled()

    fireEvent.change(descriptionInputs()[0], { target: { value: " Membership level, VIP or standard " } })
    fireEvent.click(updateButton())

    await waitFor(() => expect(putCalls()).toHaveLength(1))
    expect(lastPutBody().prompt_context_fields).toEqual([
      { name: "customer.tier", max_length: 32, description: "Membership level, VIP or standard" },
    ])
  })

  it("shows a description error inline and does not save", async () => {
    installApi([{ name: "customer.tier", max_length: 32 }])
    await renderLoaded()

    fireEvent.change(descriptionInputs()[0], { target: { value: "use `tier`" } })
    expect(screen.getByText(`${FIELDS}.errors.description.invalidChars`)).toBeInTheDocument()
    expect(descriptionInputs()[0]).toHaveClass("border-destructive")

    fireEvent.change(descriptionInputs()[0], { target: { value: "a".repeat(201) } })
    expect(screen.getByText(`${FIELDS}.errors.description.tooLong`)).toBeInTheDocument()

    fireEvent.click(updateButton())

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith(`${FIELDS}.invalid`, expect.anything()))
    expect(putCalls()).toHaveLength(0)
  })

  it("shows format, duplicate and range errors inline and does not save", async () => {
    installApi([{ name: "customer.tier", max_length: 32 }])
    await renderLoaded()

    fireEvent.change(nameInputs()[0], { target: { value: "1tier" } })
    expect(screen.getByText(`${FIELDS}.errors.name.invalid`)).toBeInTheDocument()
    expect(nameInputs()[0]).toHaveAttribute("aria-invalid", "true")
    // The shared Input's aria-invalid: variant is not generated by this Tailwind config.
    expect(nameInputs()[0]).toHaveClass("border-destructive")

    fireEvent.change(nameInputs()[0], { target: { value: "tier" } })
    fireEvent.click(addButton())
    fireEvent.change(nameInputs()[1], { target: { value: "tier" } })
    expect(screen.getAllByText(`${FIELDS}.errors.name.duplicate`)).toHaveLength(2)

    fireEvent.change(nameInputs()[1], { target: { value: "other" } })
    fireEvent.change(lengthInputs()[1], { target: { value: "1025" } })
    expect(screen.getByText(`${FIELDS}.errors.maxLength`)).toBeInTheDocument()
    expect(lengthInputs()[1]).toHaveClass("border-destructive")
    expect(lengthInputs()[0]).not.toHaveClass("border-destructive")

    fireEvent.click(updateButton())

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith(`${FIELDS}.invalid`, expect.anything()))
    expect(putCalls()).toHaveLength(0)
  })

  it("reveals the required-name error only after a save attempt", async () => {
    installApi([])
    await renderLoaded()

    fireEvent.click(addButton())
    expect(screen.queryByText(`${FIELDS}.errors.name.required`)).toBeNull()

    fireEvent.click(updateButton())

    await waitFor(() =>
      expect(screen.getByText(`${FIELDS}.errors.name.required`)).toBeInTheDocument()
    )
    expect(putCalls()).toHaveLength(0)
  })

  it("disables Add at the 32-field limit", async () => {
    installApi(Array.from({ length: 31 }, (_, i) => ({ name: `f${i}`, max_length: 10 })))
    await renderLoaded()

    expect(addButton()).not.toBeDisabled()
    fireEvent.click(addButton())
    expect(nameInputs()).toHaveLength(32)
    expect(addButton()).toBeDisabled()
  })

  it("surfaces the server 422 message and keeps the edit", async () => {
    installApi([], () =>
      new Response(
        JSON.stringify({
          detail: [{ loc: ["body", "prompt_context_fields"], msg: "Value error, prompt_context_fields names must be unique" }],
        }),
        { status: 422 },
      )
    )
    await renderLoaded()

    fireEvent.click(addButton())
    fireEvent.change(nameInputs()[0], { target: { value: "tier" } })
    fireEvent.click(updateButton())

    await waitFor(() =>
      expect(toastErrorMock).toHaveBeenCalledWith(
        "Value error, prompt_context_fields names must be unique",
        expect.anything(),
      )
    )
    expect(nameInputs()[0].value).toBe("tier")
    expect(updateButton()).not.toBeDisabled()
  })

  it("sends declarations when creating an agent", async () => {
    installApi([])
    await renderNewAgent()

    fireEvent.click(addButton())
    fireEvent.change(nameInputs()[0], { target: { value: "order_id" } })
    fireEvent.change(lengthInputs()[0], { target: { value: "64" } })
    fireEvent.click(screen.getByText("builds.editor.header.create"))

    await waitFor(() => expect(postCalls()).toHaveLength(1))
    const body = JSON.parse((postCalls()[0][1] as { body: string }).body)
    expect(body.prompt_context_fields).toEqual([{ name: "order_id", max_length: 64, description: null }])
  })

  it("keeps unsaved declarations when the assistant creates the agent", async () => {
    installApi([])
    await renderNewAgent()
    const staleOnUpdateConfig = chatProps[0].onUpdateConfig

    fireEvent.click(addButton())
    fireEvent.change(nameInputs()[0], { target: { value: "order_id" } })
    act(() => staleOnUpdateConfig({ id: Number(AGENT_ID), created: true }))

    await waitFor(() => expect(screen.getByDisplayValue("Fields Agent")).toBeInTheDocument())
    expect(nameInputs().map((input) => input.value)).toEqual(["order_id"])
    expect(updateButton()).not.toBeDisabled()

    fireEvent.click(updateButton())

    await waitFor(() => expect(putCalls()).toHaveLength(1))
    expect(lastPutBody().prompt_context_fields).toEqual([{ name: "order_id", max_length: 256, description: null }])
  })

  it("keeps rows edited while the assistant-created agent loads", async () => {
    installApi([])
    await renderNewAgent()
    const staleOnUpdateConfig = chatProps[0].onUpdateConfig

    fireEvent.click(addButton())
    fireEvent.change(nameInputs()[0], { target: { value: "order_id" } })
    // Same synchronous stretch: the agent GET has started but not resolved.
    act(() => staleOnUpdateConfig({ id: Number(AGENT_ID), created: true }))
    expect(apiRequestMock.mock.calls.some(([url]) => String(url).endsWith(`/api/agents/${AGENT_ID}`))).toBe(true)
    fireEvent.change(nameInputs()[0], { target: { value: "order_ref" } })

    await waitFor(() => expect(screen.getByDisplayValue("Fields Agent")).toBeInTheDocument())
    expect(nameInputs().map((input) => input.value)).toEqual(["order_ref"])
  })

  it("shows a second assistant-created agent's stored rows, not the first agent's", async () => {
    installApi([], undefined, true, { otherAgents: { 6: [{ name: "order_id", max_length: 10 }] } })
    await renderNewAgent()
    const staleOnUpdateConfig = chatProps[0].onUpdateConfig

    fireEvent.click(addButton())
    fireEvent.change(nameInputs()[0], { target: { value: "customer.tier" } })
    act(() => staleOnUpdateConfig({ id: Number(AGENT_ID), created: true }))
    await waitFor(() => expect(screen.getByDisplayValue("Fields Agent")).toBeInTheDocument())
    expect(nameInputs().map((input) => input.value)).toEqual(["customer.tier"])

    act(() => staleOnUpdateConfig({ id: 6, created: true }))

    await waitFor(() => expect(nameInputs().map((input) => input.value)).toEqual(["order_id"]))
    expect(updateButton()).toBeDisabled()
  })

  it("treats an assistant update from the new-agent page as a switch, not a create", async () => {
    installApi([], undefined, true, { otherAgents: { 6: [{ name: "order_id", max_length: 10 }] } })
    await renderNewAgent()
    const staleOnUpdateConfig = chatProps[0].onUpdateConfig

    fireEvent.click(addButton())
    fireEvent.change(nameInputs()[0], { target: { value: "typed_field" } })
    act(() => staleOnUpdateConfig({ id: 6 }))

    await waitFor(() => expect(nameInputs().map((input) => input.value)).toEqual(["order_id"]))
    expect(updateButton()).toBeDisabled()
  })

  it("does not keep rows for a created agent whose first load was superseded", async () => {
    installApi([], undefined, true, { otherAgents: { 6: [{ name: "order_id", max_length: 10 }] } })
    await renderNewAgent()
    const staleOnUpdateConfig = chatProps[0].onUpdateConfig

    fireEvent.click(addButton())
    fireEvent.change(nameInputs()[0], { target: { value: "typed_field" } })
    act(() => {
      staleOnUpdateConfig({ id: Number(AGENT_ID), created: true })
      staleOnUpdateConfig({ id: 6 })
    })
    await waitFor(() => expect(nameInputs().map((input) => input.value)).toEqual(["order_id"]))

    act(() => staleOnUpdateConfig({ id: Number(AGENT_ID) }))

    await waitFor(() => expect(nameInputs()).toHaveLength(0))
    expect(updateButton()).toBeDisabled()
  })

  it("keeps rows when a same-agent update follows the create before its load settles", async () => {
    installApi([])
    await renderNewAgent()
    const staleOnUpdateConfig = chatProps[0].onUpdateConfig

    fireEvent.click(addButton())
    fireEvent.change(nameInputs()[0], { target: { value: "order_id" } })
    act(() => {
      staleOnUpdateConfig({ id: Number(AGENT_ID), created: true })
      staleOnUpdateConfig({ id: Number(AGENT_ID) })
    })

    await waitFor(() => expect(screen.getByDisplayValue("Fields Agent")).toBeInTheDocument())
    expect(nameInputs().map((input) => input.value)).toEqual(["order_id"])
    expect(updateButton()).not.toBeDisabled()
  })

  it("shows declarations read-only to a viewer who cannot edit", async () => {
    installApi([{ name: "customer.tier", max_length: 32 }], undefined, false)
    await renderLoaded()

    expect(nameInputs()[0]).toBeDisabled()
    expect(lengthInputs()[0]).toBeDisabled()
    expect(deleteButtons()[0]).toBeDisabled()
    expect(addButton()).toBeDisabled()
  })
})
