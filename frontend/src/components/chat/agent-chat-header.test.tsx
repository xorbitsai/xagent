import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import en from "@/i18n/locales/en"
import zh from "@/i18n/locales/zh"
import { AgentChatHeader } from "./agent-chat-header"
import { AgentProfileDialog } from "./agent-profile-dialog"

const mocks = vi.hoisted(() => ({
  locale: "en",
  userId: "alice",
  request: vi.fn(),
  dispatch: vi.fn(),
  success: vi.fn(),
  push: vi.fn(),
}))
const translate = (key: string, params?: Record<string, string>) => {
  const translation = key.split(".").reduce<unknown>((object, part) =>
    object && typeof object === "object" ? (object as Record<string, unknown>)[part] : undefined, mocks.locale === "zh" ? zh : en)
  const value = typeof translation === "string" ? translation : key
  return Object.entries(params ?? {}).reduce((text, [name, replacement]) => text.replace(`{${name}}`, replacement), value)
}
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: translate }) }))
vi.mock("@/contexts/app-context-chat", () => ({ useApp: () => ({ dispatch: mocks.dispatch }) }))
vi.mock("@/contexts/auth-context", () => ({ useAuth: () => ({ user: { id: mocks.userId } }) }))
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: mocks.push }) }))
vi.mock("@/lib/api-wrapper", () => ({ apiRequest: mocks.request }))
vi.mock("@/components/ui/sonner", () => ({ toast: { success: mocks.success } }))
vi.mock("@/components/build/agent-triggers-dialog", () => ({ AgentTriggersDialog: () => null }))
vi.mock("@/components/build/deploy-agent-dialog", () => ({ DeployAgentDialog: () => null }))

let savedProfile: Record<string, unknown>

async function openProfile() {
  fireEvent.click(screen.getByRole("button", { name: /More options for/ }))
  fireEvent.click(screen.getByRole("button", { name: "Profile" }))
  await screen.findByLabelText("Standing rules")
}

describe("Agent profile editing", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.locale = "en"
    mocks.userId = "alice"
    savedProfile = {
      id: 7, name: "Ben", description: "Support specialist",
      instructions: "Answer support requests.", status: "published",
      can_edit: true, readonly: false,
    }
    mocks.request.mockImplementation(async (_url: string, options?: RequestInit) => {
      if (options?.method === "PUT") {
        savedProfile = { ...savedProfile, ...JSON.parse(String(options.body)) }
      }
      return Response.json(savedProfile)
    })
  })
  afterEach(cleanup)

  it("closes the old profile and clears header details when accounts switch", async () => {
    const view = render(<AgentChatHeader agentId={7} name="Ben" description="Private hint" />)
    await openProfile()
    const save = screen.getByRole("button", { name: "Save" })
    fireEvent.change(screen.getByLabelText("Standing rules"), { target: { value: "Alice's private draft" } })
    mocks.userId = "bob"
    mocks.request.mockResolvedValue(new Response(null, { status: 403 }))
    view.rerender(<AgentChatHeader agentId={7} name="Ben" description="Private hint" />)
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(screen.queryByText("Ben")).not.toBeInTheDocument()
    expect(screen.queryByText("Private hint")).not.toBeInTheDocument()
    fireEvent.click(save)
    await act(async () => {})
    expect(mocks.request.mock.calls.some(([, options]) => options?.method === "PUT")).toBe(false)
  })

  it.each(["header", "profile", "save"])("discards a late %s response after an account switch", async (pending) => {
    let resolveOld!: (response: Response) => void
    const oldProfile = { ...savedProfile, name: "Alice private employee", instructions: "Private rules" }
    if (pending === "header") mocks.request.mockImplementationOnce(() => new Promise<Response>((resolve) => { resolveOld = resolve }))
    const view = render(<AgentChatHeader agentId={7} name="Ben" />)
    if (pending === "profile") {
      await screen.findByText("Ben")
      mocks.request.mockImplementationOnce(() => new Promise<Response>((resolve) => { resolveOld = resolve }))
      fireEvent.click(screen.getByRole("button", { name: /More options for/ }))
      fireEvent.click(screen.getByRole("button", { name: "Profile" }))
    }
    if (pending === "save") {
      await openProfile()
      mocks.request.mockImplementationOnce(() => new Promise<Response>((resolve) => { resolveOld = resolve }))
      fireEvent.change(screen.getByLabelText("Name"), { target: { value: oldProfile.name } })
      fireEvent.click(screen.getByRole("button", { name: "Save" }))
    }
    mocks.userId = "bob"
    mocks.request.mockResolvedValue(new Response(null, { status: 403 }))
    view.rerender(<AgentChatHeader agentId={7} name="Ben" />)
    await act(async () => resolveOld(Response.json(oldProfile)))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(screen.queryByText(oldProfile.name)).not.toBeInTheDocument()
    expect(screen.queryByDisplayValue(oldProfile.instructions)).not.toBeInTheDocument()
    expect(mocks.success).not.toHaveBeenCalled()
    expect(mocks.dispatch).not.toHaveBeenCalled()
  })

  it("protects a directly mounted profile even if its parent does not remount", async () => {
    const onClose = vi.fn()
    const props = { agent: { id: 7, name: "Ben", status: "published" }, avatar: null, onClose, onSaved: vi.fn() }
    const view = render(<AgentProfileDialog {...props} />)
    await screen.findByLabelText("Standing rules")
    fireEvent.change(screen.getByLabelText("Standing rules"), { target: { value: "Private draft" } })
    mocks.userId = "bob"
    view.rerender(<AgentProfileDialog {...props} />)
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(onClose).toHaveBeenCalled()
    expect(mocks.request.mock.calls.some(([, options]) => options?.method === "PUT")).toBe(false)
  })

  it("saves editable fields through the API and refreshes the header and sidebar", async () => {
    render(<AgentChatHeader agentId={7} name="Ben" />)
    await openProfile()
    expect(screen.getByLabelText("Status")).toHaveAttribute("readonly")
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Ben Support" } })
    fireEvent.change(screen.getByLabelText("Description"), { target: { value: "" } })
    fireEvent.change(screen.getByLabelText("Standing rules"), { target: { value: "Ask before sending refunds." } })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    const saveCall = mocks.request.mock.calls.find(([, options]) => options?.method === "PUT")
    expect(saveCall?.[0]).toContain("/api/agents/7")
    expect(JSON.parse(saveCall?.[1].body)).toEqual({
      name: "Ben Support", description: "", instructions: "Ask before sending refunds.",
    })
    expect(screen.getByText("Ben Support")).toBeInTheDocument()
    expect(mocks.dispatch).toHaveBeenCalledWith({ type: "TRIGGER_TASK_UPDATE" })
    await openProfile()
    expect(screen.getByLabelText("Name")).toHaveValue("Ben Support")
    expect(screen.getByLabelText("Standing rules")).toHaveValue("Ask before sending refunds.")
  })

  it("discards edits on cancel and does not submit empty names or rules", async () => {
    render(<AgentChatHeader agentId={7} name="Ben" />)
    await openProfile()
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "   " } })
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Unsaved" } })
    fireEvent.change(screen.getByLabelText("Standing rules"), { target: { value: "" } })
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
    await openProfile()
    expect(screen.getByLabelText("Name")).toHaveValue("Ben")
    expect(screen.getByLabelText("Standing rules")).toHaveValue("Answer support requests.")
    expect(mocks.request.mock.calls.some(([, options]) => options?.method === "PUT")).toBe(false)
  })

  it("preserves edits and exposes backend errors when saving fails", async () => {
    render(<AgentChatHeader agentId={7} name="Ben" />)
    await openProfile()
    mocks.request.mockResolvedValueOnce(Response.json({ detail: "Agent with this name already exists" }, { status: 400 }))
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Duplicate" } })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    expect(await screen.findByRole("alert")).toHaveTextContent("Agent with this name already exists")
    expect(screen.getByLabelText("Name")).toHaveValue("Duplicate")
    expect(mocks.dispatch).not.toHaveBeenCalled()
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "New name" } })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(screen.getByText("New name")).toBeInTheDocument()
  })

  it("honors read-only access and never displays a save button", async () => {
    savedProfile = { ...savedProfile, can_edit: false, readonly: true }
    render(<AgentChatHeader agentId={7} name="Ben" />)
    await openProfile()
    for (const field of ["Name", "Description", "Standing rules"]) {
      expect(screen.getByLabelText(field)).toHaveAttribute("readonly")
    }
    expect(screen.queryByRole("button", { name: "Save" })).not.toBeInTheDocument()
    expect(screen.getByText("You have read-only access to this teammate.")).toBeInTheDocument()
  })

  it("prevents duplicate saves and ignores a late response after changing employees", async () => {
    const { rerender } = render(<AgentChatHeader agentId={7} name="Ben" />)
    await openProfile()
    let resolveSave!: (value: Response) => void
    mocks.request.mockImplementationOnce(() => new Promise<Response>((resolve) => { resolveSave = resolve }))
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Saved Ben" } })
    const save = screen.getByRole("button", { name: "Save" })
    fireEvent.click(save)
    fireEvent.click(save)
    expect(mocks.request.mock.calls.filter(([, options]) => options?.method === "PUT")).toHaveLength(1)
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled()
    savedProfile = { ...savedProfile, id: 8, name: "Maya" }
    rerender(<AgentChatHeader agentId={8} name="Maya" />)
    await screen.findByText("Maya")
    await act(async () => { resolveSave(Response.json({ ...savedProfile, id: 7, name: "Saved Ben" })) })
    expect(screen.getByText("Maya")).toBeInTheDocument()
    expect(screen.queryByText("Saved Ben")).not.toBeInTheDocument()
    expect(mocks.dispatch).not.toHaveBeenCalled()
  })

  it.each([
    ["published", "已启用"],
    ["draft", "草稿"],
    ["archived", "已归档"],
  ])("localizes the %s profile status and actions in Chinese", async (status, label) => {
    mocks.locale = "zh"
    savedProfile = { ...savedProfile, status }
    render(<AgentChatHeader agentId={7} name="Ben" showBack />)
    expect(screen.getByRole("button", { name: "返回新建任务" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "例行任务" })).toBeInTheDocument()
    fireEvent.click(await screen.findByRole("button", { name: "Ben 的更多选项" }))
    fireEvent.click(screen.getByRole("button", { name: "员工资料" }))
    expect(await screen.findByRole("button", { name: "构建器设置 指令、模型、知识、技能、工具和触发器" })).toBeInTheDocument()
    expect(await screen.findByLabelText("状态")).toHaveValue(label)
    expect(screen.getByLabelText("状态")).toHaveAttribute("readonly")
    expect(screen.queryByText("Builder settings")).not.toBeInTheDocument()
  })

  it("edits a multiline description and removes the memory entry from Profile", async () => {
    render(<AgentChatHeader agentId={7} name="Ben" />)
    await openProfile()
    expect(screen.queryByLabelText("Title")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /remembers|memory/i })).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Builder settings/ })).toBeInTheDocument()
    const field = screen.getByLabelText("Description")
    expect(field.tagName).toBe("TEXTAREA")
    const description = "Triages support requests.\nDrafts replies for review."
    fireEvent.change(field, { target: { value: description } })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    const saveCall = mocks.request.mock.calls.find(([, options]) => options?.method === "PUT")
    expect(JSON.parse(saveCall?.[1].body)).toEqual({
      name: "Ben", description,
    })
    await openProfile()
    expect(screen.getByLabelText("Description")).toHaveValue(description)
  })

  it.each([null, "", "   "])("preserves unchanged rules (%j) when saving metadata", async (instructions) => {
    savedProfile = { ...savedProfile, instructions }
    render(<AgentChatHeader agentId={7} name="Ben" />)
    await openProfile()
    const rules = screen.getByLabelText("Standing rules")
    expect(rules).not.toBeRequired()
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Ben Support" } })
    fireEvent.change(screen.getByLabelText("Description"), { target: { value: "Updated description" } })
    expect(screen.getByRole("button", { name: "Save" })).toBeEnabled()
    expect(rules.closest("form")!.checkValidity()).toBe(true)
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    const saveCall = mocks.request.mock.calls.find(([, options]) => options?.method === "PUT")
    expect(JSON.parse(saveCall?.[1].body)).toEqual({ name: "Ben Support", description: "Updated description" })
    expect(savedProfile.instructions).toBe(instructions)
    await openProfile()
    expect(screen.getByLabelText("Standing rules")).toHaveValue(instructions ?? "")
  })
})
