import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const authMock = vi.hoisted(() => ({ user: { is_admin: true } }))

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))
vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})
vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => authMock,
}))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))
vi.mock("@/components/ui/sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}))
vi.mock("lucide-react", () => {
  const Icon = (props: React.SVGProps<SVGSVGElement>) => <svg {...props} />
  return { Plus: Icon, Trash2: Icon, Edit2: Icon, Search: Icon }
})
vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) => open ? <div>{children}</div> : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => <section>{children}</section>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <header>{children}</header>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <p>{children}</p>,
  DialogFooter: ({ children }: { children: React.ReactNode }) => <footer>{children}</footer>,
}))
vi.mock("@/components/ui/stepper", () => ({
  Stepper: ({ steps, currentStep }: { steps: Array<{ content: React.ReactNode }>; currentStep: number }) => (
    <div>{steps[currentStep - 1]?.content}</div>
  ),
}))
vi.mock("@/components/ui/select-radix", () => ({
  Select: ({ value, onValueChange, disabled, children }: {
    value?: string
    onValueChange?: (value: string) => void
    disabled?: boolean
    children: React.ReactNode
  }) => (
    <div data-disabled={disabled ? "true" : "false"} data-value={value}>
      <button
        type="button"
        disabled={disabled}
        onClick={() => onValueChange?.(value === "oauth" ? "stdio" : "Other")}
      >select:{value}</button>
      {children}
    </div>
  ),
  SelectTrigger: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
  SelectValue: () => null,
  SelectContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SelectItem: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
}))
vi.mock("@/components/ui/switch", () => ({
  Switch: ({ checked, disabled, onCheckedChange, "aria-label": ariaLabel }: {
    checked?: boolean
    disabled?: boolean
    onCheckedChange?: (checked: boolean) => void
    "aria-label"?: string
  }) => (
    <button
      type="button"
      role="switch"
      aria-label={ariaLabel}
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onCheckedChange?.(!checked)}
    />
  ),
}))
vi.mock("@/components/ui/table", () => ({
  Table: ({ children }: { children: React.ReactNode }) => <table>{children}</table>,
  TableHeader: ({ children }: { children: React.ReactNode }) => <thead>{children}</thead>,
  TableBody: ({ children }: { children: React.ReactNode }) => <tbody>{children}</tbody>,
  TableRow: ({ children }: { children: React.ReactNode }) => <tr>{children}</tr>,
  TableHead: ({ children }: { children: React.ReactNode }) => <th>{children}</th>,
  TableCell: ({ children }: { children: React.ReactNode }) => <td>{children}</td>,
}))

import AdminMcpPage from "./admin-mcp"
import en from "../../i18n/locales/en"
import zh from "../../i18n/locales/zh"

type App = {
  id: number
  app_id: string
  name: string
  description: string | null
  icon: string | null
  transport: string
  provider_name: string | null
  category: string | null
  oauth_scopes: string[] | null
  is_visible_in_connector: boolean
  launch_config: Record<string, unknown> | null
  is_builtin: boolean
}

const builtinApp: App = {
  id: 1,
  app_id: "gmail",
  name: "Gmail",
  description: "Original description",
  icon: "gmail.svg",
  transport: "oauth",
  provider_name: "google",
  category: "Communication",
  oauth_scopes: ["scope:mail"],
  is_visible_in_connector: true,
  launch_config: {
    command: "python",
    args: ["-m", "xagent.web.tools.mcp.gmail"],
  },
  is_builtin: true,
}

const customApp: App = {
  ...builtinApp,
  id: 2,
  app_id: "custom-mail",
  name: "Custom Mail",
  provider_name: null,
  transport: "stdio",
  is_builtin: false,
}

function response(body: unknown) {
  return { ok: true, json: vi.fn().mockResolvedValue(body) }
}

function renderPage(apps: App[]) {
  apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
    if (!options?.method && url.endsWith("/providers")) return Promise.resolve(response([]))
    if (!options?.method && url.endsWith("/apps")) return Promise.resolve(response(apps))
    return Promise.reject(new Error(`Unexpected request: ${options?.method ?? "GET"} ${url}`))
  })
  return render(<AdminMcpPage />)
}

async function openEditor(appId: string) {
  const cell = await screen.findByText(appId)
  const row = cell.closest("tr")
  if (!row) throw new Error(`Missing row for ${appId}`)
  const buttons = within(row).getAllByRole("button")
  fireEvent.click(buttons[0])
}

describe("AdminMcpPage app updates", () => {
  beforeEach(() => {
    vi.stubGlobal("React", React)
    apiRequestMock.mockReset()
  })
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it("warns administrators not to store credentials in launch config", () => {
    expect(en.adminMcp.apps.form.launchConfigGuidance).toContain("Do not include credential or secret values")
    expect(en.adminMcp.apps.form.launchConfigGuidance).toContain("connector credential flow")
    expect(zh.adminMcp.apps.form.launchConfigGuidance).toContain("不要填写凭证或密钥值")
    expect(zh.adminMcp.apps.form.launchConfigGuidance).toContain("连接器凭证流程")
  })

  it("offers Productivity as an app category", async () => {
    renderPage([builtinApp])
    await openEditor("gmail")

    expect(screen.getByText("Productivity")).not.toBeNull()
  })

  it("patches only visibility and applies the authoritative response", async () => {
    const authoritative = {
      ...customApp,
      provider_name: "authoritative-provider",
      is_visible_in_connector: false,
    }
    renderPage([customApp])
    await screen.findByText("custom-mail")

    apiRequestMock.mockResolvedValueOnce(response(authoritative))
    fireEvent.click(screen.getByRole("switch", { name: "adminMcp.apps.form.visibleInConnector" }))

    await waitFor(() => expect(apiRequestMock).toHaveBeenLastCalledWith(
      "http://api.local/api/admin/mcp/apps/2",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ is_visible_in_connector: false }),
      }),
    ))
    expect(await screen.findByText("authoritative-provider")).not.toBeNull()
  })

  it("locks managed built-in fields while keeping custom fields editable", async () => {
    renderPage([builtinApp, customApp])
    await openEditor("gmail")

    expect((screen.getByDisplayValue("gmail") as HTMLInputElement).disabled).toBe(true)
    expect((screen.getByDisplayValue("Gmail") as HTMLInputElement).disabled).toBe(true)
    expect((screen.getByText("select:oauth").closest("button") as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByDisplayValue("scope:mail") as HTMLInputElement).disabled).toBe(true)
    expect((screen.getByDisplayValue(/xagent\.web\.tools\.mcp\.gmail/) as HTMLTextAreaElement).disabled).toBe(true)
    expect(screen.getByText("adminMcp.apps.form.managedFieldsDescription")).not.toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "adminMcp.modal.back" }))
    expect((screen.getByText("select:google").closest("button") as HTMLButtonElement).disabled).toBe(true)

    cleanup()
    renderPage([customApp])
    await openEditor("custom-mail")

    expect((screen.getByDisplayValue("custom-mail") as HTMLInputElement).disabled).toBe(true)
    expect((screen.getByDisplayValue("Custom Mail") as HTMLInputElement).disabled).toBe(false)
    expect((screen.getByText("select:stdio").closest("button") as HTMLButtonElement).disabled).toBe(false)
    expect((screen.getByDisplayValue(/xagent\.web\.tools\.mcp\.gmail/) as HTMLTextAreaElement).disabled).toBe(false)
    fireEvent.click(screen.getByRole("button", { name: "adminMcp.modal.back" }))
    expect((screen.getByText("select:none").closest("button") as HTMLButtonElement).disabled).toBe(false)
  })

  it("offers delete only for custom catalog apps", async () => {
    renderPage([builtinApp, customApp])

    const builtinRow = (await screen.findByText("gmail")).closest("tr")
    const customRow = (await screen.findByText("custom-mail")).closest("tr")
    if (!builtinRow || !customRow) throw new Error("Missing app rows")

    expect(within(builtinRow).queryByRole("button", { name: "adminMcp.apps.deleteAction" })).toBeNull()
    expect(within(customRow).getByRole("button", { name: "adminMcp.apps.deleteAction" })).not.toBeNull()
  })

  it("sends only presentation deltas when editing a built-in app", async () => {
    const authoritative = { ...builtinApp, description: "Updated description" }
    renderPage([builtinApp])
    await openEditor("gmail")

    fireEvent.change(screen.getByDisplayValue("Original description"), {
      target: { value: "Updated description" },
    })
    apiRequestMock.mockResolvedValueOnce(response(authoritative))
    fireEvent.click(screen.getByRole("button", { name: "adminMcp.modal.saveApp" }))

    await waitFor(() => expect(apiRequestMock).toHaveBeenLastCalledWith(
      "http://api.local/api/admin/mcp/apps/1",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ description: "Updated description" }),
      }),
    ))
  })
})
