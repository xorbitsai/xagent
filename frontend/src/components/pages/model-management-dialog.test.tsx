import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { Model, ProviderConfig } from "./models"

const authState = vi.hoisted(() => ({
  user: { id: "7", is_admin: false },
}))
const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: authState.user, token: "token" }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
    tDynamic: (key: string) => key,
  }),
}))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) => open ? <div>{children}</div> : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <p>{children}</p>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
}))

vi.mock("@/components/ui/multi-select", () => ({
  MultiSelect: () => <div data-testid="multi-select" />,
}))

vi.mock("@/components/ui/switch", () => ({
  Switch: ({ checked, onCheckedChange }: { checked: boolean; onCheckedChange: (checked: boolean) => void }) => (
    <input type="checkbox" checked={checked} onChange={(event) => onCheckedChange(event.target.checked)} />
  ),
}))

vi.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("@/components/ui/confirm-dialog", () => ({
  ConfirmDialog: () => null,
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))

import { ModelManagementDialog } from "./model-management-dialog"

const providers: ProviderConfig[] = [
  {
    id: "bedrock",
    name: "Amazon Bedrock",
    description: "Bedrock",
    icon: <span>aws</span>,
    category: ["llm"],
  },
  {
    id: "openai",
    name: "OpenAI",
    description: "OpenAI",
    icon: <span>openai</span>,
    category: ["llm"],
    defaultBaseUrl: "https://api.openai.com/v1",
  },
]

function savedBedrock(overrides: Partial<Model> = {}): Model {
  return {
    id: 11,
    model_id: "saved-bedrock",
    category: "llm",
    model_provider: "bedrock",
    model_name: "arn:aws:bedrock:us-west-2:123456789012:inference-profile/example",
    base_url: "",
    bedrock_region: "us-west-2",
    bedrock_auth_mode: "credentials_chain",
    abilities: ["chat"],
    is_active: true,
    is_owner: true,
    can_edit: true,
    can_delete: true,
    is_shared: false,
    ...overrides,
  }
}

function renderDialog(overrides: Partial<React.ComponentProps<typeof ModelManagementDialog>> = {}) {
  return render(
    <ModelManagementDialog
      isOpen
      onOpenChange={vi.fn()}
      initialViewMode="form"
      initialProviderId="bedrock"
      activeTab="llm"
      providers={providers}
      enabledModels={[]}
      defaultModels={{}}
      onSuccess={vi.fn().mockResolvedValue(undefined)}
      {...overrides}
    />,
  )
}

function clickSelectValue(value: string) {
  fireEvent.click(screen.getByText(value))
}

describe("ModelManagementDialog Bedrock settings", () => {
  beforeEach(() => {
    authState.user = { id: "7", is_admin: false }
    apiRequestMock.mockReset()
  })

  afterEach(cleanup)

  it("blocks a regular user's saved credential chain and requires a key when switching auth modes", () => {
    renderDialog({ initialEditingModel: savedBedrock() })

    const save = screen.getByRole("button", { name: "models.form.update" })
    expect(save).toBeDisabled()
    expect(screen.queryByLabelText("models.form.apiKey")).not.toBeInTheDocument()
    expect(screen.getByDisplayValue("us-west-2")).toBeInTheDocument()

    clickSelectValue("models.form.awsCredentialsChainAdminOnly")
    fireEvent.click(screen.getByRole("button", { name: "models.form.bedrockApiKey" }))

    const apiKey = screen.getByLabelText("models.form.apiKey")
    expect(save).toBeDisabled()
    fireEvent.change(apiKey, { target: { value: "bedrock-token" } })
    expect(save).toBeEnabled()
  })

  it("allows an administrator to keep a saved credential chain without exposing an API-key field", () => {
    authState.user = { id: "7", is_admin: true }

    renderDialog({ initialEditingModel: savedBedrock() })

    expect(screen.getByRole("button", { name: "models.form.update" })).toBeEnabled()
    expect(screen.queryByLabelText("models.form.apiKey")).not.toBeInTheDocument()
  })

  it("allows an API-key model edit to retain its saved secret", () => {
    renderDialog({
      initialEditingModel: savedBedrock({ bedrock_auth_mode: "api_key" }),
    })

    expect(screen.getByLabelText("models.form.apiKey")).toHaveValue("")
    expect(screen.getByRole("button", { name: "models.form.update" })).toBeEnabled()
  })

  it("resets provider-specific values when switching away from and back to Bedrock", () => {
    renderDialog()

    fireEvent.change(screen.getByLabelText("models.form.region"), { target: { value: "eu-west-1" } })
    fireEvent.change(screen.getByLabelText("models.form.apiKey"), { target: { value: "secret" } })

    clickSelectValue("Amazon Bedrock")
    fireEvent.click(screen.getByRole("button", { name: "OpenAI" }))
    expect(screen.queryByLabelText("models.form.region")).not.toBeInTheDocument()
    expect(screen.getByLabelText("models.form.apiKey")).toHaveValue("")

    clickSelectValue("OpenAI")
    fireEvent.click(screen.getByRole("button", { name: "Amazon Bedrock" }))
    expect(screen.getByLabelText("models.form.region")).toHaveValue("")
    expect(screen.getByLabelText("models.form.apiKey")).toHaveValue("")
  })

  it("starts with an empty region, labels the endpoint optional, and preserves a custom ARN without refresh", async () => {
    renderDialog({ initialViewMode: "connect" })

    const next = screen.getByRole("button", { name: /models\.dialog\.connect\.next/ })
    expect(screen.getByLabelText("models.form.region")).toHaveValue("")
    expect(next).toBeDisabled()
    expect(screen.getByText(/models\.form\.baseUrl.*common\.optional/)).toBeInTheDocument()
    expect(screen.getByPlaceholderText("models.form.bedrockEndpointPlaceholder")).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("models.form.region"), { target: { value: "us-west-2" } })
    fireEvent.change(screen.getByPlaceholderText("models.dialog.connect.apiKeyPlaceholder"), { target: { value: "token" } })
    expect(next).toBeEnabled()
    fireEvent.click(next)

    await waitFor(() => expect(screen.getByText("models.form.name")).toBeInTheDocument())
    expect(screen.queryByText("common.refresh")).not.toBeInTheDocument()

    const arn = "arn:aws:bedrock:us-west-2:123456789012:inference-profile/example"
    fireEvent.click(screen.getByText("models.form.enterModelName"))
    fireEvent.change(screen.getByPlaceholderText("models.form.customModel"), { target: { value: arn } })
    fireEvent.click(screen.getByRole("button", { name: "models.form.addCustom" }))

    expect(screen.getByText(arn)).toBeInTheDocument()
    expect(screen.queryByText("common.refresh")).not.toBeInTheDocument()
  })
})
