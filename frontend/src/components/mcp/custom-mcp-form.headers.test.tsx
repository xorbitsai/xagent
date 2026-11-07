import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { CustomMcpForm } from "./custom-mcp-form"
import { MCPServerFormData } from "./custom-api-form"

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: vi.fn(),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

const KEY_PLACEHOLDER = "tools.mcp.dialog.headerKeyPlaceholder"
const VALUE_PLACEHOLDER = "tools.mcp.dialog.headerValuePlaceholder"
const ADD_HEADER = "tools.mcp.dialog.addHeader"
const CONFIRM_KEY = "tools.mcp.dialog.removeSecretConfirm"

function renderHeadersForm(
  headers: Record<string, string>,
  canEditGlobal = true
) {
  function Harness() {
    const [formData, setFormData] = React.useState<MCPServerFormData>({
      name: "proxy",
      transport: "streamable_http",
      description: "",
      config: { url: "https://mcp.example.com/mcp", headers },
      can_edit_global: canEditGlobal,
    })
    return (
      <CustomMcpForm
        mcpFormData={formData}
        setMcpFormData={setFormData}
        serverId={42}
      />
    )
  }

  const result = render(<Harness />)
  fireEvent.click(screen.getByText("tools.mcp.dialog.advancedOptions"))
  return result
}

function removeButtonForHeader(key: string): HTMLButtonElement {
  const button = screen.getByDisplayValue(key).parentElement?.querySelector("button")
  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`Remove button not found for header ${key}`)
  }
  return button
}

describe("CustomMcpForm request header rows", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    cleanup()
  })

  it("persisted_key_locked: a masked header keeps its name locked and its value hidden", () => {
    renderHeadersForm({ "X-Proxy-Client-Secret": "********" })

    const keyInput = screen.getByPlaceholderText(KEY_PLACEHOLDER)
    const valueInput = screen.getByPlaceholderText(VALUE_PLACEHOLDER)
    expect(keyInput).toHaveValue("X-Proxy-Client-Secret")
    expect(keyInput).toBeDisabled()
    expect(valueInput).toHaveAttribute("type", "password")
    expect(valueInput).toHaveValue("********")
  })

  it("blur_restores_mask: clearing a persisted header value restores the mask", async () => {
    renderHeadersForm({ "X-Proxy-Client-Secret": "********" })
    const keyInput = screen.getByPlaceholderText(KEY_PLACEHOLDER)
    const valueInput = screen.getByPlaceholderText(VALUE_PLACEHOLDER)

    fireEvent.change(valueInput, { target: { value: "new-secret" } })
    await waitFor(() => expect(valueInput).toHaveValue("new-secret"))
    expect(keyInput).toBeDisabled()

    fireEvent.change(valueInput, { target: { value: "" } })
    expect(keyInput).toBeDisabled()
    fireEvent.blur(valueInput)
    await waitFor(() => expect(valueInput).toHaveValue("********"))
    expect(keyInput).toBeDisabled()
  })

  it("delete_persisted_confirms: removing a persisted header asks first", () => {
    const confirmMock = vi.spyOn(window, "confirm").mockReturnValue(false)
    renderHeadersForm({ "X-Proxy-Client-Secret": "********" })

    fireEvent.click(removeButtonForHeader("X-Proxy-Client-Secret"))
    expect(confirmMock).toHaveBeenCalledWith(CONFIRM_KEY)
    expect(screen.getByDisplayValue("X-Proxy-Client-Secret")).toBeInTheDocument()

    confirmMock.mockReturnValue(true)
    fireEvent.click(removeButtonForHeader("X-Proxy-Client-Secret"))
    expect(confirmMock).toHaveBeenCalledTimes(2)
    expect(screen.queryByDisplayValue("X-Proxy-Client-Secret")).not.toBeInTheDocument()
  })

  it("new_header_plain: a new header row is editable and removed without confirmation", () => {
    const confirmMock = vi.spyOn(window, "confirm").mockReturnValue(false)
    renderHeadersForm({})

    fireEvent.click(screen.getByRole("button", { name: ADD_HEADER }))
    const keyInput = screen.getByPlaceholderText(KEY_PLACEHOLDER)
    fireEvent.change(keyInput, { target: { value: "X-New" } })
    expect(keyInput).not.toBeDisabled()

    fireEvent.click(removeButtonForHeader("X-New"))
    expect(confirmMock).not.toHaveBeenCalled()
    expect(screen.queryByDisplayValue("X-New")).not.toBeInTheDocument()
  })

  it("shows the header-specific empty text when there are no headers", () => {
    renderHeadersForm({})

    expect(screen.getByText("tools.mcp.dialog.noCustomHeaders")).toBeInTheDocument()
    expect(screen.queryByText("tools.mcp.dialog.noEnvVariables")).not.toBeInTheDocument()
  })

  it("read-only: a caller who cannot edit global config cannot change headers", () => {
    renderHeadersForm({ "X-Proxy-Client-Secret": "********" }, false)

    expect(screen.getByPlaceholderText(KEY_PLACEHOLDER)).toBeDisabled()
    expect(screen.getByPlaceholderText(VALUE_PLACEHOLDER)).toBeDisabled()
    expect(screen.queryByRole("button", { name: ADD_HEADER })).toBeNull()
  })
})
