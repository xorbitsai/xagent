import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
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
  const seen: MCPServerFormData[] = []
  function Harness({ onChange }: { onChange: (data: MCPServerFormData) => void }) {
    const [formData, setFormData] = React.useState<MCPServerFormData>({
      name: "proxy",
      transport: "streamable_http",
      description: "",
      config: { url: "https://mcp.example.com/mcp", headers },
      can_edit_global: canEditGlobal,
    })
    React.useEffect(() => {
      onChange(formData)
    }, [formData, onChange])
    return (
      <CustomMcpForm
        mcpFormData={formData}
        setMcpFormData={setFormData}
        serverId={42}
      />
    )
  }

  render(<Harness onChange={(data) => seen.push(data)} />)
  fireEvent.click(screen.getByText("tools.mcp.dialog.advancedOptions"))
  return { latestHeaders: () => seen[seen.length - 1]?.config?.headers }
}

// The row element holds the name input, the value input and, when editable,
// the icon-only delete button.
function headerRow(key: string): HTMLElement {
  const row = screen.getByDisplayValue(key).parentElement
  if (!row) throw new Error(`Row not found for header ${key}`)
  return row
}

function removeButtonForHeader(key: string): HTMLElement {
  return within(headerRow(key)).getByRole("button")
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
    const { latestHeaders } = renderHeadersForm({ "X-Proxy-Client-Secret": "********" })
    const keyInput = screen.getByPlaceholderText(KEY_PLACEHOLDER)
    const valueInput = screen.getByPlaceholderText(VALUE_PLACEHOLDER)

    fireEvent.change(valueInput, { target: { value: "new-secret" } })
    await waitFor(() => expect(valueInput).toHaveValue("new-secret"))
    expect(keyInput).toBeDisabled()
    expect(latestHeaders()).toEqual({ "X-Proxy-Client-Secret": "new-secret" })

    fireEvent.change(valueInput, { target: { value: "" } })
    expect(keyInput).toBeDisabled()
    expect(latestHeaders()).toEqual({ "X-Proxy-Client-Secret": "" })
    fireEvent.blur(valueInput)
    await waitFor(() => expect(valueInput).toHaveValue("********"))
    expect(keyInput).toBeDisabled()
    expect(latestHeaders()).toEqual({ "X-Proxy-Client-Secret": "********" })
  })

  it("delete_persisted_confirms: removing a persisted header asks first", () => {
    const confirmMock = vi.spyOn(window, "confirm").mockReturnValue(false)
    const { latestHeaders } = renderHeadersForm({ "X-Proxy-Client-Secret": "********" })

    fireEvent.click(removeButtonForHeader("X-Proxy-Client-Secret"))
    expect(confirmMock).toHaveBeenCalledWith(CONFIRM_KEY)
    expect(screen.getByDisplayValue("X-Proxy-Client-Secret")).toBeInTheDocument()
    expect(latestHeaders()).toEqual({ "X-Proxy-Client-Secret": "********" })

    confirmMock.mockReturnValue(true)
    fireEvent.click(removeButtonForHeader("X-Proxy-Client-Secret"))
    expect(confirmMock).toHaveBeenCalledTimes(2)
    expect(screen.queryByDisplayValue("X-Proxy-Client-Secret")).not.toBeInTheDocument()
    expect(latestHeaders()).toEqual({})
  })

  it("new_header_plain: a new header row is editable and removed without confirmation", () => {
    const confirmMock = vi.spyOn(window, "confirm").mockReturnValue(false)
    const { latestHeaders } = renderHeadersForm({})

    fireEvent.click(screen.getByRole("button", { name: ADD_HEADER }))
    const keyInput = screen.getByPlaceholderText(KEY_PLACEHOLDER)
    fireEvent.change(keyInput, { target: { value: "X-New" } })
    expect(keyInput).not.toBeDisabled()
    expect(latestHeaders()).toEqual({ "X-New": "" })

    fireEvent.click(removeButtonForHeader("X-New"))
    expect(confirmMock).not.toHaveBeenCalled()
    expect(screen.queryByDisplayValue("X-New")).not.toBeInTheDocument()
    expect(latestHeaders()).toEqual({})
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
    const row = headerRow("X-Proxy-Client-Secret")
    expect(row).toContainElement(screen.getByPlaceholderText(VALUE_PLACEHOLDER))
    expect(within(row).queryByRole("button")).toBeNull()
  })
})
