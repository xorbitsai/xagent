import { fireEvent, render, screen, within } from "@testing-library/react"
import React from "react"
import { describe, expect, it, vi } from "vitest"

import { MultiSelect } from "./multi-select"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

describe("MultiSelect", () => {
  it("filters same-name options by description without changing their values", () => {
    const onValuesChange = vi.fn()
    render(<MultiSelect
      values={[]}
      onValuesChange={onValuesChange}
      options={[
        { value: "local", label: "GPT", description: "localhost:9997 (openai-compatible)" },
        { value: "remote", label: "GPT", description: "models.example.com:8443 (openai-compatible)" },
      ]}
      placeholder="Select models"
      searchable
    />)

    fireEvent.click(screen.getByText("Select models"))
    fireEvent.change(screen.getByPlaceholderText("Search..."), {
      target: { value: "MODELS.EXAMPLE.COM:8443" },
    })

    expect(screen.queryByRole("button", {
      name: "GPT localhost:9997 (openai-compatible)",
    })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", {
      name: "GPT models.example.com:8443 (openai-compatible)",
    }))
    expect(onValuesChange).toHaveBeenCalledWith(["remote"])
  })

  it("renders selectedLabel in tags and falls back to label when it is omitted", () => {
    const onValuesChange = vi.fn()
    const selectedLabel = "GPT · localhost:9997 (openai-compatible)"
    render(<MultiSelect
      values={["local", "default"]}
      onValuesChange={onValuesChange}
      options={[
        { value: "local", label: "GPT", selectedLabel },
        { value: "default", label: "Default model" },
      ]}
    />)

    expect(screen.getByText(selectedLabel)).toBeInTheDocument()
    expect(screen.getByText("Default model")).toBeInTheDocument()
    const tag = screen.getByText(selectedLabel).parentElement!
    fireEvent.click(within(tag).getByRole("button"))
    expect(onValuesChange).toHaveBeenCalledWith(["default"])
  })
})
