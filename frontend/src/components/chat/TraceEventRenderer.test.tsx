/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const clipboardWriteTextMock = vi.hoisted(() => vi.fn())
const appContextMock = vi.hoisted(() => ({
  dispatch: vi.fn(),
  filesDisabled: false,
  openFilePreview: vi.fn(),
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) => {
      if (vars?.tool) return `${key}:${vars.tool}`
      if (vars?.worker) return `${key}:${vars.worker}`
      if (vars && "count" in vars) return `${key}:${vars.count}`
      return key
    },
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => appContextMock,
}))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
  }
})

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/components/file/docx-preview-renderer", () => ({
  DocxPreviewRenderer: ({ base64Content }: { base64Content: string }) => (
    <div data-testid="docx-preview">{base64Content}</div>
  ),
}))

vi.mock("@/components/file/excel-preview-renderer", () => ({
  ExcelPreviewRenderer: ({ base64Content }: { base64Content: string }) => (
    <div data-testid="excel-preview">{base64Content}</div>
  ),
}))

vi.mock("@/components/file/pptx-preview-renderer", () => ({
  PptxPreviewRenderer: ({
    base64Content,
    fileId,
  }: {
    base64Content?: string
    fileId?: string
  }) => <div data-testid="pptx-preview">{base64Content ?? fileId ?? ""}</div>,
}))

import { TraceEventRenderer } from "./TraceEventRenderer"

describe("TraceEventRenderer", () => {
  it("ignores null, primitive, and malformed trace event entries", () => {
    expect(() => render(
      <TraceEventRenderer
        events={[
          null,
          42,
          "not-an-event",
          { event_type: 17, data: "not-an-object" },
          { event_type: "agent_progress", data: null },
        ] as unknown as React.ComponentProps<typeof TraceEventRenderer>["events"]}
      />,
    )).not.toThrow()
  })

  beforeEach(() => {
    window.scrollTo = vi.fn()
    appContextMock.filesDisabled = false
    appContextMock.dispatch.mockReset()
    appContextMock.openFilePreview.mockReset()
    clipboardWriteTextMock.mockReset()
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: clipboardWriteTextMock },
    })
  })

  afterEach(() => {
    cleanup()
    apiRequestMock.mockReset()
    vi.restoreAllMocks()
  })

  it("renders image artifacts inline from tool results", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(["image-bytes"], { type: "image/png" }),
    })

    render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "step-1",
            timestamp: Date.now(),
            data: { step_name: "Generate image", description: "Generate image" },
          },
          {
            event_id: "tool-start",
            event_type: "tool_execution_start",
            step_id: "step-1",
            timestamp: Date.now(),
            data: { tool_name: "generate_image", tool_args: { prompt: "test" } },
          },
          {
            event_id: "tool-end",
            event_type: "tool_execution_end",
            step_id: "step-1",
            timestamp: Date.now(),
            data: {
              result: {
                success: true,
                artifacts: [
                  {
                    type: "image",
                    file_id: "582e7b79-4de9-4905-b73b-7d5a70ad64fe",
                    filename: "generated_image.png",
                    display: "inline",
                  },
                ],
              },
            },
          },
        ]}
      />,
    )

    fireEvent.click(
      screen.getByRole("button", {
        name: /traceEventRenderer.executeTool:Generate Image/,
      }),
    )

    const image = await screen.findByAltText("generated_image.png")
    await waitFor(() => {
      expect(image.getAttribute("src")).toMatch(/^blob:/)
    })
    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/files/preview/582e7b79-4de9-4905-b73b-7d5a70ad64fe",
      expect.objectContaining({ cache: "no-cache" }),
    )
  })

  it("renders trace file artifacts and markdown as inert text when files are disabled", () => {
    appContextMock.filesDisabled = true

    render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "step-disabled",
            timestamp: Date.now(),
            data: {
              step_name: "Create disabled artifact",
              description: "Create disabled artifact",
            },
          },
          {
            event_id: "tool-start",
            event_type: "tool_execution_start",
            step_id: "step-disabled",
            timestamp: Date.now() + 1,
            data: {
              tool_name: "document_tool",
              tool_args: {},
            },
          },
          {
            event_id: "tool-end",
            event_type: "tool_execution_end",
            step_id: "step-disabled",
            timestamp: Date.now() + 2,
            data: {
              result: {
                success: true,
                output: "[trace report.docx](file:trace-output-id)",
                artifacts: [
                  {
                    type: "document",
                    file_id: "trace-artifact-id",
                    filename: "artifact.docx",
                    display: "inline",
                  },
                  {
                    type: "image",
                    preview_url: "https://files.example/secret-preview",
                    display: "inline",
                  },
                ],
              },
            },
          },
        ]}
      />,
    )

    fireEvent.click(
      screen.getByRole("button", {
        name: /traceEventRenderer.executeTool:Document Tool/,
      }),
    )

    expect(screen.getByText("artifact.docx")).not.toHaveAttribute("href")
    expect(screen.getByText("image")).not.toHaveAttribute("href")
    expect(screen.getByText("trace report.docx")).not.toHaveAttribute("href")
    expect(document.body.innerHTML).not.toContain(
      "https://files.example/secret-preview",
    )
    expect(screen.queryByTestId("docx-preview")).not.toBeInTheDocument()
    expect(apiRequestMock).not.toHaveBeenCalled()

    fireEvent.click(screen.getByText("artifact.docx"))
    fireEvent.click(screen.getByText("trace report.docx"))
    expect(appContextMock.openFilePreview).not.toHaveBeenCalled()
    expect(appContextMock.dispatch).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "SET_FILE_PREVIEW_CONTENT" }),
    )
  })

  it("uses the files-disabled representation for trace object DOM and copies", () => {
    appContextMock.filesDisabled = true

    const output = {
      artifacts: [{
        fileId: "trace-output-id",
        path: "/private/trace-output.pdf",
        previewUrl: "https://files.example/preview/trace-output",
        signed_url: "https://files.example/signed/trace-output",
        url: "https://files.example/generic-trace-output-url",
        fileName: "trace-output.pdf",
        type: "application/pdf",
        text: "[open trace output](file:trace-output-id)",
      }],
      message: "Open (/private/tenant/secret.txt) with path=/private/tenant/config.json and output/report.pdf",
      summary: "Summary /private/tenant/summary.txt",
      detail: "Detail /tmp/tenant/detail.txt",
      note: "Note /workspace/tenant/note.txt",
      stdout: "stdout: /sandbox/tenant/stdout.log",
      stderr: "stderr: /data/tenant/stderr.log",
      runtimePaths: "Runtime /app/src, /opt/xagent, /var/tmp, /srv/app, and /etc/passwd",
      routes: "Keep /v1/openapi.json and /care/export.csv",
      taskUrl: "https://api.example/tasks/42",
    }

    const { container } = render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "trace-safe-output",
            timestamp: Date.now(),
            data: { step_name: "Inspect output", description: "Inspect output" },
          },
          {
            event_id: "tool-start",
            event_type: "tool_execution_start",
            step_id: "trace-safe-output",
            timestamp: Date.now() + 1,
            data: { tool_name: "inspect_output", tool_args: {} },
          },
          {
            event_id: "tool-end",
            event_type: "tool_execution_end",
            step_id: "trace-safe-output",
            timestamp: Date.now() + 2,
            data: { result: { success: true, output: output as unknown as string } },
          },
        ]}
      />,
    )

    fireEvent.click(screen.getByRole("button", {
      name: /traceEventRenderer.executeTool:Inspect Output/,
    }))

    expect(container).toHaveTextContent("trace-output.pdf")
    expect(container).toHaveTextContent("open trace output")
    expect(container).toHaveTextContent("Open (secret.txt) with path=config.json and report.pdf")
    expect(container).toHaveTextContent("Summary summary.txt")
    expect(container).toHaveTextContent("Detail detail.txt")
    expect(container).toHaveTextContent("Note note.txt")
    expect(container).toHaveTextContent("stdout: stdout.log")
    expect(container).toHaveTextContent("stderr: stderr.log")
    expect(container).toHaveTextContent("Runtime src, xagent, tmp, app, and passwd")
    expect(container).toHaveTextContent("Keep /v1/openapi.json and /care/export.csv")
    expect(container).toHaveTextContent("https://api.example/tasks/42")
    expect(container.innerHTML).not.toContain("trace-output-id")
    expect(container.innerHTML).not.toContain("/private/trace-output.pdf")
    expect(container.innerHTML).not.toContain("https://files.example/preview/trace-output")
    expect(container.innerHTML).not.toContain("https://files.example/signed/trace-output")
    expect(container.innerHTML).not.toContain("https://files.example/generic-trace-output-url")
    expect(container.innerHTML).not.toContain("/private/tenant/secret.txt")
    expect(container.innerHTML).not.toContain("/private/tenant/config.json")
    expect(container.innerHTML).not.toContain("output/report.pdf")
    expect(container.innerHTML).not.toContain("/app/src")
    expect(container.innerHTML).not.toContain("/opt/xagent")
    expect(container.innerHTML).not.toContain("/var/tmp")
    expect(container.innerHTML).not.toContain("/srv/app")
    expect(container.innerHTML).not.toContain("/etc/passwd")

    const copyButtons = screen.getAllByTitle("traceEventRenderer.copy")
    fireEvent.click(copyButtons[copyButtons.length - 1])

    const copied = clipboardWriteTextMock.mock.calls.at(-1)?.[0]
    expect(copied).not.toContain("trace-output-id")
    expect(copied).not.toContain("/private/trace-output.pdf")
    expect(copied).not.toContain("https://files.example/preview/trace-output")
    expect(copied).not.toContain("https://files.example/signed/trace-output")
    expect(copied).not.toContain("https://files.example/generic-trace-output-url")
    expect(copied).not.toContain("/private/tenant/secret.txt")
    expect(copied).not.toContain("/private/tenant/config.json")
    expect(copied).not.toContain("output/report.pdf")
    expect(copied).not.toContain("/app/src")
    expect(copied).not.toContain("/opt/xagent")
    expect(copied).not.toContain("/var/tmp")
    expect(copied).not.toContain("/srv/app")
    expect(copied).not.toContain("/etc/passwd")
    expect(JSON.parse(copied)).toEqual({
      artifacts: [{
        fileName: "trace-output.pdf",
        type: "application/pdf",
        text: "open trace output",
      }],
      message: "Open (secret.txt) with path=config.json and report.pdf",
      summary: "Summary summary.txt",
      detail: "Detail detail.txt",
      note: "Note note.txt",
      stdout: "stdout: stdout.log",
      stderr: "stderr: stderr.log",
      runtimePaths: "Runtime src, xagent, tmp, app, and passwd",
      routes: "Keep /v1/openapi.json and /care/export.csv",
      taskUrl: "https://api.example/tasks/42",
    })
  })

  it("terminates cyclic tool output at the renderer boundary without exposing local paths", () => {
    appContextMock.filesDisabled = true
    const output: Record<string, unknown> = {
      label: "cyclic output",
      file_path: "/private/tenant/cyclic-secret.txt",
    }
    output.self = output

    const { container } = render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "cyclic-output",
            timestamp: Date.now(),
            data: { step_name: "Inspect cyclic output", description: "Inspect cyclic output" },
          },
          {
            event_id: "tool-start",
            event_type: "tool_execution_start",
            step_id: "cyclic-output",
            timestamp: Date.now() + 1,
            data: { tool_name: "inspect_cyclic", tool_args: {} },
          },
          {
            event_id: "tool-end",
            event_type: "tool_execution_end",
            step_id: "cyclic-output",
            timestamp: Date.now() + 2,
            data: { result: { success: true, output } },
          },
        ]}
      />,
    )

    fireEvent.click(screen.getByRole("button", {
      name: /traceEventRenderer.executeTool:Inspect Cyclic/,
    }))

    expect(container).toHaveTextContent("[Circular]")
    expect(container.innerHTML).not.toContain("/private/tenant/cyclic-secret.txt")
  })

  it("projects the complete tool result before selecting a files-disabled message", () => {
    appContextMock.filesDisabled = true

    const { container } = render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "pptx-safe-output",
            timestamp: Date.now(),
            data: { step_name: "Create presentation", description: "Create presentation" },
          },
          {
            event_id: "tool-start",
            event_type: "tool_execution_start",
            step_id: "pptx-safe-output",
            timestamp: Date.now() + 1,
            data: { tool_name: "pptx_tool", tool_args: {} },
          },
          {
            event_id: "tool-end",
            event_type: "tool_execution_end",
            step_id: "pptx-safe-output",
            timestamp: Date.now() + 2,
            data: {
              result: {
                success: true,
                output_path: "/private/generated/deck.pptx",
                message: "Created PPTX file: /private/generated/deck.pptx",
              },
            },
          },
        ]}
      />,
    )

    fireEvent.click(screen.getByRole("button", {
      name: /traceEventRenderer.executeTool:Pptx Tool/,
    }))

    expect(container).toHaveTextContent("Created PPTX file: deck.pptx")
    expect(container.innerHTML).not.toContain("/private/generated/deck.pptx")

    const copyButtons = screen.getAllByTitle("traceEventRenderer.copy")
    fireEvent.click(copyButtons[copyButtons.length - 1])
    expect(clipboardWriteTextMock).toHaveBeenLastCalledWith(
      "Created PPTX file: deck.pptx",
    )
  })

  it("keeps the selected raw tool message when files are enabled", () => {
    const { container } = render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "pptx-legacy-output",
            timestamp: Date.now(),
            data: { step_name: "Create presentation", description: "Create presentation" },
          },
          {
            event_id: "tool-start",
            event_type: "tool_execution_start",
            step_id: "pptx-legacy-output",
            timestamp: Date.now() + 1,
            data: { tool_name: "pptx_tool", tool_args: {} },
          },
          {
            event_id: "tool-end",
            event_type: "tool_execution_end",
            step_id: "pptx-legacy-output",
            timestamp: Date.now() + 2,
            data: {
              result: {
                success: true,
                output_path: "/private/generated/deck.pptx",
                message: "Created PPTX file: /private/generated/deck.pptx",
              },
            },
          },
        ]}
      />,
    )

    fireEvent.click(screen.getByRole("button", {
      name: /traceEventRenderer.executeTool:Pptx Tool/,
    }))

    expect(container).toHaveTextContent(
      "Created PPTX file: /private/generated/deck.pptx",
    )

    const copyButtons = screen.getAllByTitle("traceEventRenderer.copy")
    fireEvent.click(copyButtons[copyButtons.length - 1])
    expect(clipboardWriteTextMock).toHaveBeenLastCalledWith(
      "Created PPTX file: /private/generated/deck.pptx",
    )
  })

  it("removes prepared HTML asset identities and locations from DOM and copies", () => {
    appContextMock.filesDisabled = true

    const { container } = render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "prepared-asset-output",
            timestamp: Date.now(),
            data: { step_name: "Prepare asset", description: "Prepare asset" },
          },
          {
            event_id: "tool-start",
            event_type: "tool_execution_start",
            step_id: "prepared-asset-output",
            timestamp: Date.now() + 1,
            data: { tool_name: "prepare_html_asset", tool_args: {} },
          },
          {
            event_id: "tool-end",
            event_type: "tool_execution_end",
            step_id: "prepared-asset-output",
            timestamp: Date.now() + 2,
            data: {
              result: {
                success: true,
                source_file_id: "source-file-id",
                asset_file_id: "asset-file-id",
                html_src: "assets/chart.png",
                file_id: "registered-file-id",
                filename: "chart.png",
                preview_url: "https://files.example/preview/registered-file-id",
              },
            },
          },
        ]}
      />,
    )

    fireEvent.click(screen.getByRole("button", {
      name: /traceEventRenderer.executeTool:Prepare Html Asset/,
    }))

    expect(container).toHaveTextContent("chart.png")
    expect(container.innerHTML).not.toContain("source-file-id")
    expect(container.innerHTML).not.toContain("asset-file-id")
    expect(container.innerHTML).not.toContain("registered-file-id")
    expect(container.innerHTML).not.toContain("assets/chart.png")
    expect(container.innerHTML).not.toContain("https://files.example/preview")

    const copyButtons = screen.getAllByTitle("traceEventRenderer.copy")
    fireEvent.click(copyButtons[copyButtons.length - 1])
    expect(JSON.parse(clipboardWriteTextMock.mock.calls.at(-1)?.[0])).toEqual({
      success: true,
      filename: "chart.png",
    })
  })

  it("does not expose file-tool paths in summaries or tool details when files are disabled", () => {
    appContextMock.filesDisabled = true

    const { container } = render(
      <TraceEventRenderer
        events={[
          {
            event_id: "python-start",
            event_type: "react_task_start",
            step_id: "python-path",
            timestamp: Date.now(),
            data: { step_name: "Run Python", description: "Run Python" },
          },
          {
            event_id: "python-tool",
            event_type: "tool_execution_start",
            step_id: "python-path",
            timestamp: Date.now() + 1,
            data: {
              tool_name: "python_executor",
              tool_args: {
                code: "print('safe python text')",
                file_path: "/private/python/report.py",
                filename: "report.py",
              },
            },
          },
          {
            event_id: "file-start",
            event_type: "react_task_start",
            step_id: "file-path",
            timestamp: Date.now() + 2,
            data: { step_name: "Write file", description: "Write file" },
          },
          {
            event_id: "file-tool",
            event_type: "tool_execution_start",
            step_id: "file-path",
            timestamp: Date.now() + 3,
            data: {
              tool_name: "file_writer",
              tool_args: {
                file_path: "/private/files/report.txt",
                path: "/private/files/alternate.txt",
                filename: "report.txt",
                text: "safe file text",
              },
            },
          },
        ]}
      />,
    )

    fireEvent.click(screen.getByRole("button", {
      name: /traceEventRenderer.executeTool:Python Executor/,
    }))
    fireEvent.click(screen.getByRole("button", {
      name: /traceEventRenderer.executeTool:File Writer/,
    }))

    expect(container).toHaveTextContent("report.py")
    expect(container).toHaveTextContent("report.txt")
    expect(container).toHaveTextContent("safe python text")
    expect(container).toHaveTextContent("safe file text")
    expect(container.innerHTML).not.toContain("/private/python/report.py")
    expect(container.innerHTML).not.toContain("/private/files/report.txt")
    expect(container.innerHTML).not.toContain("/private/files/alternate.txt")
  })

  it("projects whole specialized tool actions before rendering code, commands, queries, and file content", () => {
    appContextMock.filesDisabled = true
    const { container } = render(
      <TraceEventRenderer
        events={[
          { event_id: "python-step", event_type: "react_task_start", step_id: "python", timestamp: 1, data: { step_name: "Python", description: "Read /private/shared/report.txt" } },
          { event_id: "python-tool", event_type: "tool_execution_start", step_id: "python", timestamp: 2, data: { tool_name: "python_executor", tool_args: { file_path: "/private/shared/report.txt", filename: "report.txt", code: "open('/private/shared/report.txt')" } } },
          { event_id: "bash-step", event_type: "react_task_start", step_id: "bash", timestamp: 3, data: { step_name: "Bash", description: "Bash" } },
          { event_id: "bash-tool", event_type: "tool_execution_start", step_id: "bash", timestamp: 4, data: { tool_name: "bash", tool_args: { output_dir: "/private/shared", filename: "report.txt", command: "cat /private/shared/report.txt" } } },
          { event_id: "search-step", event_type: "react_task_start", step_id: "search", timestamp: 5, data: { step_name: "Search", description: "Search" } },
          { event_id: "search-tool", event_type: "tool_execution_start", step_id: "search", timestamp: 6, data: { tool_name: "web_search", tool_args: { source_path: "/private/shared", filename: "report.txt", query: "find /private/shared/report.txt" } } },
          { event_id: "file-step", event_type: "react_task_start", step_id: "file", timestamp: 7, data: { step_name: "File", description: "File" } },
          { event_id: "file-tool", event_type: "tool_execution_start", step_id: "file", timestamp: 8, data: { tool_name: "file_writer", tool_args: { file_path: "/private/shared/report.txt", filename: "report.txt", content: "write /private/shared/report.txt" } } },
        ]}
      />,
    )

    for (const button of screen.getAllByRole("button", { name: /traceEventRenderer.executeTool/ })) {
      fireEvent.click(button)
    }

    expect(container.innerHTML).not.toContain("/private/shared")
    expect(container).toHaveTextContent("report.txt")
  })

  it("uses the shared safe representation for reasoning and failure trace DOM", () => {
    appContextMock.filesDisabled = true

    const { container } = render(
      <TraceEventRenderer
        events={[
          {
            event_id: "llm-step",
            event_type: "react_task_start",
            step_id: "llm-safe-failure",
            timestamp: Date.now(),
            data: { step_name: "Reason safely", description: "Reason safely" },
          },
          {
            event_id: "llm-start",
            event_type: "llm_call_start",
            step_id: "llm-safe-failure",
            timestamp: Date.now() + 1,
            data: { model_name: "test-model" },
          },
          {
            event_id: "llm-end",
            event_type: "llm_call_end",
            step_id: "llm-safe-failure",
            timestamp: Date.now() + 2,
            data: {
              response: {
                reasoning: "[reasoning report](file:reasoning-secret-id) from `/private/reasoning.txt`",
              },
            },
          },
          {
            event_id: "llm-failure-step",
            event_type: "react_task_start",
            step_id: "llm-error-safe-failure",
            timestamp: Date.now() + 3,
            data: { step_name: "LLM failure", description: "LLM failure" },
          },
          {
            event_id: "llm-failure-start",
            event_type: "llm_call_start",
            step_id: "llm-error-safe-failure",
            timestamp: Date.now() + 4,
            data: { model_name: "test-model" },
          },
          {
            event_id: "llm-failed",
            event_type: "llm_call_failed",
            step_id: "llm-error-safe-failure",
            timestamp: Date.now() + 5,
            data: {
              error: "[LLM failure report](file:llm-failure-secret-id)",
            },
          },
          {
            event_id: "tool-step",
            event_type: "react_task_start",
            step_id: "tool-safe-failure",
            timestamp: Date.now() + 6,
            data: { step_name: "Fail safely", description: "Fail safely" },
          },
          {
            event_id: "tool-start",
            event_type: "tool_execution_start",
            step_id: "tool-safe-failure",
            timestamp: Date.now() + 7,
            data: { tool_name: "inspect_output", tool_args: {} },
          },
          {
            event_id: "tool-failed",
            event_type: "tool_execution_failed",
            step_id: "tool-safe-failure",
            timestamp: Date.now() + 8,
            data: {
              error: "[tool failure report](file:tool-failure-secret-id)",
            },
          },
          {
            event_id: "generic-step",
            event_type: "react_task_start",
            step_id: "generic-safe-failure",
            timestamp: Date.now() + 9,
            data: { step_name: "Generic failure", description: "Generic failure" },
          },
          {
            event_id: "generic-failed",
            event_type: "trace_error",
            step_id: "generic-safe-failure",
            timestamp: Date.now() + 10,
            data: {
              error: "[generic failure report](file:generic-failure-secret-id)",
            },
          },
        ]}
      />,
    )

    fireEvent.click(screen.getByRole("button", {
      name: /traceEventRenderer.executeTool:Inspect Output/,
    }))
    fireEvent.click(screen.getByRole("button", {
      name: /traceEventRenderer.executionFailed/,
    }))

    expect(container).toHaveTextContent("reasoning report")
    expect(container).toHaveTextContent("reasoning.txt")
    expect(container).toHaveTextContent("LLM failure report")
    expect(container).toHaveTextContent("tool failure report")
    expect(container).toHaveTextContent("generic failure report")
    expect(container.innerHTML).not.toContain("reasoning-secret-id")
    expect(container.innerHTML).not.toContain("/private/reasoning.txt")
    expect(container.innerHTML).not.toContain("llm-failure-secret-id")
    expect(container.innerHTML).not.toContain("tool-failure-secret-id")
    expect(container.innerHTML).not.toContain("generic-failure-secret-id")
  })

  it("renders pptx artifacts inline through PptxPreviewRenderer", async () => {
    render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "step-1",
            timestamp: Date.now(),
            data: { step_name: "Create slides", description: "Create slides" },
          },
          {
            event_id: "tool-start",
            event_type: "tool_execution_start",
            step_id: "step-1",
            timestamp: Date.now(),
            data: { tool_name: "pptx_tool", tool_args: { topic: "test" } },
          },
          {
            event_id: "tool-end",
            event_type: "tool_execution_end",
            step_id: "step-1",
            timestamp: Date.now(),
            data: {
              result: {
                success: true,
                artifacts: [
                  {
                    type: "presentation",
                    file_id: "slides-file-id",
                    filename: "report.pptx",
                    display: "inline",
                  },
                ],
              },
            },
          },
        ]}
      />,
    )

    fireEvent.click(
      screen.getByRole("button", {
        name: /traceEventRenderer.executeTool:Pptx Tool/,
      }),
    )

    // Managed fileId path: mount PptxPreviewRenderer immediately and let it
    // probe the PDF endpoint first instead of eagerly downloading raw bytes.
    expect(await screen.findByTestId("pptx-preview")).toHaveTextContent("slides-file-id")
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      "http://api.local/api/files/public/preview/slides-file-id",
      expect.anything(),
    )
  })

  it("renders docx artifacts inline with the document renderer", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      arrayBuffer: async () => new Uint8Array([65, 66]).buffer,
    })

    render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "step-1",
            timestamp: Date.now(),
            data: { step_name: "Create doc", description: "Create doc" },
          },
          {
            event_id: "tool-start",
            event_type: "tool_execution_start",
            step_id: "step-1",
            timestamp: Date.now(),
            data: { tool_name: "document_tool", tool_args: {} },
          },
          {
            event_id: "tool-end",
            event_type: "tool_execution_end",
            step_id: "step-1",
            timestamp: Date.now(),
            data: {
              result: {
                success: true,
                artifacts: [
                  {
                    type: "document",
                    file_id: "doc-file-id",
                    filename: "summary.docx",
                    display: "inline",
                  },
                ],
              },
            },
          },
        ]}
      />,
    )

    fireEvent.click(
      screen.getByRole("button", {
        name: /traceEventRenderer.executeTool:Document Tool/,
      }),
    )

    expect(await screen.findByTestId("docx-preview")).toHaveTextContent("QUI=")
    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/files/public/preview/doc-file-id",
      expect.objectContaining({ cache: "no-cache" }),
    )
  })

  it("renders xlsx artifacts inline with the spreadsheet renderer", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      arrayBuffer: async () => new Uint8Array([88, 89]).buffer,
    })

    render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "step-1",
            timestamp: Date.now(),
            data: { step_name: "Create workbook", description: "Create workbook" },
          },
          {
            event_id: "tool-start",
            event_type: "tool_execution_start",
            step_id: "step-1",
            timestamp: Date.now(),
            data: { tool_name: "excel", tool_args: {} },
          },
          {
            event_id: "tool-end",
            event_type: "tool_execution_end",
            step_id: "step-1",
            timestamp: Date.now(),
            data: {
              result: {
                success: true,
                artifacts: [
                  {
                    type: "spreadsheet",
                    file_id: "sheet-file-id",
                    filename: "data.xlsx",
                    display: "inline",
                  },
                ],
              },
            },
          },
        ]}
      />,
    )

    fireEvent.click(
      screen.getByRole("button", {
        name: /traceEventRenderer.executeTool:Excel/,
      }),
    )

    expect(await screen.findByTestId("excel-preview")).toHaveTextContent("WFk=")
  })

  it("renders assistant content on the tool call details", () => {
    render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "step-1",
            timestamp: Date.now(),
            data: { step_name: "Search", description: "Search" },
          },
          {
            event_id: "tool-start",
            event_type: "tool_execution_start",
            step_id: "step-1",
            timestamp: Date.now(),
            data: {
              tool_name: "web_search",
              tool_params: { query: "ai news" },
              assistant_content: "I need current search results first.",
            },
          },
        ]}
      />,
    )

    expect(screen.getByText("I need current search results first.")).toBeInTheDocument()
    expect(screen.queryByText("traceEventRenderer.toolCallNote")).not.toBeInTheDocument()
  })

  it("interleaves agent progress into the active thinking process", () => {
    render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "step-1",
            timestamp: 1000,
            data: {},
          },
          {
            event_id: "first-tool",
            event_type: "tool_execution_start",
            step_id: "step-1",
            timestamp: 2000,
            data: { tool_name: "first_tool", tool_params: { query: "first" } },
          },
          {
            event_id: "second-tool",
            event_type: "tool_execution_start",
            step_id: "step-1",
            timestamp: 4000,
            data: { tool_name: "second_tool", tool_params: { query: "second" } },
          },
          {
            event_id: "end",
            event_type: "react_task_end",
            step_id: "step-1",
            timestamp: 5000,
            data: {},
          },
          {
            event_id: "progress",
            event_type: "agent_progress",
            step_id: "step-1",
            timestamp: 3000,
            data: {
              message: "Still searching the remaining sources.",
              message_type: "progress",
            },
          },
          {
            event_id: "legacy-progress",
            event_type: "agent_message",
            timestamp: 3500,
            data: {
              message: "Legacy progress also stays in the process.",
              message_type: "progress",
              expect_response: false,
            },
          },
        ]}
      />,
    )

    const stepToggles = screen.getAllByRole("button", {
      name: /traceEventRenderer\.(thoughtProcess|taskExecution)/,
    })
    expect(stepToggles).toHaveLength(1)

    fireEvent.click(stepToggles[0])

    expect(screen.getByText("Legacy progress also stays in the process.")).toBeInTheDocument()
    expect(screen.getByText("Still searching the remaining sources.")).toBeInTheDocument()
    expect(screen.queryByText("traceEventRenderer.progressMessage")).not.toBeInTheDocument()

    const renderedText = document.body.textContent || ""
    expect(renderedText.indexOf("traceEventRenderer.executeTool:First Tool")).toBeLessThan(
      renderedText.indexOf("Still searching the remaining sources."),
    )
    expect(renderedText.indexOf("Legacy progress also stays in the process.")).toBeLessThan(
      renderedText.indexOf("traceEventRenderer.executeTool:Second Tool"),
    )
  })

  it("keeps trace event ordering stable when timestamps are invalid", () => {
    render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "step-1",
            timestamp: -1,
            data: {},
          },
          {
            event_id: "invalid-start",
            event_type: "tool_execution_start",
            step_id: "step-1",
            timestamp: "not-a-date" as unknown as number,
            data: { tool_name: "invalid_time_tool", tool_params: { query: "invalid" } },
          },
          {
            event_id: "also-invalid-start",
            event_type: "tool_execution_start",
            step_id: "step-1",
            timestamp: undefined as unknown as number,
            data: { tool_name: "missing_time_tool", tool_params: { query: "missing" } },
          },
        ]}
      />,
    )

    const renderedText = document.body.textContent || ""
    expect(renderedText.indexOf("traceEventRenderer.executeTool:Invalid Time Tool")).toBeLessThan(
      renderedText.indexOf("traceEventRenderer.executeTool:Missing Time Tool"),
    )
  })

  it("collapses completed thinking process and keeps it visibly expandable", () => {
    render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "step-1",
            timestamp: Date.now(),
            data: {},
          },
          {
            event_id: "tool-start",
            event_type: "tool_execution_start",
            step_id: "step-1",
            timestamp: Date.now(),
            data: { tool_name: "web_search", tool_params: { query: "ai news" } },
          },
          {
            event_id: "tool-end",
            event_type: "tool_execution_end",
            step_id: "step-1",
            timestamp: Date.now(),
            data: { result: { success: true, output: "done" } },
          },
          {
            event_id: "end",
            event_type: "react_task_end",
            step_id: "step-1",
            timestamp: Date.now(),
            data: {},
          },
        ]}
      />,
    )

    const toggle = screen.getByRole("button", {
      name: /traceEventRenderer.thoughtProcess/,
    })

    expect(toggle).toHaveAttribute("aria-expanded", "false")
    expect(screen.getByText("traceEventRenderer.showProcess")).toBeInTheDocument()
    expect(screen.queryByText(/traceEventRenderer.executeTool:Web Search/)).not.toBeInTheDocument()

    fireEvent.click(toggle)

    expect(toggle).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByText("traceEventRenderer.hideProcess")).toBeInTheDocument()
    expect(screen.getByText(/traceEventRenderer.executeTool:Web Search/)).toBeInTheDocument()
  })

  it("shows the execution plan action only in a process with DAG execution events", () => {
    const onOpenExecutionPlan = vi.fn()
    const { rerender } = render(
      <TraceEventRenderer
        onOpenExecutionPlan={onOpenExecutionPlan}
        events={[
          {
            event_id: "react-start",
            event_type: "react_task_start",
            step_id: "step-1",
            timestamp: 1000,
            data: {},
          },
        ]}
      />,
    )

    expect(screen.queryByTitle("chatPage.executionPlan.tooltip")).not.toBeInTheDocument()

    rerender(
      <TraceEventRenderer
        onOpenExecutionPlan={onOpenExecutionPlan}
        events={[
          {
            event_id: "dag-execution",
            event_type: "dag_execution",
            timestamp: 1000,
            data: {
              phase: "executing",
              steps: [{ id: "1", task: "Create audio", dependencies: [] }],
            },
          },
        ]}
      />,
    )

    fireEvent.click(screen.getByText("chatPage.executionPlan.dagSectionOne:1"))
    expect(onOpenExecutionPlan).toHaveBeenCalledOnce()

    rerender(
      <TraceEventRenderer
        onOpenExecutionPlan={onOpenExecutionPlan}
        events={[
          {
            event_id: "dag-execution",
            event_type: "dag_execution",
            timestamp: 1000,
            data: {
              phase: "executing",
              steps: [
                { id: "1", task: "Create audio", dependencies: [] },
                { id: "2", task: "Edit audio", dependencies: ["1"] },
              ],
            },
          },
        ]}
      />,
    )

    expect(screen.getByText("chatPage.executionPlan.dagSectionOther:2")).toBeInTheDocument()
  })

  it("stops running process spinners when the parent task has failed", () => {
    const { container } = render(
      <TraceEventRenderer
        taskStatus="failed"
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "step-1",
            timestamp: 1000,
            data: {},
          },
          {
            event_id: "llm-start",
            event_type: "llm_call_start",
            step_id: "step-1",
            timestamp: 2000,
            data: { model_name: "gpt-test" },
          },
          {
            event_id: "llm-failed",
            event_type: "llm_call_failed",
            step_id: "step-1",
            timestamp: 3000,
            data: { error: "OpenAI bad request" },
          },
        ]}
      />,
    )

    expect(container.querySelector(".animate-spin")).toBeNull()
  })

  it("infers a failed process from terminal trace errors when task status is unavailable", () => {
    const { container } = render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "step-1",
            timestamp: 1000,
            data: {},
          },
          {
            event_id: "llm-start",
            event_type: "llm_call_start",
            step_id: "step-1",
            timestamp: 2000,
            data: { model_name: "gpt-test" },
          },
          {
            event_id: "trace-error",
            event_type: "trace_error",
            step_id: "step-1",
            timestamp: 3000,
            data: {
              error_type: "agent_error",
              status: "failed",
              error_message: "All patterns failed",
            },
          },
        ]}
      />,
    )

    expect(container.querySelector(".animate-spin")).toBeNull()
  })

  it("stops spinning when a step-local trace error has no explicit status", () => {
    const { container } = render(
      <TraceEventRenderer
        events={[
          {
            event_id: "start",
            event_type: "react_task_start",
            step_id: "react-step",
            timestamp: 1000,
            data: {},
          },
          {
            event_id: "llm-start",
            event_type: "llm_call_start",
            step_id: "react-step",
            timestamp: 2000,
            data: { model_name: "gpt-test" },
          },
          {
            event_id: "trace-error",
            event_type: "trace_error",
            step_id: "react-step",
            timestamp: 3000,
            data: {
              error_type: "agent_pattern_error",
              error_message: "OpenAI bad request",
            },
          },
        ]}
      />,
    )

    expect(container.querySelector(".animate-spin")).toBeNull()
    expect(container).toHaveTextContent("OpenAI bad request")
  })

  it("renders workforce delegation trace events as a dedicated step", () => {
    const onAgentExecutionClick = vi.fn()
    render(
      <TraceEventRenderer
        events={[
          {
            event_id: "delegation-start",
            event_type: "workforce_delegation_start",
            timestamp: Date.now(),
            data: {
              workforce_run_id: 5,
              worker_member_id: 7,
              worker_task_id: 99,
              worker_alias: "Researcher",
              tool_name: "research_worker",
            },
          },
          {
            event_id: "delegation-end",
            event_type: "workforce_delegation_end",
            timestamp: Date.now(),
            data: {
              worker_task_id: 99,
              output: "Research complete",
            },
          },
        ]}
        onAgentExecutionClick={onAgentExecutionClick}
      />,
    )

    fireEvent.click(screen.getByRole("button", {
      name: "traceEventRenderer.viewAgentExecution",
    }))
    expect(onAgentExecutionClick).toHaveBeenCalledWith(expect.objectContaining({
      workerTaskId: "99",
      agentName: "Researcher",
      workerAlias: "Researcher",
      status: "completed",
    }))

    expect(screen.getByRole("heading", {
      name: /traceEventRenderer.delegateToWorker:Researcher/,
    })).toBeInTheDocument()
    expect(screen.queryByRole("button", {
      name: "traceEventRenderer.showProcess",
    })).not.toBeInTheDocument()
    expect(screen.queryByText(/Research complete/)).not.toBeInTheDocument()
  })

  it("renders each workforce manager summary after its completed Agent call", () => {
    render(
      <TraceEventRenderer
        onAgentExecutionClick={vi.fn()}
        events={[
          {
            event_id: "manager-start",
            event_type: "react_task_start",
            step_id: "manager-react",
            timestamp: 1000,
            data: {},
          },
          {
            event_id: "research-start",
            event_type: "workforce_delegation_start",
            timestamp: 2000,
            data: {
              worker_task_id: "agent_research_run",
              worker_alias: "Researcher",
            },
          },
          {
            event_id: "research-end",
            event_type: "workforce_delegation_end",
            timestamp: 3000,
            data: {
              worker_task_id: "agent_research_run",
              output: '{"status":"complete"}',
            },
          },
          {
            event_id: "research-summary",
            event_type: "agent_progress",
            step_id: "manager-react",
            timestamp: 4000,
            data: {
              message: "## Research Complete\n\nThe evidence is ready for script writing.",
            },
          },
          {
            event_id: "writer-start",
            event_type: "workforce_delegation_start",
            timestamp: 5000,
            data: {
              worker_task_id: "agent_writer_run",
              worker_alias: "Writer",
            },
          },
        ]}
      />,
    )

    expect(screen.getByRole("heading", {
      name: /traceEventRenderer.delegateToWorker:Researcher/,
    })).toBeInTheDocument()
    expect(screen.getByText("Research Complete")).toBeInTheDocument()
    expect(screen.getByText("The evidence is ready for script writing.")).toBeInTheDocument()

    const pageText = document.body.textContent || ""
    expect(pageText.indexOf("traceEventRenderer.delegateToWorker:Researcher"))
      .toBeLessThan(pageText.indexOf("Research Complete"))
    expect(pageText.indexOf("Research Complete"))
      .toBeLessThan(pageText.indexOf("traceEventRenderer.delegateToWorker:Writer"))
  })

  it("can render completed Agent details expanded by default", () => {
    render(
      <TraceEventRenderer
        defaultExpandSteps
        events={[
          {
            event_id: "worker-start",
            event_type: "react_task_start",
            step_id: "react-worker",
            timestamp: Date.now(),
            data: {
              source: "xagent-agent-tool-child",
              worker_task_id: "agent_17_d8306189",
              agent_name: "Video Generation Agent",
            },
          },
          {
            event_id: "worker-end",
            event_type: "react_task_end",
            step_id: "react-worker",
            timestamp: Date.now() + 1,
            data: {
              source: "xagent-agent-tool-child",
              worker_task_id: "agent_17_d8306189",
              agent_name: "Video Generation Agent",
              result: "Pilot scenes generated",
            },
          },
        ]}
        taskStatus="completed"
      />,
    )

    expect(screen.getByRole("button", { name: "Video Generation Agent" })).toHaveAttribute(
      "aria-expanded",
      "true",
    )
    expect(screen.queryByRole("button", {
      name: "traceEventRenderer.hideProcess",
    })).not.toBeInTheDocument()
  })

  it("groups delegated agent internals into the worker execution step", () => {
    const timestamp = Date.now()
    render(
      <TraceEventRenderer
        events={[
          {
            event_id: "delegation-start",
            event_type: "workforce_delegation_start",
            timestamp,
            data: {
              worker_task_id: "agent_17_run",
              worker_alias: "Video Generation Agent",
              tool_name: "agent_17",
            },
          },
          {
            event_id: "child-react-start",
            event_type: "react_task_start",
            step_id: "react-child",
            timestamp: timestamp + 1,
            data: {
              source: "xagent-agent-tool-child",
              worker_task_id: "agent_17_run",
              agent_name: "Video Generation Agent",
            },
          },
          {
            event_id: "child-tool-start",
            event_type: "tool_execution_start",
            step_id: "react-child",
            timestamp: timestamp + 2,
            data: {
              source: "xagent-agent-tool-child",
              worker_task_id: "agent_17_run",
              agent_name: "Video Generation Agent",
              tool_name: "generate_video",
              tool_call_id: "video-call-1",
              tool_params: { seconds: "4" },
            },
          },
          {
            event_id: "child-tool-failed",
            event_type: "tool_execution_failed",
            step_id: "react-child",
            timestamp: timestamp + 3,
            data: {
              source: "xagent-agent-tool-child",
              worker_task_id: "agent_17_run",
              agent_name: "Video Generation Agent",
              tool_name: "generate_video",
              tool_call_id: "video-call-1",
              error: "Invalid duration",
            },
          },
        ]}
      />,
    )

    const workerToggle = screen
      .getAllByRole("button", { name: /Video Generation Agent/ })
      .find((button) => button.hasAttribute("aria-expanded"))
    expect(workerToggle).toBeDefined()
    if (workerToggle?.getAttribute("aria-expanded") === "false") {
      fireEvent.click(workerToggle)
    }

    const toolToggle = screen.getByRole("button", { name: /Generate Video/ })
    fireEvent.click(toolToggle)
    expect(screen.getByText("Invalid duration")).toBeInTheDocument()
  })

  it("renders workforce delegation failures as errors", () => {
    render(
      <TraceEventRenderer
        events={[
          {
            event_id: "delegation-start",
            event_type: "workforce_delegation_start",
            timestamp: Date.now(),
            data: {
              worker_task_id: 99,
              worker_alias: "Researcher",
            },
          },
          {
            event_id: "delegation-error",
            event_type: "workforce_delegation_error",
            timestamp: Date.now(),
            data: {
              worker_task_id: 99,
              error: "Worker timed out",
            },
          },
        ]}
      />,
    )

    expect(screen.getByRole("button", {
      name: /traceEventRenderer.delegateToWorker:Researcher/,
    })).toBeInTheDocument()

    const errorToggle = screen.getByRole("button", {
      name: /traceEventRenderer.workerFailed/,
    })
    fireEvent.click(errorToggle)

    expect(screen.getByText("Worker timed out")).toBeInTheDocument()
  })
})
