/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) => {
      if (vars?.tool) return `${key}:${vars.tool}`
      return key
    },
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    openFilePreview: vi.fn(),
    dispatch: vi.fn(),
  }),
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
  PptxPreviewRenderer: ({ base64Content }: { base64Content: string }) => (
    <div data-testid="pptx-preview">{base64Content}</div>
  ),
}))

import { TraceEventRenderer } from "./TraceEventRenderer"

describe("TraceEventRenderer", () => {
  afterEach(() => {
    cleanup()
    apiRequestMock.mockReset()
    vi.restoreAllMocks()
  })

  it("renders image artifacts inline from tool results", async () => {
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

    fireEvent.click(screen.getByRole("button"))

    const image = screen.getByAltText("generated_image.png")
    expect(image).toHaveAttribute(
      "src",
      "http://api.local/api/files/public/preview/582e7b79-4de9-4905-b73b-7d5a70ad64fe",
    )
  })

  it("renders pptx artifacts inline through PptxPreviewRenderer", async () => {
    // Arbitrary sentinel bytes; base64 encodes to "AQI=". See
    // inline-file-preview.test.tsx for why we avoid "PK" here.
    apiRequestMock.mockResolvedValue({
      ok: true,
      arrayBuffer: async () => new Uint8Array([0x01, 0x02]).buffer,
    })

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

    fireEvent.click(screen.getByRole("button"))

    // Browsers can't render raw .pptx in an iframe, and the backend's
    // /api/files/public/preview endpoint now returns the raw bytes, so
    // we mirror the docx/xlsx path: fetch + base64 + canvas render
    // via pptxviewjs.
    expect(await screen.findByTestId("pptx-preview")).toHaveTextContent("AQI=")
    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/files/public/preview/slides-file-id",
      expect.objectContaining({ cache: "no-cache" }),
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

    fireEvent.click(screen.getByRole("button"))

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

    fireEvent.click(screen.getByRole("button"))

    expect(await screen.findByTestId("excel-preview")).toHaveTextContent("WFk=")
  })
})
