/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const clipboardWriteTextMock = vi.hoisted(() => vi.fn())
const appContextMock = vi.hoisted(() => ({
  filesDisabled: false,
  openFilePreview: vi.fn(),
}))
const clarificationFormMock = vi.hoisted(() => vi.fn())

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => appContextMock,
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
    tDynamic: (key: string) => key,
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

vi.mock("@/lib/task-runtime-ui-extension", async () => {
  const actual = await vi.importActual<
    typeof import("@/lib/task-runtime-ui-extension")
  >("@/lib/task-runtime-ui-extension")
  const ReactModule = await vi.importActual<typeof import("react")>("react")
  return {
    ...actual,
    TaskRuntimeMessageMetadataExtension: ({ bindings }: { bindings: string[] }) => (
      bindings.includes("browser_relay")
        ? ReactModule.createElement(
          "div",
          { role: "note", "aria-label": "Computer use · My browser" },
          "My browser",
        )
        : null
    ),
  }
})

vi.mock("@/components/file/docx-preview-renderer", () => ({
  DocxPreviewRenderer: ({ base64Content }: { base64Content: string }) => (
    <div data-testid="docx-preview">{base64Content}</div>
  ),
}))

vi.mock("./TraceEventRenderer", async () => {
  // Keep the real getFriendlyToolName (imported by ChatMessage) so the
  // active-tool-name status-line tests exercise the actual implementation;
  // only the heavy TraceEventRenderer component itself is stubbed out.
  const actual = await vi.importActual<typeof import("./TraceEventRenderer")>(
    "./TraceEventRenderer",
  )
  return {
    ...actual,
    TraceEventRenderer: () => <div data-testid="trace-renderer" />,
  }
})

vi.mock("./clarification-form", () => ({
  ClarificationForm: (props: unknown) => {
    clarificationFormMock(props)
    return null
  },
}))

import { ChatMessage } from "./ChatMessage"

describe("ChatMessage Session file capability", () => {
  beforeEach(() => {
    appContextMock.filesDisabled = false
    appContextMock.openFilePreview.mockReset()
    clarificationFormMock.mockReset()
    apiRequestMock.mockReset()
    clipboardWriteTextMock.mockReset()
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: clipboardWriteTextMock },
    })
    vi.stubGlobal("ResizeObserver", class {
      observe() {}
      disconnect() {}
    })
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
      callback(0)
      return 1
    })
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => undefined)
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("renders assistant file markdown as inert text without preview egress", () => {
    appContextMock.filesDisabled = true

    const { container } = render(
      <ChatMessage
        role="assistant"
        content={[
          "[assistant report.docx](file:assistant-doc-id)",
          "![assistant image](file:output/assistant.png)",
        ].join("\n\n")}
      />,
    )

    expect(screen.getByText("assistant report.docx")).not.toHaveAttribute("href")
    expect(screen.getByText("assistant image")).not.toHaveAttribute("src")
    expect(screen.queryByRole("link")).not.toBeInTheDocument()
    expect(screen.queryByRole("img")).not.toBeInTheDocument()
    expect(screen.queryByTestId("docx-preview")).not.toBeInTheDocument()
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(container.innerHTML).not.toContain("assistant-doc-id")
    expect(container.innerHTML).not.toContain("output/assistant.png")

    fireEvent.click(screen.getByText("assistant report.docx"))
    fireEvent.click(screen.getByText("assistant image"))
    expect(appContextMock.openFilePreview).not.toHaveBeenCalled()
  })

  it("passes the resolved file capability to clarification interactions", () => {
    appContextMock.filesDisabled = true

    render(
      <ChatMessage
        role="assistant"
        content="Clarification required"
        interactions={[{ type: "file_upload", field: "evidence", label: "Evidence" }]}
      />,
    )

    expect(clarificationFormMock).toHaveBeenCalledWith(
      expect.objectContaining({ filesDisabled: true }),
    )
  })

  it("replaces the raw backend pause text with a localized message when every interaction is connect_apps", () => {
    render(
      <ChatMessage
        role="assistant"
        content="I need access to Gmail to continue. Please connect it below, then let me know once you have."
        interactions={[{ type: "connect_apps", field: "connect_apps", apps: ["Gmail"] }]}
      />,
    )

    expect(
      screen.getByText("chatPage.clarification.connectApps.needAccess"),
    ).toBeInTheDocument()
    expect(
      screen.queryByText(/I need access to Gmail to continue\. Please connect/),
    ).not.toBeInTheDocument()
  })

  it("keeps the raw backend content when the interaction list mixes connect_apps with another type", () => {
    render(
      <ChatMessage
        role="assistant"
        content="Which app should I use?"
        interactions={[
          { type: "connect_apps", field: "connect_apps", apps: ["Gmail"] },
          { type: "select", field: "app_choice", label: "App" },
        ]}
      />,
    )

    expect(screen.getByText("Which app should I use?")).toBeInTheDocument()
    expect(
      screen.queryByText("chatPage.clarification.connectApps.needAccess"),
    ).not.toBeInTheDocument()
  })

  it("renders Computer use context on a user message", () => {
    render(
      <ChatMessage
        role="user"
        content="Open the selected page"
        contextBadges={[{
          kind: "computer_use",
          label: "Computer use",
          detail: "Local browser",
        }]}
      />,
    )

    expect(
      screen.getByRole("note", { name: "Computer use · Local browser" }),
    ).toBeInTheDocument()
  })

  it("renders distribution message metadata without replacing the message", () => {
    render(
      <ChatMessage
        role="user"
        content="Inspect my browser"
        taskRuntimeExtensionMetadata={{
          bindings: ["browser_relay"],
          publicMetadata: {
            browser_relay: { kind: "browser_relay", connected: true },
          },
        }}
      />,
    )

    expect(screen.getByText("Inspect my browser")).toBeInTheDocument()
    expect(
      screen.getByRole("note", { name: "Computer use · My browser" }),
    ).toBeInTheDocument()
  })

  it("does not render distribution message metadata on assistant messages", () => {
    render(
      <ChatMessage
        role="assistant"
        content="I inspected your browser"
        taskRuntimeExtensionMetadata={{
          bindings: ["browser_relay"],
          publicMetadata: {
            browser_relay: { kind: "browser_relay", connected: true },
          },
        }}
      />,
    )

    expect(screen.getByText("I inspected your browser")).toBeInTheDocument()
    expect(
      screen.queryByRole("note", { name: "Computer use · My browser" }),
    ).not.toBeInTheDocument()
  })

  it("uses the same safe projection for normal assistant DOM and copied content", () => {
    appContextMock.filesDisabled = true
    const { container } = render(
      <ChatMessage
        role="assistant"
        content="Saved /private/reports/secret.txt from /app/src, /opt/xagent, /var/tmp, /srv/app, and /etc/passwd; keep /v1/openapi.json and /care/export.csv."
      />,
    )

    expect(container).toHaveTextContent(
      "Saved secret.txt from src, xagent, tmp, app, and passwd; keep /v1/openapi.json and /care/export.csv.",
    )
    expect(container.innerHTML).not.toContain("/private/reports/secret.txt")
    expect(container.innerHTML).not.toContain("/app/src")
    expect(container.innerHTML).not.toContain("/opt/xagent")
    expect(container.innerHTML).not.toContain("/var/tmp")
    expect(container.innerHTML).not.toContain("/srv/app")
    expect(container.innerHTML).not.toContain("/etc/passwd")
    fireEvent.click(screen.getByTitle("common.copy"))
    expect(clipboardWriteTextMock).toHaveBeenLastCalledWith(
      "Saved secret.txt from src, xagent, tmp, app, and passwd; keep /v1/openapi.json and /care/export.csv.",
    )
  })

  it("uses the parser-complete file projection for assistant DOM and clipboard", () => {
    appContextMock.filesDisabled = true
    const content = [
      "[reference label][artifact]",
      "",
      "[artifact]: file:clipboard-reference-secret",
      "",
      "[balanced](file:tenant(private)/clipboard-balanced-secret)",
      "",
      "Bare file:clipboard-bare(secret)/clipboard-secret-id, file:clipboard-bare-secret, and <file:clipboard-autolink-secret>.",
    ].join("\n")
    const { container } = render(<ChatMessage role="assistant" content={content} />)

    expect(screen.queryByRole("link")).not.toBeInTheDocument()
    expect(container).toHaveTextContent("reference label")
    expect(container).toHaveTextContent("balanced")
    expect(container).toHaveTextContent("Bare file, file, and file.")
    expect(container.innerHTML).not.toContain("clipboard-reference-secret")
    expect(container.innerHTML).not.toContain("clipboard-balanced-secret")
    expect(container.innerHTML).not.toContain("clipboard-bare-secret")
    expect(container.innerHTML).not.toContain("clipboard-secret-id")
    expect(container.innerHTML).not.toContain("clipboard-autolink-secret")

    fireEvent.click(screen.getByTitle("common.copy"))
    const copied = clipboardWriteTextMock.mock.calls.at(-1)?.[0]
    expect(copied).toContain("reference label")
    expect(copied).toContain("balanced")
    expect(copied).toContain("Bare file, file, and file.")
    expect(copied).not.toContain("clipboard-reference-secret")
    expect(copied).not.toContain("clipboard-balanced-secret")
    expect(copied).not.toContain("clipboard-bare-secret")
    expect(copied).not.toContain("clipboard-secret-id")
    expect(copied).not.toContain("clipboard-autolink-secret")
  })

  it("uses semantic Markdown text for entity-encoded DOM and clipboard projection", () => {
    appContextMock.filesDisabled = true
    const { container } = render(
      <ChatMessage
        role="assistant"
        content="Saved /private&#47;tenant&#47;entity-secret.txt and file&#58;entity-file-secret."
      />,
    )

    expect(container).toHaveTextContent("Saved entity-secret.txt and file.")
    expect(container.innerHTML).not.toContain("/private")
    expect(container.innerHTML).not.toContain("entity-file-secret")

    fireEvent.click(screen.getByTitle("common.copy"))
    expect(clipboardWriteTextMock).toHaveBeenLastCalledWith(
      "Saved entity-secret.txt and file.",
    )
  })

  it("renders user file-chip syntax as plain non-clickable text when files are disabled", () => {
    appContextMock.filesDisabled = true

    const { container } = render(
      <ChatMessage
        role="user"
        content={"Uploaded [report.txt](file:user-file-id) and `output/notes.txt`"}
      />,
    )

    expect(screen.getByText("Uploaded report.txt and notes.txt")).toBeInTheDocument()
    expect(container.innerHTML).not.toContain("user-file-id")
    expect(container.innerHTML).not.toContain("output/notes.txt")
    expect(apiRequestMock).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText("Uploaded report.txt and notes.txt"))
    expect(appContextMock.openFilePreview).not.toHaveBeenCalled()
  })

  it("preserves legacy assistant file preview behavior when files are enabled", () => {
    render(
      <ChatMessage
        role="assistant"
        content="[legacy archive.zip](file:legacy-file-id)"
      />,
    )

    fireEvent.click(screen.getByText("legacy archive.zip"))
    expect(appContextMock.openFilePreview).toHaveBeenCalledWith(
      "legacy-file-id",
      "legacy archive.zip",
      [{ fileName: "legacy archive.zip", fileId: "legacy-file-id" }],
    )
  })

  it("copies the files-disabled JSON representation instead of file metadata", () => {
    appContextMock.filesDisabled = true
    const content = JSON.stringify({
      artifact: {
        file_id: "clipboard-file-id",
        file_path: "/private/clipboard.pdf",
        download_url: "https://files.example/download/clipboard",
        url: "https://files.example/generic-clipboard-url",
        filename: "clipboard.pdf",
        mime_type: "application/pdf",
        text: "[open clipboard](file:clipboard-file-id)",
      },
      requestUrl: "https://api.example/tasks/42",
    })

    render(<ChatMessage role="assistant" content={content} />)

    fireEvent.click(screen.getByTitle("common.copy"))

    expect(clipboardWriteTextMock).toHaveBeenCalledTimes(1)
    const copied = clipboardWriteTextMock.mock.calls[0][0]
    expect(copied).not.toContain("clipboard-file-id")
    expect(copied).not.toContain("/private/clipboard.pdf")
    expect(copied).not.toContain("https://files.example/download/clipboard")
    expect(copied).not.toContain("https://files.example/generic-clipboard-url")
    expect(JSON.parse(copied)).toEqual({
      artifact: {
        filename: "clipboard.pdf",
        mime_type: "application/pdf",
        text: "open clipboard",
      },
      requestUrl: "https://api.example/tasks/42",
    })
  })

  it("sanitizes failed message display and copy content only when files are disabled", () => {
    const failureText = "Failed to read [secret-report.pdf](file:failed-file-id) at `/private/run/secret-report.pdf`"
    appContextMock.filesDisabled = true

    const { container, rerender } = render(
      <ChatMessage
        role="assistant"
        content={failureText}
        processStatus="failed"
      />,
    )

    expect(container).not.toHaveTextContent("failed-file-id")
    expect(container).not.toHaveTextContent("/private/run/secret-report.pdf")
    expect(container).toHaveTextContent(
      "Failed to read secret-report.pdf at secret-report.pdf",
    )
    fireEvent.click(screen.getByTitle("common.copy"))
    expect(clipboardWriteTextMock).toHaveBeenCalledWith(
      "Failed to read secret-report.pdf at secret-report.pdf",
    )

    appContextMock.filesDisabled = false
    rerender(
      <ChatMessage
        role="assistant"
        content={failureText}
        processStatus="failed"
      />,
    )

    expect(container).toHaveTextContent(failureText)
    fireEvent.click(screen.getByTitle("common.copy"))
    expect(clipboardWriteTextMock).toHaveBeenLastCalledWith(failureText)
  })

  it("sanitizes trace-derived failed message text when files are disabled", () => {
    const traceFailure = "Unable to open [trace-secret.csv](file:trace-file-id) from `/private/run/trace-secret.csv`"
    appContextMock.filesDisabled = true

    const { container } = render(
      <ChatMessage
        role="assistant"
        content={null}
        processStatus="failed"
        traceEvents={[{
          event_type: "task_failed",
          data: { error: traceFailure },
        }]}
      />,
    )

    expect(container).not.toHaveTextContent("trace-file-id")
    expect(container).not.toHaveTextContent("/private/run/trace-secret.csv")
    expect(container).toHaveTextContent(
      "Unable to open trace-secret.csv from trace-secret.csv",
    )
    fireEvent.click(screen.getByTitle("common.copy"))
    expect(clipboardWriteTextMock).toHaveBeenCalledWith(
      "Unable to open trace-secret.csv from trace-secret.csv",
    )
    expect(clipboardWriteTextMock).not.toHaveBeenCalledWith(traceFailure)
  })

  it("copies the resolved failure instead of mismatched raw content", () => {
    const traceFailure = "Unable to open [trace-secret.csv](file:trace-file-id) from `/private/run/trace-secret.csv`"
    const rawContent = "Ignored [raw-secret.csv](file:raw-file-id) from `/private/raw-secret.csv`"
    appContextMock.filesDisabled = true

    render(
      <ChatMessage
        role="assistant"
        content={null}
        rawContent={rawContent}
        processStatus="failed"
        traceEvents={[{
          event_type: "task_failed",
          data: { error: traceFailure },
        }]}
      />,
    )

    fireEvent.click(screen.getByTitle("common.copy"))
    expect(clipboardWriteTextMock).toHaveBeenCalledWith(
      "Unable to open trace-secret.csv from trace-secret.csv",
    )
    const copied = clipboardWriteTextMock.mock.calls[0][0]
    expect(copied).not.toContain("raw-file-id")
    expect(copied).not.toContain("/private/raw-secret.csv")
  })

  it("names the active tool in the status line while it is running", () => {
    render(
      <ChatMessage
        role="assistant"
        content={null}
        taskStatus="running"
        traceEvents={[
          { event_type: "tool_execution_start", data: { tool_name: "web_search" } },
        ]}
      />,
    )

    expect(screen.getByText(/traceEventRenderer\.toolNames\.web_search/)).toBeInTheDocument()
  })

  it("falls back to the generic done label once the tool has finished, instead of still naming it", () => {
    render(
      <ChatMessage
        role="assistant"
        content={null}
        taskStatus="running"
        traceEvents={[
          { event_type: "tool_execution_end", data: { tool_name: "web_search" } },
        ]}
      />,
    )

    expect(
      screen.queryByText(/traceEventRenderer\.toolNames\.web_search/),
    ).not.toBeInTheDocument()
    expect(screen.getByText(/agent\.logs\.event\.actions\.tool_execution_end/)).toBeInTheDocument()
  })

  it("does not throw when traceEvents contains null, primitive, or malformed entries", () => {
    // traceEvents comes off an external WS/API payload; mirrors the
    // malformed-input coverage in TraceEventRenderer.test.tsx.
    expect(() =>
      render(
        <ChatMessage
          role="assistant"
          content={null}
          taskStatus="running"
          traceEvents={[
            null,
            42,
            "not-an-event",
            { event_type: 17, data: "not-an-object" },
            {
              event_type: "agent_progress",
              data: { message: "Still valid despite the noise around it." },
            },
          ] as unknown as React.ComponentProps<typeof ChatMessage>["traceEvents"]}
        />,
      ),
    ).not.toThrow()
    expect(
      screen.getByText(/Still valid despite the noise around it\./),
    ).toBeInTheDocument()
  })

  it("surfaces the model's own progress narration in the status line, not a generic action label", () => {
    render(
      <ChatMessage
        role="assistant"
        content={null}
        taskStatus="running"
        traceEvents={[
          { event_type: "tool_execution_start", data: { tool_name: "web_search" } },
          { event_type: "tool_execution_end", data: { tool_name: "web_search" } },
          {
            event_type: "agent_progress",
            data: { message: "Looking into LangChain and CrewAI pricing now." },
          },
        ]}
      />,
    )

    expect(
      screen.getByText(/Looking into LangChain and CrewAI pricing now\./),
    ).toBeInTheDocument()
    expect(screen.queryByText(/agent\.logs\.event\.actions/)).not.toBeInTheDocument()
  })

  it("keeps showing the last narration even after a newer, un-narrated tool call", () => {
    render(
      <ChatMessage
        role="assistant"
        content={null}
        taskStatus="running"
        traceEvents={[
          {
            event_type: "agent_progress",
            data: { message: "Checking pricing pages first." },
          },
          { event_type: "tool_execution_start", data: { tool_name: "browser_use" } },
        ]}
      />,
    )

    // The narration is still accurate — a later tool call with no message
    // of its own shouldn't bump it back to a generic tool-name label.
    expect(screen.getByText(/Checking pricing pages first\./)).toBeInTheDocument()
    expect(
      screen.queryByText(/traceEventRenderer\.toolNames\.browser_use/),
    ).not.toBeInTheDocument()
  })

  it("expires a narration once a second later tool call starts, instead of staying stale for the rest of the run", () => {
    render(
      <ChatMessage
        role="assistant"
        content={null}
        taskStatus="running"
        traceEvents={[
          {
            event_type: "agent_progress",
            data: { message: "Checking pricing pages first." },
          },
          { event_type: "tool_execution_start", data: { tool_name: "browser_use" } },
          { event_type: "tool_execution_end", data: { tool_name: "browser_use" } },
          { event_type: "tool_execution_start", data: { tool_name: "write_file" } },
        ]}
      />,
    )

    // Two tool calls have started since the narration — it no longer
    // describes what's happening now, so the generic label takes over
    // instead of naming a step the trace has long since moved past.
    expect(screen.queryByText(/Checking pricing pages first\./)).not.toBeInTheDocument()
  })

  it("shows the terminal 'Done' label once the turn has completed, not a leftover narration", () => {
    // A completed turn with no answer text (an empty final response) still
    // renders GeneratingIndicator — without gating on process status, the
    // narration priority would keep naming a step forever since narration
    // has no other way to learn the turn is over.
    render(
      <ChatMessage
        role="assistant"
        content={null}
        processStatus="completed"
        traceEvents={[
          {
            event_type: "agent_progress",
            data: { message: "Checking pricing pages first." },
          },
          { event_type: "task_completion", data: {} },
        ]}
      />,
    )

    expect(screen.queryByText(/Checking pricing pages first\./)).not.toBeInTheDocument()
    expect(screen.getByText(/agent\.logs\.event\.actions\.task_completion/)).toBeInTheDocument()
  })

  it("shows the most recent narration when there are several, not an earlier one", () => {
    // Pins the backward-scan direction in latestProgressNarration — a
    // regression that scanned forward instead would still pass every other
    // narration test here (each of them only has one narration).
    render(
      <ChatMessage
        role="assistant"
        content={null}
        taskStatus="running"
        traceEvents={[
          { event_type: "agent_progress", data: { message: "First: scoping the task." } },
          {
            event_type: "agent_progress",
            data: { message: "Second: now writing the file." },
          },
        ]}
      />,
    )

    expect(screen.getByText(/Second: now writing the file\./)).toBeInTheDocument()
    expect(screen.queryByText(/First: scoping the task\./)).not.toBeInTheDocument()
  })

  it("excludes a question from narration by message_type alone, without expect_response set", () => {
    // The exclusion is `expect_response !== true && message_type !== 'question'`
    // — two independent disjuncts. The other test for this only sets
    // expect_response; this one isolates message_type so deleting either
    // half of the exclusion would fail a test.
    render(
      <ChatMessage
        role="assistant"
        content={null}
        taskStatus="running"
        traceEvents={[
          {
            event_type: "agent_message",
            data: { message: "Which region should I search?", message_type: "question" },
          },
        ]}
      />,
    )

    expect(screen.queryByText(/Which region should I search\?/)).not.toBeInTheDocument()
  })

  it("does not throw when the last trace event is malformed and no narration short-circuits first", () => {
    // The existing malformed-entries test ends with a valid narration, so
    // latestProgressNarration() short-circuits the `||` before
    // getEventTitle ever runs on the malformed tail — its own guards go
    // unexercised. Drop the narration so this one actually reaches them.
    expect(() =>
      render(
        <ChatMessage
          role="assistant"
          content={null}
          taskStatus="running"
          traceEvents={[
            null,
            42,
            "not-an-event",
            { event_type: 17, data: "not-an-object" },
          ] as unknown as React.ComponentProps<typeof ChatMessage>["traceEvents"]}
        />,
      ),
    ).not.toThrow()
  })

  it("does not treat a pending question as narration in the status line", () => {
    // expect_response=true means this agent_message is a question awaiting
    // the user's answer, not a progress update — it must not surface in the
    // status line the way real narration would.
    render(
      <ChatMessage
        role="assistant"
        content={null}
        taskStatus="running"
        traceEvents={[
          {
            event_type: "agent_message",
            data: { message: "Which region should I search?", expect_response: true },
          },
        ]}
      />,
    )

    expect(screen.queryByText(/Which region should I search\?/)).not.toBeInTheDocument()
  })
})
