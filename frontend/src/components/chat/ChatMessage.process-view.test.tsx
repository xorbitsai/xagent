import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
    tDynamic: (_key: string, fallback: string) => fallback,
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({ openFilePreview: vi.fn() }),
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("./TraceEventRenderer", () => ({
  TraceEventRenderer: () => <div data-testid="trace-renderer" />,
}))

vi.mock("@/components/ui/markdown-renderer", () => ({
  MarkdownRenderer: ({ content }: { content: string }) => <div>{content}</div>,
}))

vi.mock("./clarification-form", () => ({
  ClarificationForm: () => <div data-testid="clarification-form" />,
}))

import { ChatMessage } from "./ChatMessage"

// A tool call is the worst case for the widget: its arguments and output are
// the raw payload the trace renderer would print.
const TRACE_EVENTS = [
  {
    event_id: "tool-1",
    event_type: "tool_call",
    timestamp: 1000,
    data: { tool_name: "web_search", args: { query: "secret" } },
  },
] as any

const RAW_ERROR = "Traceback: KeyError('api_key') in web_search"

const FAILED_TRACE_EVENTS = [
  ...TRACE_EVENTS,
  {
    event_id: "err-1",
    event_type: "trace_error",
    timestamp: 2000,
    data: { error: RAW_ERROR },
  },
] as any

afterEach(() => {
  cleanup()
})

describe("ChatMessage process view", () => {
  it("renders the trace for internal pages", () => {
    render(
      <ChatMessage
        role="assistant"
        content="Here is the answer"
        traceEvents={TRACE_EVENTS}
        showProcessView={true}
      />
    )

    expect(screen.getByTestId("trace-renderer")).toBeTruthy()
    expect(screen.getByText("Here is the answer")).toBeTruthy()
  })

  it("renders the answer without the trace when the process view is off", () => {
    render(
      <ChatMessage
        role="assistant"
        content="Here is the answer"
        traceEvents={TRACE_EVENTS}
        showProcessView={false}
      />
    )

    expect(screen.queryByTestId("trace-renderer")).toBeNull()
    expect(screen.getByText("Here is the answer")).toBeTruthy()
  })

  it("drops a trace-only turn instead of leaving an empty bubble", () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={TRACE_EVENTS}
        showProcessView={false}
        showEmptyStatus={false}
      />
    )

    expect(screen.queryByTestId("trace-renderer")).toBeNull()
    // The timeline separates children with space-y-*, so a childless wrapper
    // would still take up its gap — nothing at all must be rendered.
    expect(container.firstChild).toBeNull()
  })

  it("drops it even when the turn carries copyable rawContent", () => {
    // rawContent alone must not resurrect the wrapper: with both the trace and
    // the bubble hidden, all that survives is a floating copy button.
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={null}
        rawContent="internal draft text"
        traceEvents={TRACE_EVENTS}
        showProcessView={false}
        showEmptyStatus={false}
      />
    )

    expect(container.firstChild).toBeNull()
  })

  it("keeps a generic status line while the answer is still streaming", () => {
    render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={TRACE_EVENTS}
        showProcessView={false}
        showEmptyStatus={true}
        taskStatus="running"
      />
    )

    // The internal step title ("calling web_search") is part of the trace, so
    // the hidden-trace status line must fall back to the neutral wording.
    expect(screen.getByText("common.thinking")).toBeTruthy()
    expect(screen.queryByText(/web_search/)).toBeNull()
  })

  it("names the running step when the process view is on", () => {
    render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={TRACE_EVENTS}
        showProcessView={true}
        showEmptyStatus={true}
        taskStatus="running"
      />
    )

    expect(screen.getByText(/tool_call/)).toBeTruthy()
    expect(screen.queryByText("common.thinking")).toBeNull()
  })

  it("does not sit on a thinking line once the turn is done", () => {
    render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={TRACE_EVENTS}
        showProcessView={false}
        showEmptyStatus={true}
        taskStatus="completed"
      />
    )

    expect(screen.getByText("common.statusDone")).toBeTruthy()
    expect(screen.queryByText("common.thinking")).toBeNull()
  })

  it("renders the trace alone for an internal trace-only turn", () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={TRACE_EVENTS}
        showProcessView={true}
        showEmptyStatus={false}
      />
    )

    expect(screen.getByTestId("trace-renderer")).toBeTruthy()
    // No bubble: the avatar (the only svg here, TraceEventRenderer is mocked)
    // must not render below the trace.
    expect(container.querySelector("svg")).toBeNull()
  })
})

describe("ChatMessage stopped turns", () => {
  it.each([
    ["paused", "common.taskPaused"],
    ["waiting_for_user", "common.waitingForUser"],
  ])("keeps a past %s turn visible when the trace is hidden", (status, statusText) => {
    // Like the past-failed case: the panel marks a superseded trace group
    // showEmptyStatus=false, and without the trace only the status line is
    // left to show the turn ever ran.
    render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={TRACE_EVENTS}
        showProcessView={false}
        showEmptyStatus={false}
        processStatus={status}
      />
    )

    expect(screen.getByText(statusText)).toBeTruthy()
    expect(screen.queryByTestId("trace-renderer")).toBeNull()
    expect(screen.queryByText(/web_search/)).toBeNull()
  })

  it("still drops a paused trace-only turn when the trace itself shows", () => {
    const { container } = render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={TRACE_EVENTS}
        showProcessView={true}
        showEmptyStatus={false}
        processStatus="paused"
      />
    )

    expect(screen.getByTestId("trace-renderer")).toBeTruthy()
    expect(container.querySelector("svg")).toBeNull()
    expect(screen.queryByText("common.taskPaused")).toBeNull()
  })
})

describe("ChatMessage failures", () => {
  it("shows the backend error on internal pages", () => {
    render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={FAILED_TRACE_EVENTS}
        showProcessView={true}
        showEmptyStatus={true}
        taskStatus="failed"
      />
    )

    expect(screen.getByText(RAW_ERROR)).toBeTruthy()
  })

  it("replaces the backend error with a generic line when the trace is hidden", () => {
    render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={FAILED_TRACE_EVENTS}
        showProcessView={false}
        showEmptyStatus={true}
        taskStatus="failed"
      />
    )

    expect(screen.queryByText(RAW_ERROR)).toBeNull()
    expect(screen.getByText("common.errors.taskFailed")).toBeTruthy()
  })

  it("keeps a past failed turn visible when the trace is hidden", () => {
    // Its trace group is no longer the latest, so the panel marks it
    // showEmptyStatus=false. Dropping the bubble as well would leave the
    // visitor's question followed by nothing at all.
    render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={FAILED_TRACE_EVENTS}
        showProcessView={false}
        showEmptyStatus={false}
        processStatus="failed"
      />
    )

    expect(screen.getByText("common.errors.taskFailed")).toBeTruthy()
    expect(screen.queryByText(RAW_ERROR)).toBeNull()
  })

  it("replaces failed-turn content with the generic line when the trace is hidden", () => {
    // content on a failed turn carries the backend's raw failure text
    // (final_answer_error streams str(exc); the terminal handler stores the
    // reason verbatim) — the same redaction as mined trace errors must apply.
    render(
      <ChatMessage
        role="assistant"
        content={RAW_ERROR}
        traceEvents={[]}
        showProcessView={false}
        processStatus="failed"
      />
    )

    expect(screen.queryByText(RAW_ERROR)).toBeNull()
    expect(screen.getByText("common.errors.taskFailed")).toBeTruthy()
  })

  it("copies the redacted text, not the raw error, on a failed hidden-trace turn", () => {
    // The copy button reads copyableContent; redacting only the bubble would
    // still hand the raw backend text to anyone clicking copy.
    const writeText = vi.fn()
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    })

    render(
      <ChatMessage
        role="assistant"
        content={RAW_ERROR}
        traceEvents={[]}
        showProcessView={false}
        processStatus="failed"
      />
    )

    fireEvent.click(screen.getByTitle("common.copy"))
    expect(writeText).toHaveBeenCalledWith("common.errors.taskFailed")
  })

  it("keeps failed-turn content verbatim on internal pages", () => {
    render(
      <ChatMessage
        role="assistant"
        content={RAW_ERROR}
        traceEvents={[]}
        showProcessView={true}
        processStatus="failed"
      />
    )

    expect(screen.getByText(RAW_ERROR)).toBeTruthy()
  })

  it("falls back to a generic unknown error when the failed trace has no error text", () => {
    render(
      <ChatMessage
        role="assistant"
        content={null}
        traceEvents={[
          ...TRACE_EVENTS,
          { event_id: "err-2", event_type: "task_failed", timestamp: 2000, data: {} },
        ]}
        showProcessView={true}
        showEmptyStatus={true}
        taskStatus="failed"
      />
    )

    expect(screen.getByText("common.errors.unknown")).toBeTruthy()
  })
})
