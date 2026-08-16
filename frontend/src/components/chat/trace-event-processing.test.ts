/**
 * Inc.7 (frontend) — tool-event attribution by tool_call_id.
 *
 * With in-turn tool concurrency, a single step can have several same-named tool
 * actions in flight. The processor must attribute each tool_execution_end /
 * _failed to the action with the matching tool_call_id, not to the
 * "last running tool" (which mis-attributes output/status the moment completion
 * order differs from reverse-start order).
 */
import { describe, it, expect, vi } from "vitest"

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))
vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({ openFilePreview: vi.fn(), dispatch: vi.fn() }),
}))
vi.mock("@/lib/api-wrapper", () => ({ apiRequest: vi.fn() }))
// These pull in heavy/optional deps (e.g. pptxviewjs) not needed for the
// pure reducer under test.
vi.mock("@/components/file/docx-preview-renderer", () => ({
  DocxPreviewRenderer: () => null,
}))
vi.mock("@/components/file/excel-preview-renderer", () => ({
  ExcelPreviewRenderer: () => null,
}))
vi.mock("@/components/file/pptx-preview-renderer", () => ({
  PptxPreviewRenderer: () => null,
}))

import {
  getFriendlyToolName,
  getProgressNarrationText,
  getRawToolName,
  processTraceEvents,
} from "./TraceEventRenderer"
import { resolveDynamicTranslation, type Locale } from "@/i18n/translations"
import type { TranslateDynamic } from "@/contexts/i18n-context"
import en from "@/i18n/locales/en"
import zh from "@/i18n/locales/zh"

const t = (key: string, vars?: Record<string, string | number>) =>
  vars?.tool ? `${key}:${vars.tool}` : key

const ev = (event_type: string, data: Record<string, unknown>) => ({
  event_type,
  step_id: "step-1",
  data,
})

const stepStart = ev("dag_step_start", { step_name: "Search" })

describe("processTraceEvents tool_call_id attribution", () => {
  it("normalizes malformed trace data before nested access", () => {
    const malformedEvents = [
      {
        event_type: "workforce_delegation_start",
        step_id: "worker-null",
        timestamp: 1,
        data: null,
      },
      {
        event_type: "workforce_delegation_end",
        step_id: "worker-string",
        timestamp: 2,
        data: "not-an-object",
      },
      {
        event_type: "tool_execution_start",
        step_id: "worker-array",
        timestamp: 3,
        data: [],
      },
    ]

    expect(() => processTraceEvents(malformedEvents as never, t)).not.toThrow()
  })

  it("does not throw when tool_name in the payload is not a string", () => {
    // tool_name's "string" type is a compile-time annotation, not a runtime
    // guarantee: a malformed/legacy backend payload could send a number,
    // object, or null. getFriendlyToolName must not crash on it.
    const malformedToolNameEvents = [
      ev("tool_execution_start", { tool_name: 42 }),
      ev("tool_execution_start", { tool_name: { unexpected: "object" } }),
      ev("tool_execution_start", { tool_name: null }),
    ]

    expect(() => processTraceEvents(malformedToolNameEvents as never, t)).not.toThrow()
  })

  it("attributes concurrent same-name tool results by tool_call_id", () => {
    const events = [
      stepStart,
      ev("tool_execution_start", {
        tool_name: "web_search",
        tool_call_id: "A",
        tool_args: { query: "a" },
      }),
      ev("tool_execution_start", {
        tool_name: "web_search",
        tool_call_id: "B",
        tool_args: { query: "b" },
      }),
      // First-started finishes first: this is the order that breaks LIFO.
      ev("tool_execution_end", {
        tool_name: "web_search",
        tool_call_id: "A",
        result: { output: "RESULT_A" },
      }),
      ev("tool_execution_end", {
        tool_name: "web_search",
        tool_call_id: "B",
        result: { output: "RESULT_B" },
      }),
    ]

    const steps = processTraceEvents(events as never, t)
    const toolActions = steps[0].actions.filter((a) => a.type === "tool")
    const a = toolActions.find((x) => x.data.tool_call_id === "A")
    const b = toolActions.find((x) => x.data.tool_call_id === "B")

    expect(a?.data.output).toBe("RESULT_A")
    expect(b?.data.output).toBe("RESULT_B")
    expect(a?.status).toBe("completed")
    expect(b?.status).toBe("completed")
  })

  it("attributes a concurrent tool failure by tool_call_id", () => {
    const events = [
      stepStart,
      ev("tool_execution_start", {
        tool_name: "web_search",
        tool_call_id: "A",
        tool_args: { query: "a" },
      }),
      ev("tool_execution_start", {
        tool_name: "web_search",
        tool_call_id: "B",
        tool_args: { query: "b" },
      }),
      ev("tool_execution_failed", {
        tool_name: "web_search",
        tool_call_id: "A",
        error: "boom-A",
      }),
      ev("tool_execution_end", {
        tool_name: "web_search",
        tool_call_id: "B",
        result: { output: "RESULT_B" },
      }),
    ]

    const steps = processTraceEvents(events as never, t)
    const toolActions = steps[0].actions.filter((a) => a.type === "tool")
    const a = toolActions.find((x) => x.data.tool_call_id === "A")
    const b = toolActions.find((x) => x.data.tool_call_id === "B")

    expect(a?.status).toBe("failed")
    expect(a?.data.error).toBe("boom-A")
    expect(b?.status).toBe("completed")
    expect(b?.data.output).toBe("RESULT_B")
  })

  it("updates step.output for sequential tools within one step", () => {
    // Two tools run one-after-another (never overlapping). step.output must
    // track the latest tool's output, exactly as it did before concurrency was
    // introduced — counting total tool actions would wrongly freeze it.
    const events = [
      stepStart,
      ev("tool_execution_start", {
        tool_name: "calculator",
        tool_call_id: "A",
        tool_args: { expression: "1+1" },
      }),
      ev("tool_execution_end", {
        tool_name: "calculator",
        tool_call_id: "A",
        result: { output: "RESULT_A" },
      }),
      ev("tool_execution_start", {
        tool_name: "calculator",
        tool_call_id: "B",
        tool_args: { expression: "2+2" },
      }),
      ev("tool_execution_end", {
        tool_name: "calculator",
        tool_call_id: "B",
        result: { output: "RESULT_B" },
      }),
    ]

    const steps = processTraceEvents(events as never, t)
    expect(steps[0].output).toBe("RESULT_B")
  })

  it("does not clobber step.output when tools run concurrently", () => {
    // Both tools are in flight at once, so the step-level scalar is ambiguous;
    // the processor leaves it unset and the per-action outputs carry the data.
    const events = [
      stepStart,
      ev("tool_execution_start", {
        tool_name: "web_search",
        tool_call_id: "A",
        tool_args: { query: "a" },
      }),
      ev("tool_execution_start", {
        tool_name: "web_search",
        tool_call_id: "B",
        tool_args: { query: "b" },
      }),
      ev("tool_execution_end", {
        tool_name: "web_search",
        tool_call_id: "A",
        result: { output: "RESULT_A" },
      }),
      ev("tool_execution_end", {
        tool_name: "web_search",
        tool_call_id: "B",
        result: { output: "RESULT_B" },
      }),
    ]

    const steps = processTraceEvents(events as never, t)
    // step.output is left at its initial value, not clobbered by whichever
    // concurrent tool happened to finish last.
    expect(steps[0].output).not.toBe("RESULT_A")
    expect(steps[0].output).not.toBe("RESULT_B")
    const toolActions = steps[0].actions.filter((a) => a.type === "tool")
    expect(toolActions.find((x) => x.data.tool_call_id === "A")?.data.output).toBe(
      "RESULT_A"
    )
    expect(toolActions.find((x) => x.data.tool_call_id === "B")?.data.output).toBe(
      "RESULT_B"
    )
  })

  it("falls back to last running tool when tool_call_id is absent (legacy)", () => {
    const events = [
      stepStart,
      ev("tool_execution_start", {
        tool_name: "calculator",
        tool_args: { expression: "1+1" },
      }),
      ev("tool_execution_end", {
        tool_name: "calculator",
        result: { output: "2" },
      }),
    ]

    const steps = processTraceEvents(events as never, t)
    const toolActions = steps[0].actions.filter((a) => a.type === "tool")
    expect(toolActions).toHaveLength(1)
    expect(toolActions[0].status).toBe("completed")
    expect(toolActions[0].data.output).toBe("2")
  })
})

describe("getFriendlyToolName", () => {
  it("prettifies an unmapped snake_case tool name", () => {
    expect(getFriendlyToolName("some_future_tool")).toBe("Some Future Tool")
  })

  it("returns empty string for malformed (non-string) input instead of throwing", () => {
    expect(() => getFriendlyToolName(42 as never)).not.toThrow()
    expect(() => getFriendlyToolName({ unexpected: "object" } as never)).not.toThrow()
    expect(() => getFriendlyToolName(null as never)).not.toThrow()
    expect(getFriendlyToolName(42 as never)).toBe("")
    expect(getFriendlyToolName({ unexpected: "object" } as never)).toBe("")
    expect(getFriendlyToolName(null as never)).toBe("")
  })
})

// The mocked useI18n() above (t: key => key) means every other test in this
// file exercises processTraceEvents' own logic, not what the toolNames map
// actually resolves to — deleting the whole map would leave them green. Use
// the real resolver here so a regression in en.ts/zh.ts's toolNames entries,
// or in getFriendlyToolName's lookup path, actually fails a test.
describe("getFriendlyToolName against the real translation trees", () => {
  const realTDynamic = (locale: Locale): TranslateDynamic => (key, fallback) =>
    resolveDynamicTranslation(locale, key, fallback)

  it("resolves a mapped tool name to its curated phrase in both locales", () => {
    expect(getFriendlyToolName("web_search", realTDynamic("en"))).toBe("Searching the web")
    expect(getFriendlyToolName("web_search", realTDynamic("zh"))).toBe("正在搜索网络")
  })

  it("falls back to the prettified raw name for a tool absent from the map", () => {
    expect(getFriendlyToolName("some_future_tool", realTDynamic("en"))).toBe(
      "Some Future Tool",
    )
    expect(getFriendlyToolName("some_future_tool", realTDynamic("zh"))).toBe(
      "Some Future Tool",
    )
  })
})

describe("getRawToolName", () => {
  it("falls through to data.tool_name when response.tool_name is a truthy non-string", () => {
    // response.tool_name and data.tool_name must be checked independently —
    // `a || b` then type-checking the combined result gives up here instead
    // of falling through, because the truthy (but non-string) 42 already
    // won the `||` before either side was ever checked for being a string.
    const event = {
      data: { response: { tool_name: 42 }, tool_name: "web_search" },
    } as never
    expect(getRawToolName(event)).toBe("web_search")
  })

  it("prefers response.tool_name over data.tool_name when both are valid strings", () => {
    const event = {
      data: { response: { tool_name: "fetch_web_content" }, tool_name: "web_search" },
    } as never
    expect(getRawToolName(event)).toBe("fetch_web_content")
  })

  it("returns an empty string when neither field is a string", () => {
    expect(getRawToolName({ data: {} } as never)).toBe("")
    expect(getRawToolName({} as never)).toBe("")
  })
})

describe("getProgressNarrationText", () => {
  it("falls through to content when message is an empty string, via ||", () => {
    // Deliberately || not ?? — an empty message string is falsy under ||
    // (falls through to content) but not under ?? (only null/undefined
    // fall through). A regression back to ?? would silently re-break this.
    const event = { data: { message: "", content: "text" } } as never
    expect(getProgressNarrationText(event)).toBe("text")
  })

  it("prefers message over content when both are present", () => {
    const event = { data: { message: "the message", content: "the content" } } as never
    expect(getProgressNarrationText(event)).toBe("the message")
  })

  it("returns an empty string for a non-string value and for no data at all", () => {
    expect(getProgressNarrationText({ data: { message: 42 } } as never)).toBe("")
    expect(getProgressNarrationText({} as never)).toBe("")
  })
})

describe("every toolNames entry resolves through getFriendlyToolName", () => {
  // The single-key test above only pins web_search; the other 55 entries
  // per locale were unpinned — a typo'd value, or a key present in one
  // locale's source object but silently dropped en route to the resolver,
  // would go unnoticed. Iterate every key actually in the source file
  // (self-referential, so no per-key value needs hand-copying here) and
  // assert getFriendlyToolName resolves it exactly.
  //
  // Typed with `satisfies Partial<…>` rather than annotated
  // `Record<Locale, …>`: a downstream build can replace @/i18n/translations
  // with one that adds locales, widening Locale beyond the source objects
  // this repo ships (src/lib/time-utils.ts types displayDateLocales as a
  // Partial for the same reason, and its test pins that annotation). An
  // exhaustive Record here would demand an entry for a locale core has no
  // module to import, breaking their type-check on a file only this repo
  // owns. `satisfies` still rejects a wrong value shape or a key that is not
  // a Locale at all.
  const toolNames = {
    en: en.traceEventRenderer.toolNames,
    zh: zh.traceEventRenderer.toolNames,
  } satisfies Partial<Record<Locale, Record<string, string>>>

  for (const locale of Object.keys(toolNames) as (keyof typeof toolNames)[]) {
    it(`resolves all ${Object.keys(toolNames[locale]).length} ${locale} entries to their literal source value`, () => {
      const tDynamic = (key: string, fallback: string) =>
        resolveDynamicTranslation(locale, key, fallback)
      for (const [toolName, expected] of Object.entries(toolNames[locale])) {
        expect(getFriendlyToolName(toolName, tDynamic)).toBe(expected)
      }
    })
  }

  it("keeps the same set of tool names mapped in both locales", () => {
    expect(Object.keys(toolNames.en).sort()).toEqual(Object.keys(toolNames.zh).sort())
  })
})
