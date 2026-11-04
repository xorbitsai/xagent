/**
 * Integration tests: when building ingest FormData with chunk_strategy recursive
 * and a separators string, the request includes the separators field as JSON.
 */
import { describe, it, expect } from "vitest"
import { parseSeparatorsInput } from "./separators"

function buildIngestFormDataEntries(config: {
  chunk_strategy: string
  separators?: string
  [key: string]: unknown
}): Record<string, string> {
  const entries: Record<string, string> = {}
  entries.chunk_strategy = config.chunk_strategy
  if (config.chunk_strategy === "recursive" && config.separators?.trim()) {
    const parsed = parseSeparatorsInput(config.separators)
    if (parsed.length > 0) {
      entries.separators = JSON.stringify(parsed)
    }
  }
  return entries
}

describe("ingest FormData integration: separators", () => {
  it("includes separators as JSON when strategy is recursive and separators input is non-empty", () => {
    const entries = buildIngestFormDataEntries({
      chunk_strategy: "recursive",
      separators: "\\n\\n,\\n,。",
    })
    expect(entries.separators).toBeDefined()
    expect(JSON.parse(entries.separators!)).toEqual(["\n\n", "\n", "。"])
  })

  it("does not include separators when strategy is fixed_size", () => {
    const entries = buildIngestFormDataEntries({
      chunk_strategy: "fixed_size",
      separators: "\\n\\n,\\n",
    })
    expect(entries.separators).toBeUndefined()
  })

  it("does not include separators when strategy is recursive but input is empty", () => {
    const entries = buildIngestFormDataEntries({
      chunk_strategy: "recursive",
      separators: "",
    })
    expect(entries.separators).toBeUndefined()
  })

  it("does not include separators when strategy is recursive but parsed list is empty", () => {
    const entries = buildIngestFormDataEntries({
      chunk_strategy: "recursive",
      separators: "   ,  , ",
    })
    expect(entries.separators).toBeUndefined()
  })

  it("includes single separator when only one item is given", () => {
    const entries = buildIngestFormDataEntries({
      chunk_strategy: "recursive",
      separators: "\\t",
    })
    expect(entries.separators).toBe(JSON.stringify(["\t"]))
  })
})
