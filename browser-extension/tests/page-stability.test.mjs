import assert from "node:assert/strict"
import test from "node:test"

import {
  looksLikeTransitionalDocument,
  samePageState,
  waitForPageStable,
} from "../.test-build/page-stability.js"

function state(overrides = {}) {
  return {
    url: "https://example.com/old",
    title: "Example",
    readyState: "complete",
    domSize: 100,
    interactiveSize: 20,
    textSize: 1_000,
    ...overrides,
  }
}

test("page state comparison includes document hydration signals", () => {
  const baseline = state()

  assert.equal(samePageState(baseline, state()), true)
  assert.equal(samePageState(baseline, state({ domSize: 101 })), false)
  assert.equal(samePageState(baseline, state({ textSize: 1_001 })), false)
})

test("a sparse replacement document is treated as transitional", () => {
  const baseline = state({ domSize: 100 })

  assert.equal(
    looksLikeTransitionalDocument(
      baseline,
      state({ url: "https://example.com/new", domSize: 1 }),
    ),
    true,
  )
  assert.equal(
    looksLikeTransitionalDocument(
      baseline,
      state({ url: "https://example.com/new", domSize: 30 }),
    ),
    false,
  )
})

test("settling waits through a loading shell and returns hydrated state", async () => {
  const samples = [
    state({
      url: "https://example.com/new",
      readyState: "loading",
      domSize: 1,
    }),
    state({ url: "https://example.com/new", domSize: 1 }),
    state({ url: "https://example.com/new", domSize: 120 }),
  ]
  const hydrated = state({
    url: "https://example.com/new",
    domSize: 120,
  })
  let reads = 0
  const result = await waitForPageStable(
    async () => {
      reads += 1
      return samples.shift() ?? hydrated
    },
    {
      baseline: state(),
      minimumWaitMs: 0,
      quietWindowMs: 2,
      pollIntervalMs: 1,
      timeoutMs: 100,
    },
  )

  assert.deepEqual(result, hydrated)
  assert.ok(reads >= 5)
})

test("settling remains bounded for a continuously changing page", async () => {
  let textSize = 0
  const result = await waitForPageStable(
    async () => state({ textSize: (textSize += 1) }),
    {
      minimumWaitMs: 0,
      quietWindowMs: 20,
      pollIntervalMs: 1,
      timeoutMs: 10,
    },
  )

  assert.ok(result)
  assert.ok(result.textSize > 0)
})
