import { describe, expect, it } from "vitest"

import { runWithConcurrencyLimit } from "./concurrency"

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

describe("runWithConcurrencyLimit", () => {
  it("never exceeds the concurrency limit", async () => {
    let inFlight = 0
    let maxInFlight = 0
    const items = [0, 1, 2, 3, 4, 5, 6]

    const results = await runWithConcurrencyLimit(items, 2, async (item) => {
      inFlight += 1
      maxInFlight = Math.max(maxInFlight, inFlight)
      await new Promise((resolve) => setTimeout(resolve, 5))
      inFlight -= 1
      return item * 2
    })

    expect(maxInFlight).toBe(2)
    expect(results.map((r) => (r.status === "fulfilled" ? r.value : null))).toEqual([
      0, 2, 4, 6, 8, 10, 12,
    ])
  })

  it("attempts every item even when an early one fails (no fail-fast)", async () => {
    const attempted: number[] = []

    const results = await runWithConcurrencyLimit([0, 1, 2, 3], 1, async (item) => {
      attempted.push(item)
      if (item === 0) {
        throw new Error("boom")
      }
      return item
    })

    expect(attempted).toEqual([0, 1, 2, 3])
    expect(results[0].status).toBe("rejected")
    expect(results.slice(1).every((r) => r.status === "fulfilled")).toBe(true)
  })

  it("preserves input order in results regardless of completion order", async () => {
    const first = deferred<number>()
    const second = deferred<number>()

    const promise = runWithConcurrencyLimit([first, second], 2, (d) => d.promise)

    // Resolve the second item before the first.
    second.resolve(20)
    first.resolve(10)

    const results = await promise
    expect(results.map((r) => (r.status === "fulfilled" ? r.value : null))).toEqual([
      10, 20,
    ])
  })

  it("handles an empty input list", async () => {
    const results = await runWithConcurrencyLimit([], 2, async () => 1)
    expect(results).toEqual([])
  })
})
