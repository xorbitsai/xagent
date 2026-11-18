import { IDBDatabase } from "fake-indexeddb"
import "fake-indexeddb/auto"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(resolvePromise => { resolve = resolvePromise })
  return { promise, resolve }
}

describe("auth coordination fallback", () => {
  beforeEach(() => {
    vi.resetModules()
    Object.defineProperty(navigator, "locks", { configurable: true, value: undefined })
  })
  afterEach(() => vi.restoreAllMocks())

  it("does not settle before an action that outlives an aborted IndexedDB lock", async () => {
    let transaction: IDBTransaction | null = null
    const originalTransaction = IDBDatabase.prototype.transaction
    vi.spyOn(IDBDatabase.prototype, "transaction").mockImplementation(function (this: IDBDatabase, ...args) {
      const created = Reflect.apply(originalTransaction, this, args) as IDBTransaction
      transaction = created
      return created
    })
    const started = deferred<void>()
    const release = deferred<void>()
    let sideEffectCompleted = false
    const { withAuthCoordination } = await import("@/lib/auth-coordination")

    const coordinated = withAuthCoordination("xagent-auth-refresh:test", async () => {
      started.resolve()
      await release.promise
      sideEffectCompleted = true
      return "done"
    })
    await started.promise
    expect(transaction).not.toBeNull()
    const aborted = deferred<void>()
    transaction!.addEventListener("abort", () => aborted.resolve(), { once: true })
    transaction!.abort()

    let settled = false
    void coordinated.finally(() => { settled = true })
    await aborted.promise
    await Promise.resolve()
    expect(settled).toBe(false)
    release.resolve()

    await expect(coordinated).resolves.toEqual({ status: "completed", value: "done" })
    expect(sideEffectCompleted).toBe(true)
  })
})
