/**
 * Bounded-concurrency execution helpers.
 *
 * Used by KB multi-file upload so that files enqueue with a small number of
 * concurrent requests instead of one-at-a-time serialization, while never
 * cancelling later files just because an earlier one failed.
 */

/** Concurrent in-flight requests when background ingestion jobs are available. */
export const KB_UPLOAD_ENQUEUE_CONCURRENCY_LIMIT = 2

/**
 * Synchronous fallback limit. Keeping this at 1 avoids launching multiple
 * long-running inline parse/embed requests in parallel.
 */
export const KB_UPLOAD_SYNC_CONCURRENCY_LIMIT = 1

/**
 * Run `worker` over every item with at most `limit` concurrent executions.
 *
 * Every item is attempted; a rejected worker does not cancel items that have
 * not started yet. Results are returned in input order as
 * `PromiseSettledResult`s so callers can report per-item success/failure.
 */
export async function runWithConcurrencyLimit<T, R>(
  items: readonly T[],
  limit: number,
  worker: (item: T, index: number) => Promise<R>
): Promise<PromiseSettledResult<R>[]> {
  const results: PromiseSettledResult<R>[] = new Array(items.length)
  const effectiveLimit = Math.max(1, Math.min(limit, items.length || 1))
  let nextIndex = 0

  async function runNext(): Promise<void> {
    while (true) {
      const currentIndex = nextIndex
      nextIndex += 1
      if (currentIndex >= items.length) {
        return
      }
      try {
        const value = await worker(items[currentIndex], currentIndex)
        results[currentIndex] = { status: "fulfilled", value }
      } catch (reason) {
        results[currentIndex] = { status: "rejected", reason }
      }
    }
  }

  const runners: Promise<void>[] = []
  for (let i = 0; i < effectiveLimit; i++) {
    runners.push(runNext())
  }
  await Promise.all(runners)
  return results
}
