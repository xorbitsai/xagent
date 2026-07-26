export interface PageStabilityState {
  url: string
  title: string
  readyState: string
  domSize: number
  interactiveSize: number
  textSize: number
}

export interface PageStabilityOptions {
  baseline?: PageStabilityState | null
  minimumWaitMs?: number
  quietWindowMs?: number
  pollIntervalMs?: number
  timeoutMs?: number
}

type PageStateReader = () => Promise<PageStabilityState>

const DEFAULT_MINIMUM_WAIT_MS = 450
const DEFAULT_QUIET_WINDOW_MS = 350
const DEFAULT_POLL_INTERVAL_MS = 100
const DEFAULT_TIMEOUT_MS = 4_000

/**
 * Wait until the page has stopped changing after an input action.
 *
 * Browser actions often resolve before a client-side route or hydrated
 * document is usable. This bounded quiet-window wait avoids returning a
 * transitional observation while still allowing continuously changing pages
 * to proceed when the timeout expires.
 */
export async function waitForPageStable(
  readState: PageStateReader,
  options: PageStabilityOptions = {},
): Promise<PageStabilityState | null> {
  const minimumWaitMs = boundedDuration(
    options.minimumWaitMs,
    DEFAULT_MINIMUM_WAIT_MS,
  )
  const quietWindowMs = boundedDuration(
    options.quietWindowMs,
    DEFAULT_QUIET_WINDOW_MS,
  )
  const pollIntervalMs = Math.max(
    1,
    boundedDuration(options.pollIntervalMs, DEFAULT_POLL_INTERVAL_MS),
  )
  const timeoutMs = Math.max(
    pollIntervalMs,
    boundedDuration(options.timeoutMs, DEFAULT_TIMEOUT_MS),
  )
  const startedAt = Date.now()
  let stableSince = startedAt
  let previous: PageStabilityState | null = null
  let latest: PageStabilityState | null = null

  while (Date.now() - startedAt < timeoutMs) {
    await delay(pollIntervalMs)
    const sampledAt = Date.now()
    let current: PageStabilityState
    try {
      current = await readState()
    } catch {
      // Navigation destroys execution contexts briefly. Treat that as page
      // activity and keep polling until the new document becomes readable.
      previous = null
      stableSince = sampledAt
      continue
    }
    latest = current
    if (previous === null || !samePageState(previous, current)) {
      stableSince = sampledAt
    }
    previous = current

    const waitedLongEnough = sampledAt - startedAt >= minimumWaitMs
    const quietLongEnough = sampledAt - stableSince >= quietWindowMs
    if (
      waitedLongEnough &&
      quietLongEnough &&
      current.readyState === "complete" &&
      !looksLikeTransitionalDocument(options.baseline, current)
    ) {
      return current
    }
  }
  return latest
}

export function samePageState(
  left: PageStabilityState,
  right: PageStabilityState,
): boolean {
  return (
    left.url === right.url &&
    left.title === right.title &&
    left.readyState === right.readyState &&
    left.domSize === right.domSize &&
    left.interactiveSize === right.interactiveSize &&
    left.textSize === right.textSize
  )
}

export function looksLikeTransitionalDocument(
  baseline: PageStabilityState | null | undefined,
  current: PageStabilityState,
): boolean {
  if (
    baseline === null ||
    baseline === undefined ||
    baseline.url === current.url ||
    baseline.domSize < 20
  ) {
    return false
  }
  return current.domSize < Math.max(8, Math.floor(baseline.domSize * 0.2))
}

function boundedDuration(value: number | undefined, fallback: number): number {
  if (value === undefined || !Number.isFinite(value)) return fallback
  return Math.min(30_000, Math.max(0, Math.round(value)))
}

function delay(durationMs: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, durationMs))
}
