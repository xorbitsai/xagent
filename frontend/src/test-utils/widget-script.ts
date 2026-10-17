import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { pathToFileURL } from "node:url"
import { createInstrumenter } from "istanbul-lib-instrument"

export const widgetScriptPath = resolve(process.cwd(), "public/widget.js")
export const widgetScript = readFileSync(widgetScriptPath, "utf8")
const scriptUrl = pathToFileURL(widgetScriptPath).href

// This standalone browser asset is read, not imported through Vite. Apply
// exactly the same Istanbul counters used for imported owners so coverage
// records its real branch outcomes, including implicit else/guard paths.
// Instrument once; every fresh controller contributes to the same file map.
// This is the collector variable used by the pinned Vitest 4 Istanbul provider.
const instrumented = createInstrumenter({
  coverageVariable: "__VITEST_COVERAGE__",
  coverageGlobalScope: "globalThis",
  coverageGlobalScopeFunc: false,
}).instrumentSync(widgetScript, widgetScriptPath)

export function evaluateWidgetScript() {
  // jsdom's eval realm and the Vitest worker do not share global properties.
  // Share the collector, not calculated coverage: the evaluated script itself
  // increments the standard Istanbul counters used by the worker's reporter.
  const worker = globalThis as typeof globalThis & { __VITEST_COVERAGE__?: Record<string, unknown> }
  const page = window as typeof window & { __VITEST_COVERAGE__?: Record<string, unknown> }
  page.__VITEST_COVERAGE__ = worker.__VITEST_COVERAGE__ ??= {}
  window.eval(`${instrumented}\n//# sourceURL=${scriptUrl}`)
}
