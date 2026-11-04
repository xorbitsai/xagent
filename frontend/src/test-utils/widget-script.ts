import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { pathToFileURL } from "node:url"
import { createInstrumenter } from "istanbul-lib-instrument"
import { inject } from "vitest"

const widgetScriptPath = resolve(process.cwd(), "public/widget.js")
export const widgetScript = readFileSync(widgetScriptPath, "utf8")
const scriptUrl = pathToFileURL(widgetScriptPath).href

// This static IIFE bypasses Vite transforms. Only coverage runs need standard
// Istanbul counters; normal tests evaluate the exact browser asset. The
// standalone instrumenter is not the provider's bundled implementation (e.g.
// line-ignore hints differ). Revalidate if this asset gains such hints.
// __VITEST_COVERAGE__ is private to the pinned provider: re-check this key
// whenever @vitest/coverage-istanbul is upgraded. Widget floors fail closed
// if it changes. Vitest's jsdom env aliases window to the worker global, so
// no cross-realm collector bridge is needed.
const script = inject("widgetCoverageEnabled") ? createInstrumenter({
  esModules: true,
  compact: false,
  coverageVariable: "__VITEST_COVERAGE__",
  coverageGlobalScope: "globalThis",
  coverageGlobalScopeFunc: false,
}).instrumentSync(widgetScript, widgetScriptPath) : widgetScript

export function evaluateWidgetScript() {
  window.eval(`${script}\n//# sourceURL=${scriptUrl}`)
}
