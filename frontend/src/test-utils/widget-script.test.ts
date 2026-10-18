import { resolve } from "node:path"
import { pathToFileURL } from "node:url"
import { afterEach, expect, inject, it, vi } from "vitest"
import { evaluateWidgetScript, widgetScript } from "./widget-script"

afterEach(() => vi.restoreAllMocks())

it("instruments the static widget only when the coverage lane is enabled", () => {
  expect(window).toBe(globalThis)
  const evaluate = vi.spyOn(window, "eval").mockImplementation(() => undefined)
  evaluateWidgetScript()
  const source = String(evaluate.mock.calls[0][0])
  if (inject("widgetCoverageEnabled")) {
    expect(source).toContain("__VITEST_COVERAGE__")
  } else {
    expect(source).toBe(`${widgetScript}\n//# sourceURL=${pathToFileURL(resolve("public/widget.js")).href}`)
  }
})
