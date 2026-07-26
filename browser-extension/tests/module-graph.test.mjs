import assert from "node:assert/strict"
import test from "node:test"

import { validateRelativeModuleImports } from "../scripts/validate-module-graph.mjs"

function moduleFile(name, source = "") {
  return { name, contents: Buffer.from(source) }
}

test("extension package accepts a complete relative module graph", () => {
  assert.doesNotThrow(() =>
    validateRelativeModuleImports([
      moduleFile(
        "service-worker.js",
        'import { settle } from "./page-stability.js"',
      ),
      moduleFile("page-stability.js", "export const settle = true"),
    ]),
  )
})

test("extension package rejects a missing relative module", () => {
  assert.throws(
    () =>
      validateRelativeModuleImports([
        moduleFile(
          "service-worker.js",
          'import { settle } from "./page-stability.js"',
        ),
      ]),
    /missing \.\/page-stability\.js imported by service-worker\.js/,
  )
})
