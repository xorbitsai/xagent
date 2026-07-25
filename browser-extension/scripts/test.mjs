import { spawn } from "node:child_process"
import { readdir, rm } from "node:fs/promises"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const output = resolve(root, ".test-build")

try {
  await run(
    resolve(root, "node_modules/typescript/bin/tsc"),
    ["-p", resolve(root, "tsconfig.json"), "--outDir", output],
  )
  const tests = (await readdir(resolve(root, "tests")))
    .filter((name) => name.endsWith(".test.mjs"))
    .sort()
    .map((name) => resolve(root, "tests", name))
  await run(process.execPath, ["--test", ...tests])
} finally {
  await rm(output, { recursive: true, force: true })
}

function run(command, args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { stdio: "inherit" })
    child.on("error", reject)
    child.on("exit", (code, signal) => {
      if (code === 0) {
        resolvePromise()
      } else {
        reject(
          new Error(
            `${command} exited with ${signal ? `signal ${signal}` : `code ${code}`}`,
          ),
        )
      }
    })
  })
}
