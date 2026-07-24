import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const dist = resolve(root, "dist")
const emittedRoot = resolve(root, ".ts-build")

await rm(dist, { recursive: true, force: true })
await mkdir(dist, { recursive: true })
await cp(resolve(root, "manifest.json"), resolve(dist, "manifest.json"))
await cp(resolve(root, "src/popup.html"), resolve(dist, "popup.html"))
await cp(resolve(root, "src/popup.css"), resolve(dist, "popup.css"))

for (const file of ["protocol.js", "popup.js", "service-worker.js"]) {
  const contents = await readFile(resolve(emittedRoot, file))
  await writeFile(resolve(dist, file), contents)
}
await rm(emittedRoot, { recursive: true, force: true })
