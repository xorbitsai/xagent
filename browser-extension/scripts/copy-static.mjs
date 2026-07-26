import {
  cp,
  mkdir,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { resolveExtensionVersion } from "./version.mjs"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const dist = resolve(root, "dist")
const emittedRoot = resolve(root, ".ts-build")

await rm(dist, { recursive: true, force: true })
await mkdir(dist, { recursive: true })
const manifest = JSON.parse(
  await readFile(resolve(root, "manifest.template.json"), "utf8"),
)
const extensionVersion = resolveExtensionVersion()
manifest.version = extensionVersion.chromeVersion
manifest.version_name = extensionVersion.displayVersion
await writeFile(
  resolve(dist, "manifest.json"),
  `${JSON.stringify(manifest, null, 2)}\n`,
)
await cp(resolve(root, "src/popup.html"), resolve(dist, "popup.html"))
await cp(resolve(root, "src/popup.css"), resolve(dist, "popup.css"))
await cp(resolve(root, "src/offscreen.html"), resolve(dist, "offscreen.html"))
await cp(
  resolve(root, "../frontend/public/xagent_logo.png"),
  resolve(dist, "xagent-logo.png"),
)

// Copy the complete emitted module graph. A fixed entrypoint list is unsafe:
// TypeScript preserves ESM imports, so a newly extracted helper module must be
// present beside the entrypoint that imports it.
for (const entry of await readdir(emittedRoot, { withFileTypes: true })) {
  await cp(
    resolve(emittedRoot, entry.name),
    resolve(dist, entry.name),
    { recursive: entry.isDirectory() },
  )
}
await rm(emittedRoot, { recursive: true, force: true })
console.log(
  `Browser Relay ${extensionVersion.displayVersion} (${extensionVersion.chromeVersion})`,
)
