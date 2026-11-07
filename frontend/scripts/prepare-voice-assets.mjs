import { copyFile, mkdir } from "node:fs/promises"
import { createRequire } from "node:module"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const require = createRequire(import.meta.url)
const frontend = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const vad = dirname(require.resolve("@ricky0123/vad-web"))
const ort = dirname(require.resolve("onnxruntime-web/wasm"))
const destination = resolve(frontend, "public/voice-vad")
await mkdir(destination, { recursive: true })
for (const [root, files] of [
  [vad, ["silero_vad_v5.onnx"]],
  [ort, ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]],
]) {
  for (const name of files) await copyFile(resolve(root, name), resolve(destination, name))
}
