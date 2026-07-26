import { createHash } from "node:crypto"
import {
  mkdir,
  readFile,
  readdir,
  writeFile,
} from "node:fs/promises"
import { dirname, relative, resolve, sep } from "node:path"
import { fileURLToPath } from "node:url"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const dist = resolve(root, "dist")
const artifacts = resolve(root, "artifacts")
const manifest = JSON.parse(await readFile(resolve(dist, "manifest.json"), "utf8"))

if (
  typeof manifest.version !== "string" ||
  typeof manifest.version_name !== "string"
) {
  throw new Error("Extension manifest is missing generated Xagent version metadata")
}

const files = await collectFiles(dist)
if (!files.some((file) => file.name === "manifest.json")) {
  throw new Error("Extension build is missing manifest.json")
}

const archive = createStoredZip(files)
const artifactVersion = manifest.version_name.replaceAll("+", "-")
const baseName = `xagent-browser-relay-${artifactVersion}.zip`
const archivePath = resolve(artifacts, baseName)
await mkdir(artifacts, { recursive: true })
await writeFile(archivePath, archive)
const digest = createHash("sha256").update(archive).digest("hex")
await writeFile(`${archivePath}.sha256`, `${digest}  ${baseName}\n`)
console.log(`${relative(root, archivePath)} (${archive.length} bytes)`)
console.log(`${digest}  ${baseName}`)

async function collectFiles(directory) {
  const entries = []
  for (const name of (await readdir(directory, { withFileTypes: true })).sort(
    (left, right) => left.name.localeCompare(right.name),
  )) {
    const path = resolve(directory, name.name)
    if (name.isDirectory()) {
      entries.push(...(await collectFiles(path)))
    } else if (name.isFile()) {
      entries.push({
        name: relative(dist, path).split(sep).join("/"),
        contents: await readFile(path),
      })
    }
  }
  return entries
}

function createStoredZip(files) {
  const localParts = []
  const centralParts = []
  let offset = 0
  for (const file of files) {
    const name = Buffer.from(file.name, "utf8")
    const checksum = crc32(file.contents)
    const local = Buffer.alloc(30)
    local.writeUInt32LE(0x04034b50, 0)
    local.writeUInt16LE(20, 4)
    local.writeUInt16LE(0x0800, 6)
    local.writeUInt16LE(0, 8)
    local.writeUInt16LE(0, 10)
    local.writeUInt16LE(0x5021, 12)
    local.writeUInt32LE(checksum, 14)
    local.writeUInt32LE(file.contents.length, 18)
    local.writeUInt32LE(file.contents.length, 22)
    local.writeUInt16LE(name.length, 26)
    local.writeUInt16LE(0, 28)
    localParts.push(local, name, file.contents)

    const central = Buffer.alloc(46)
    central.writeUInt32LE(0x02014b50, 0)
    central.writeUInt16LE(20, 4)
    central.writeUInt16LE(20, 6)
    central.writeUInt16LE(0x0800, 8)
    central.writeUInt16LE(0, 10)
    central.writeUInt16LE(0, 12)
    central.writeUInt16LE(0x5021, 14)
    central.writeUInt32LE(checksum, 16)
    central.writeUInt32LE(file.contents.length, 20)
    central.writeUInt32LE(file.contents.length, 24)
    central.writeUInt16LE(name.length, 28)
    central.writeUInt16LE(0, 30)
    central.writeUInt16LE(0, 32)
    central.writeUInt16LE(0, 34)
    central.writeUInt16LE(0, 36)
    central.writeUInt32LE(0, 38)
    central.writeUInt32LE(offset, 42)
    centralParts.push(central, name)
    offset += local.length + name.length + file.contents.length
  }

  const centralDirectory = Buffer.concat(centralParts)
  const end = Buffer.alloc(22)
  end.writeUInt32LE(0x06054b50, 0)
  end.writeUInt16LE(0, 4)
  end.writeUInt16LE(0, 6)
  end.writeUInt16LE(files.length, 8)
  end.writeUInt16LE(files.length, 10)
  end.writeUInt32LE(centralDirectory.length, 12)
  end.writeUInt32LE(offset, 16)
  end.writeUInt16LE(0, 20)
  return Buffer.concat([...localParts, centralDirectory, end])
}

function crc32(contents) {
  const crcTable = Array.from({ length: 256 }, (_, index) => {
    let entry = index
    for (let bit = 0; bit < 8; bit += 1) {
      entry = (entry >>> 1) ^ (entry & 1 ? 0xedb88320 : 0)
    }
    return entry >>> 0
  })
  let value = 0xffffffff
  for (const byte of contents) {
    value = (value >>> 8) ^ crcTable[(value ^ byte) & 0xff]
  }
  return (value ^ 0xffffffff) >>> 0
}
