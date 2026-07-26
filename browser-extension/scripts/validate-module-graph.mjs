import { posix } from "node:path"

const IMPORT_PATTERNS = [
  /\bfrom\s*["'](\.[^"']+)["']/g,
  /\bimport\s*["'](\.[^"']+)["']/g,
  /\bimport\s*\(\s*["'](\.[^"']+)["']\s*\)/g,
]

export function validateRelativeModuleImports(files) {
  const names = new Set(files.map((file) => file.name))
  for (const file of files) {
    if (!file.name.endsWith(".js")) continue
    const source = file.contents.toString("utf8")
    for (const pattern of IMPORT_PATTERNS) {
      pattern.lastIndex = 0
      for (const match of source.matchAll(pattern)) {
        const specifier = match[1]
        const importedName = posix.normalize(
          posix.join(posix.dirname(file.name), specifier),
        )
        if (importedName.startsWith("../") || !names.has(importedName)) {
          throw new Error(
            `Extension build is missing ${specifier} imported by ${file.name}`,
          )
        }
      }
    }
  }
}
