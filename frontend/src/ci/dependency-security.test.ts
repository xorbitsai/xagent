import { spawnSync } from "node:child_process"
import { readFileSync } from "node:fs"
import postcss from "postcss"
import tailwindcss from "tailwindcss"
import { describe, expect, it } from "vitest"

const manifest = JSON.parse(readFileSync("package.json", "utf8"))
const lock = JSON.parse(readFileSync("package-lock.json", "utf8"))

describe("patched dependency resolution", () => {
  it.each([
    ["form-data", "4.0.6"],
    ["image-size", "2.0.3"],
    ["postcss-selector-parser", "7.1.6"],
  ])("pins every transitive %s copy to its verified patch", (name, version) => {
    expect(manifest.overrides[name]).toBe(version)
    const copies = Object.entries(lock.packages).filter(([location]) =>
      location.endsWith(`/node_modules/${name}`) || location === `node_modules/${name}`,
    )
    expect(copies.length).toBeGreaterThan(0)
    for (const [, entry] of copies) expect((entry as { version: string }).version).toBe(version)
  })

  it("keeps the verified test runner, coverage provider and Vite in the lockfile", () => {
    for (const name of ["vitest", "@vitest/coverage-istanbul"]) {
      expect(manifest.devDependencies[name]).toBe("4.1.12")
      expect(lock.packages[`node_modules/${name}`].version).toBe("4.1.12")
    }
    expect(lock.packages["node_modules/vite"].version).toBe("7.3.7")
    expect(Object.keys(lock.packages).some(location => location.endsWith("node_modules/tinypool"))).toBe(false)
    for (const orphan of ["@bcoe/v8-coverage", "ast-v8-to-istanbul", "ast-v8-to-istanbul/node_modules/js-tokens"]) {
      expect(lock.packages[`node_modules/${orphan}`]).toBeUndefined()
    }
  })

  it("preserves Tailwind variant CSS across the selector-parser major override", async () => {
    const candidates: string[] = JSON.parse(readFileSync("src/ci/fixtures/tailwind-selector-variants.json", "utf8"))
    // Generated with Tailwind 3.4.19 and its declared-compatible parser 6.1.4,
    // then compared byte-for-byte with 7.1.6. Keep this reviewed baseline:
    // a successful Next build alone cannot catch silent selector drift.
    const expected = readFileSync("src/ci/fixtures/tailwind-selector-compat.css", "utf8")
      .replace(/\r\n/g, "\n").trim()
    const result = await postcss([tailwindcss({
      content: [{ raw: `<div class="${candidates.join(" ")}"></div>`, extension: "html" }],
      corePlugins: { preflight: false },
      plugins: [],
    })]).process("@tailwind utilities;", { from: undefined })
    expect(result.css.trim()).toBe(expected)
  })

  it("still creates a real Node PPTX containing the original PNG", () => {
    expect(manifest.dependencies.pptxgenjs).toBe("4.0.1")
    const result = spawnSync(process.execPath, ["-e", `
      const assert = require('node:assert/strict');
      const PptxGenJS = require('pptxgenjs');
      const JSZip = require('jszip');
      const { imageSize } = require('image-size');
      const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZKxkAAAAASUVORK5CYII=', 'base64');
      assert.equal(imageSize(png).width, 1);
      const presentation = new PptxGenJS();
      presentation.addSlide().addImage({ data: 'image/png;base64,' + png.toString('base64'), x: 1, y: 1, w: 1, h: 1 });
      (async () => {
        const bytes = await presentation.write({ outputType: 'nodebuffer' });
        const zip = await JSZip.loadAsync(bytes);
        const media = zip.file(/^ppt\\/media\\/.*\\.png$/);
        assert.equal(media.length, 1);
        assert.deepEqual(await media[0].async('nodebuffer'), png);
        console.log('PPTX image roundtrip passed');
      })().catch(error => { console.error(error); process.exitCode = 1; });
    `], { encoding: "utf8", timeout: 15000 })
    expect(result.status, result.stderr).toBe(0)
    expect(result.stdout).toContain("PPTX image roundtrip passed")
  }, 20_000)
})
