# Dependency security follow-up (CASA 5)

This branch follows core PR #3180 and keeps its SheetJS/KaTeX patches. It does
not remove `pptxgenjs`: Docker also uses that package for backend presentations.

Verified changes:

- Vitest and its matching Istanbul coverage provider 4.1.12 fix the old
  UI-server arbitrary read/execution and mock-redirect traversal advisories.
  Vite 7.3.7 fixes the remaining Vite 7 advisories.
- Vitest 4 no longer uses `tinypool`; the dependency and its major-version
  override are removed. Override every `form-data` copy to 4.0.6
  (multipart CRLF injection), `image-size` to 2.0.3
  (image-parser loops), and `postcss-selector-parser` to 7.1.6 (CPU exhaustion).
- The image-size/selector overrides cross their parents' declared major
  ranges. Keep the real Node PPTX roundtrip, frontend build and full required
  coverage lane as compatibility gates. Do not merely suppress audit findings.

The lock was regenerated with npm 11.6.2 after npm 10.9.4 hit an internal
`edgesOut` peer-resolution error during the Vitest 4 migration. Clean
`npm ci` and the committed lockfile remain the normal installation contract.

The initial fresh audit found 20 affected package nodes (3 critical, 11 high,
6 moderate). PR #3182 reduced this to 11 (0 critical, 8 high, 3 moderate).
The Vitest 4 follow-up reduces it to 8 (0 critical, 8 high, 0 moderate).
These are npm dependency-node counts, not counts of independent application
exploits. This does not audit all Python, SaaS or Toby dependencies.

## Test-toolchain compatibility

Keep all existing per-owner coverage floors and required suite entries. The
migration adds behavioral coverage for authentication refresh outcomes,
unavailable login methods, sidebar task/release presentation, agent-card lookup,
file-mention keyboard handling, malformed session reports and widget teardown.
Vitest 4 requires constructor mocks to be constructible functions; standalone
mock call histories are explicitly cleared between tests without removing
restore behavior. No application behavior is changed to satisfy the tests.

The required lane uses the official Istanbul provider. The static
`public/widget.js` asset is evaluated by tests rather than imported through
Vite, so `src/test-utils/widget-script.ts` applies standard Istanbul counters to
the actual file and shares the pinned provider's collector across the jsdom
and worker realms. It does not fabricate hits, exclude branches, or rewrite
production widget code. A separate smoke test executes the exact
uninstrumented asset. This avoids relying on V8 range-to-AST remapping of
repeatedly evaluated script instances. All owners, including the widget, still
have to pass the unchanged thresholds.

## Explicitly not closed

- `braces` recursion DoS still propagates through micromatch/fast-glob/chokidar,
  Tailwind 3 and the Next ESLint configuration. Do not downgrade Next merely
  because npm proposes an older lint configuration as an automatic fix.
- Do not expose Vite/Vitest development or UI servers to untrusted networks.
  Repository test code itself must be trusted. Production dependency closure
  still requires upgrading the SaaS core pin, rebuilding and rescanning the
  actual deployed images.
