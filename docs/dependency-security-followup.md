# Dependency security follow-up (CASA 5)

This migration follows merged core PRs #3180 and #3182 and keeps their
SheetJS/KaTeX and parser patches. It retains `pptxgenjs`: Docker also uses that
package for backend presentations.

Verified changes:

- Vitest and its matching Istanbul coverage provider 4.1.12 fix the old
  UI-server arbitrary read/execution and mock-redirect traversal advisories.
  Vite 7.3.7 fixes the remaining Vite 7 advisories.
- Vitest 4 no longer uses `tinypool`; the dependency and its major-version
  override are removed. Override every `form-data` copy to 4.0.6
  (multipart CRLF injection), `image-size` to 2.0.3
  (image-parser loops), and `postcss-selector-parser` to 7.1.6 (CPU exhaustion).
- `dependency-security.test.ts` compares Tailwind's generated variant CSS with
  a reviewed baseline from Tailwind 3.4.19 / selector-parser 6.1.4. The 7.1.6
  override produces identical CSS for all 20 fixture candidates: ordinary and
  named groups/peers, chained states, arbitrary variants and compound spacing/
  divider selectors. Keep this snapshot alongside the frontend build and full
  required coverage lane; a build alone cannot reveal silent selector drift.
  The selector-parser 7.1.6 override crosses Tailwind 3's declared `^6.1.2`
  major range, so this compatibility gate remains necessary.
- `pptxgenjs` 4.0.1 declares `image-size` but its shipped runtime does not import
  it. The 2.0.3 override therefore has no runtime effect on that package; the
  real PPTX/PNG roundtrip guards `pptxgenjs`/`jszip`, not `image-size` API
  compatibility. A separate direct image-size assertion checks PNG dimensions.
  The test's 20-second budget exceeds its child-process timeout.

The lock was regenerated with npm 11.6.2 after npm 10.9.4 hit an internal
`edgesOut` peer-resolution error during the Vitest 4 migration. Clean
`npm ci` and the committed lockfile remain the normal installation contract.

The initial fresh audit found 20 affected package nodes (3 critical, 11 high,
6 moderate). PR #3182 reduced this to 11 (0 critical, 8 high, 3 moderate).
The Vitest 4 follow-up reduces it to 8 (0 critical, 8 high, 0 moderate).
These are npm dependency-node counts, not counts of independent application
exploits. This does not audit all Python dependencies, downstream applications
or deployed images.

Frontend development/testing requires Node.js `^20.19.0 || >=22.12.0` for
Vite 7.3.7; Node 18 is no longer supported by this test toolchain.

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
the actual file only in coverage runs. Normal tests evaluate the unmodified
asset. Vitest's jsdom environment aliases `window` to the worker global;
there is no separate-realm collector bridge. The standalone instrumenter is
not identical to the provider's bundled implementation, including line-ignore
hint handling. Revalidate the helper if the asset gains such hints, and check
its private `__VITEST_COVERAGE__` key whenever the provider pin changes.
It does not fabricate hits, exclude branches, or rewrite
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
