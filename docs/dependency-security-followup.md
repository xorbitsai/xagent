# Dependency security follow-up (CASA 5)

These dependency updates complement the SheetJS/KaTeX security patches. They
retain `pptxgenjs`: Docker also uses that package for backend presentations.

Verified changes:

- Vitest and its V8 coverage provider 3.2.6 fix the old UI-server arbitrary
  read/execution advisory. Vite 7.3.7 fixes the remaining Vite 7 advisories.
- Override every `tinypool` copy to 2.1.2 (worker-options prototype gadgets),
  `form-data` to 4.0.6 (multipart CRLF injection), `image-size` to 2.0.3
  (image-parser loops), and `postcss-selector-parser` to 7.1.6 (CPU exhaustion).
- The worker/selector overrides cross their parents' declared major ranges.
  `dependency-security.test.ts` compares Tailwind's generated variant CSS with
  a reviewed baseline from Tailwind 3.4.19 / selector-parser 6.1.4. The 7.1.6
  override produces identical CSS for all 20 fixture candidates: ordinary and
  named groups/peers, chained states, arbitrary variants and compound spacing/
  divider selectors. Keep this snapshot alongside the frontend build and full
  required coverage lane; a build alone cannot reveal silent selector drift.
- `pptxgenjs` 4.0.1 declares `image-size` but its shipped runtime does not import
  it. The 2.0.3 override therefore has no runtime effect on that package; the
  real PPTX/PNG roundtrip guards `pptxgenjs`/`jszip`, not `image-size` API
  compatibility. The test's 20-second budget exceeds its child-process timeout.

The lock was regenerated with npm 11.6.2 after npm 10.9.4 hit an internal
`edgesOut` peer-resolution error during the trial Vitest 4 migration. Clean
`npm ci` and the committed lockfile remain the normal installation contract.

The initial fresh audit found 20 affected package nodes (3 critical, 11 high,
6 moderate). The follow-up audit found 11 (0 critical, 8 high, 3 moderate).
These are npm dependency-node counts, not counts of independent application
exploits. This does not audit all Python dependencies, downstream applications
or deployed images.

Frontend development/testing requires Node.js `^20.19.0 || >=22.12.0` for
Vite 7.3.7; Node 18 is no longer supported by this test toolchain.

## Explicitly not closed

- `braces` recursion DoS still propagates through micromatch/fast-glob/chokidar,
  Tailwind 3 and the Next ESLint configuration. Do not downgrade Next merely
  because npm proposes an older lint configuration as an automatic fix.
- Vitest 3 still has the moderate mock-redirect traversal advisory. A trial
  Vitest 4.1.11 run passed all 2,352 required tests after isolation fixes, but
  its changed coverage mapping failed several existing per-file floors. Those
  floors were not lowered. The broader migration needs additional coverage
  tests; do not present this intermediate patch as full toolchain closure.
- Do not expose Vite/Vitest development or UI servers to untrusted networks.
  Repository test code itself must be trusted. Production dependency closure
  still requires upgrading the SaaS core pin, rebuilding and rescanning the
  actual deployed images.
