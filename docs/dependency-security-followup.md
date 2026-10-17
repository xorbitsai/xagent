# Dependency security follow-up (CASA 5)

This branch follows core PR #3180 and keeps its SheetJS/KaTeX patches. It does
not remove `pptxgenjs`: Docker also uses that package for backend presentations.

Verified changes:

- Vitest and its V8 coverage provider 3.2.6 fix the old UI-server arbitrary
  read/execution advisory. Vite 7.3.7 fixes the remaining Vite 7 advisories.
- Override every `tinypool` copy to 2.1.2 (worker-options prototype gadgets),
  `form-data` to 4.0.6 (multipart CRLF injection), `image-size` to 2.0.3
  (image-parser loops), and `postcss-selector-parser` to 7.1.6 (CPU exhaustion).
- The image-size/worker/selector overrides cross their parents' declared major
  ranges. Keep the real Node PPTX roundtrip, frontend build and full required
  coverage lane as compatibility gates. Do not merely suppress audit findings.

The lock was regenerated with npm 11.6.2 after npm 10.9.4 hit an internal
`edgesOut` peer-resolution error during the trial Vitest 4 migration. Clean
`npm ci` and the committed lockfile remain the normal installation contract.

The initial fresh audit found 20 affected package nodes (3 critical, 11 high,
6 moderate). The follow-up audit found 11 (0 critical, 8 high, 3 moderate).
These are npm dependency-node counts, not counts of independent application
exploits. This does not audit all Python, SaaS or Toby dependencies.

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
