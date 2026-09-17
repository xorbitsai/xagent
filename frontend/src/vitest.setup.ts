import "@testing-library/jest-dom/vitest"
import { beforeEach, vi } from "vitest"

// jsdom does not implement blob URL APIs; InlineFilePreview authenticated image
// previews rely on them in tests that assert blob: src values or spy on revoke.
URL.createObjectURL = vi.fn(
  (blob: Blob) => `blob:mock-${blob.size}`
) as typeof URL.createObjectURL
URL.revokeObjectURL = vi.fn() as typeof URL.revokeObjectURL

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

globalThis.ResizeObserver = ResizeObserverMock

// jsdom does not implement DOMMatrixReadOnly; @xyflow/react reads the scale
// off it when recomputing node internals (e.g. after a node's type changes).
class DOMMatrixReadOnlyMock {
  m22: number

  constructor(transform: string) {
    const scale = transform?.match(/scale\(([0-9.]+)\)/)?.[1]
    this.m22 = scale !== undefined ? Number(scale) : 1
  }
}

// @ts-expect-error partial DOMMatrixReadOnly polyfill, sufficient for @xyflow/react in tests
globalThis.DOMMatrixReadOnly = DOMMatrixReadOnlyMock

class LocalStorageMock implements Storage {
  private store = new Map<string, string>()

  get length() {
    return this.store.size
  }

  clear() {
    this.store.clear()
  }

  getItem(key: string) {
    return this.store.get(key) ?? null
  }

  key(index: number) {
    return Array.from(this.store.keys())[index] ?? null
  }

  removeItem(key: string) {
    this.store.delete(key)
  }

  setItem(key: string, value: string) {
    this.store.set(key, String(value))
  }
}

Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: new LocalStorageMock(),
})

Object.defineProperty(window, "localStorage", {
  configurable: true,
  value: globalThis.localStorage,
})

beforeEach(() => {
  localStorage.clear()
})
