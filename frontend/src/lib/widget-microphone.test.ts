import { afterEach, describe, expect, it } from "vitest"
import { pageMayUseMicrophone } from "./widget-microphone"

type Policy = { allowsFeature?: unknown }
const allow = (value: boolean): Policy => ({ allowsFeature: (feature: string) => feature === "microphone" && value })
const throwing: Policy = { allowsFeature: () => { throw new Error("policy query failed") } }

const interfaces: Array<{ label: string; set: () => void; expected: boolean }> = [
  { label: "permissionsPolicy allows", set: () => define("permissionsPolicy", allow(true)), expected: true },
  { label: "permissionsPolicy denies", set: () => define("permissionsPolicy", allow(false)), expected: false },
  { label: "only featurePolicy allows", set: () => define("featurePolicy", allow(true)), expected: true },
  { label: "only featurePolicy denies", set: () => define("featurePolicy", allow(false)), expected: false },
  { label: "allowsFeature throws", set: () => define("featurePolicy", throwing), expected: true },
  { label: "allowsFeature is not a function", set: () => define("featurePolicy", { allowsFeature: "yes" }), expected: true },
  { label: "no policy interface", set: () => {}, expected: true },
]

function define(name: "permissionsPolicy" | "featurePolicy", value: Policy) {
  Object.defineProperty(document, name, { configurable: true, value })
}
const secureDescriptor = Object.getOwnPropertyDescriptor(window, "isSecureContext")

afterEach(() => {
  delete (document as { permissionsPolicy?: unknown }).permissionsPolicy
  delete (document as { featurePolicy?: unknown }).featurePolicy
  if (secureDescriptor) Object.defineProperty(window, "isSecureContext", secureDescriptor)
  else delete (window as { isSecureContext?: unknown }).isSecureContext
})

describe("pageMayUseMicrophone", () => {
  it.each(interfaces)("secure context, $label", ({ set, expected }) => {
    Object.defineProperty(window, "isSecureContext", { configurable: true, value: true })
    set()
    expect(pageMayUseMicrophone()).toBe(expected)
  })

  it("prefers document.permissionsPolicy over document.featurePolicy when both exist", () => {
    Object.defineProperty(window, "isSecureContext", { configurable: true, value: true })
    define("permissionsPolicy", allow(true))
    define("featurePolicy", allow(false))
    expect(pageMayUseMicrophone()).toBe(true)
  })

  it.each([
    { label: "false", value: false },
    { label: "missing", value: undefined },
  ])("never allows a page whose isSecureContext is $label", ({ value }) => {
    for (const { set } of interfaces) {
      delete (document as { permissionsPolicy?: unknown }).permissionsPolicy
      delete (document as { featurePolicy?: unknown }).featurePolicy
      if (value === undefined) delete (window as { isSecureContext?: unknown }).isSecureContext
      else Object.defineProperty(window, "isSecureContext", { configurable: true, value })
      set()
      expect(pageMayUseMicrophone()).toBe(false)
    }
  })
})
