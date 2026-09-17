import type {
  HomeGetStartedDestinationOverrides,
  HomePageExtensionComponent,
} from "@/lib/page-extension-contracts"

export const HomePageExtension: HomePageExtensionComponent = () => null

// Optional export: a replacement module may omit `homeGetStartedDestinationOverrides`
// entirely and still satisfy this module's contract (only `HomePageExtension` is
// required). The canonical default here is the empty object, i.e. all destinations
// resolve to their OSS defaults.
export const homeGetStartedDestinationOverrides: HomeGetStartedDestinationOverrides = {}
