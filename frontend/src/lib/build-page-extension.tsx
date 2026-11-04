import React from "react"

import type {
  BuildAgentCardExtensionComponent,
  BuildPageExtensionProviderComponent,
} from "@/lib/page-extension-contracts"

export const BuildPageExtensionProvider: BuildPageExtensionProviderComponent = ({
  children,
}) => <>{children}</>

export const BuildAgentCardExtension: BuildAgentCardExtensionComponent =
  () => null
