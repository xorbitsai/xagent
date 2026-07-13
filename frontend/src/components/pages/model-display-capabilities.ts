export interface DisplayCapabilityModel {
  category: string
  abilities?: string[]
}

/**
 * Build provider-card capability badges without collapsing distinct audio
 * model categories that share the generic backend `generate` ability.
 */
export function getProviderDisplayCapabilities(
  models: DisplayCapabilityModel[],
  activeTab: string,
): string[] {
  const capabilities = new Set<string>()

  models.forEach((model) => {
    if (activeTab === "audio" && model.category === "sound_effect") {
      capabilities.add("sound_effect")
      return
    }
    if (activeTab === "audio" && model.category === "music") {
      capabilities.add("music")
      return
    }

    model.abilities?.forEach((ability) => capabilities.add(ability))
  })

  return Array.from(capabilities)
}
