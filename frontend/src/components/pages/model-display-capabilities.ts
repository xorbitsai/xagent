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
    const isSoundEffect =
      activeTab === "audio" && model.category === "sound_effect"
    const isMusic = activeTab === "audio" && model.category === "music"

    if (isSoundEffect) {
      capabilities.add("sound_effect")
    } else if (isMusic) {
      capabilities.add("music")
    }

    model.abilities?.forEach((ability) => {
      if (ability !== "generate" || (!isSoundEffect && !isMusic)) {
        capabilities.add(ability)
      }
    })
  })

  return Array.from(capabilities)
}
