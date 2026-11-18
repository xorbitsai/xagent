import type { AvatarAccentColor, AvatarPersona } from "@/lib/page-extension-contracts";

/**
 * Build-time extension point for distributions that draw colored avatars. The
 * OSS implementation is deliberately inert.
 *
 * Returns the color a persona's avatar is drawn in, or null when the avatar has
 * no color of its own (here an avatar is an uploaded image or a neutral
 * initial, so there never is one). A conversation with an agent whose avatar
 * has a color is tinted to match (see `useAgentAccentAttributes`). It is a hook
 * so that a color that is only known later, such as one fetched after the first
 * render, can still reach the conversation.
 *
 * A replacement must keep this export and key anything it fetches or caches on
 * `persona.agentId` (and `persona.name`), never on the identity of the
 * `persona` object: callers build it on render. It must accept a null or
 * undefined persona and return null for it.
 */
export function usePersonaAccentColor(
  // eslint-disable-next-line @typescript-eslint/no-unused-vars -- part of the contract a replacement implements
  persona: AvatarPersona | null | undefined,
): AvatarAccentColor | null {
  return null;
}
