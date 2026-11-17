import React from "react";
import { cn } from "@/lib/utils";

/** What an avatar is drawn from. `agentId` names the Agent a persona stands
 * for, when it stands for one: a distribution that lets people style an
 * Agent's avatar looks the style up by it. Nothing here reads it. */
export interface AvatarPersona {
  name: string;
  avatar?: string | null;
  agentId?: number | string | null;
}

interface PersonaAvatarProps {
  /** Only `name` and `avatar` are read - narrower than the full
   * `PersonaInfo` type so callers with just an agent's name/logo (no real
   * persona) can reuse this without fabricating unused fields. */
  persona: AvatarPersona;
  /** Tailwind size classes, e.g. "h-11 w-11" or "h-16 w-16". */
  sizeClassName: string;
  /** Tailwind text-size class for the fallback initial, e.g. "text-sm" or "text-xl". */
  textClassName?: string;
  className?: string;
  /** Set when the caller's own visible text already names this persona
   * (e.g. a picker button rendering the name right next to the avatar) -
   * without this, an `<img alt="Maya">`/initial sitting beside a "Maya"
   * text node inside the same interactive control gets its accessible
   * name announced twice ("Maya Maya"). */
  decorative?: boolean;
  /** Inline style pass-through - e.g. a caller-computed box-shadow ring color
   * that can't be expressed as a static Tailwind class. */
  style?: React.CSSProperties;
  /** Set while this persona is working on a reply, so the avatar reads as
   * busy rather than idle. */
  thinking?: boolean;
}

/** The color a persona's avatar is drawn in, as an RGB triple, or null when
 * the avatar has no color of its own. Here an avatar is an uploaded image or
 * a neutral initial, so there never is one. A distribution that draws colored
 * avatars replaces this module and reports the color, and a conversation with
 * that agent is then tinted to match (see `useAgentAccentAttributes`). It is a
 * hook so that a color that is only known later, such as one fetched after the
 * first render, can still reach the conversation.
 *
 * A replacement must keep every export of this module and accept `thinking`. */
export function usePersonaAccentColor(
  // eslint-disable-next-line @typescript-eslint/no-unused-vars -- part of the contract a replacement implements
  persona: AvatarPersona | null | undefined,
): readonly [number, number, number] | null {
  return null;
}

/** A persona's circular avatar image, falling back to its first initial on a
 * muted background when no avatar URL is set. Shared between the
 * marketplace card and detail page so the fallback rendering can't drift
 * between the two. */
export function PersonaAvatar({
  persona,
  sizeClassName,
  textClassName = "text-sm",
  className,
  decorative = false,
  style,
  thinking = false,
}: PersonaAvatarProps) {
  const thinkingClassName = thinking && "motion-safe:animate-pulse";
  if (persona.avatar) {
    return (
      <img
        src={persona.avatar}
        alt={decorative ? "" : persona.name}
        style={style}
        className={cn(sizeClassName, "flex-shrink-0 rounded-full object-cover", thinkingClassName, className)}
      />
    );
  }

  return (
    <div
      aria-hidden={decorative || undefined}
      style={style}
      className={cn(
        sizeClassName,
        "flex flex-shrink-0 items-center justify-center rounded-full bg-muted font-semibold text-muted-foreground",
        textClassName,
        thinkingClassName,
        className
      )}
    >
      {persona.name.slice(0, 1).toUpperCase()}
    </div>
  );
}
