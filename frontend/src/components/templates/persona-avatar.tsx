import React from "react";
import { cn } from "@/lib/utils";

interface PersonaAvatarProps {
  /** Only `name` and `avatar` are read - narrower than the full
   * `PersonaInfo` type so callers with just an agent's name/logo (no real
   * persona) can reuse this without fabricating unused fields. */
  persona: { name: string; avatar?: string | null };
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
}: PersonaAvatarProps) {
  if (persona.avatar) {
    return (
      <img
        src={persona.avatar}
        alt={decorative ? "" : persona.name}
        className={cn(sizeClassName, "flex-shrink-0 rounded-full object-cover", className)}
      />
    );
  }

  return (
    <div
      aria-hidden={decorative || undefined}
      className={cn(
        sizeClassName,
        "flex flex-shrink-0 items-center justify-center rounded-full bg-muted font-semibold text-muted-foreground",
        textClassName,
        className
      )}
    >
      {persona.name.slice(0, 1).toUpperCase()}
    </div>
  );
}
