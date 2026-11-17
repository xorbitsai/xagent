import { useMemo, type CSSProperties } from "react";

import { usePersonaAccentColor, type AvatarPersona } from "@/components/templates/persona-avatar";

type Rgb = readonly [number, number, number];

// WCAG AA for body text: the accent sits behind white message text.
const MIN_CONTRAST_WITH_WHITE = 4.5;
const DARKEN_STEPS = 50;

function relativeLuminance(color: Rgb): number {
  const [red, green, blue] = color.map((channel) => {
    const value = channel / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

/** Darkens `color` only as far as white text on it needs to stay readable, so
 * a light avatar color (mint, amber) still makes a usable message bubble. */
export function readableAccent(color: Rgb): Rgb {
  for (let step = 0; step < DARKEN_STEPS; step++) {
    const scale = 1 - step / DARKEN_STEPS;
    const shade: Rgb = [
      Math.round(color[0] * scale),
      Math.round(color[1] * scale),
      Math.round(color[2] * scale),
    ];
    if (1.05 / (relativeLuminance(shade) + 0.05) >= MIN_CONTRAST_WITH_WHITE) {
      return shade;
    }
  }
  return [0, 0, 0];
}

export interface AgentAccentAttributes {
  "data-agent-accent"?: "";
  style?: CSSProperties;
}

/** Attributes for the element that wraps a conversation with one agent. When
 * the agent's avatar has a color, elements inside that carry an
 * `agent-accent-*` class (see globals.css) take it on; otherwise this is
 * empty and they keep the theme's colors. */
export function useAgentAccentAttributes(
  persona: AvatarPersona | null | undefined,
): AgentAccentAttributes {
  const color = usePersonaAccentColor(persona);
  const [red, green, blue] = color ?? [];
  return useMemo(() => {
    if (red === undefined || green === undefined || blue === undefined) return {};
    return {
      "data-agent-accent": "",
      style: {
        "--agent-accent": `rgb(${readableAccent([red, green, blue]).join(" ")})`,
      } as CSSProperties,
    };
  }, [red, green, blue]);
}
