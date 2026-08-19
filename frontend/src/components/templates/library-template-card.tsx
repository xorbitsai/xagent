"use client";

import React, { type KeyboardEvent, type MouseEvent } from "react";
import { Clock, Heart, Loader2, Play, Users } from "lucide-react";
import type { Template } from "@/types/template";
import { cn } from "@/lib/utils";
import { isNestedInteractiveElement } from "./template-card-utils";
import { PersonaAvatar } from "./persona-avatar";

interface LibraryTemplateCardProps {
  template: Template;
  categoryLabel?: string;
  useLabel: string;
  defaultSetupTime: string;
  workforceBadgeLabel?: string;
  formatAgentsCount?: (count: number) => string;
  /** Opens the AI Team Marketplace detail page for a persona-bearing,
   * non-workforce template instead of instantiating anything directly -
   * `onOpen` and `formatMeetLabel` are bundled into one prop so a caller
   * can't wire one without the other (a "Meet {name}" button that silently
   * calls `onUse` instead of navigating, or vice versa). Falls back to
   * `onUse` when omitted entirely, so this component still works without it. */
  onOpenPersona?: {
    onOpen: (templateId: string) => void;
    formatMeetLabel: (name: string) => string;
  };
  /** True while this specific template's "Use" action is in flight (currently only
   * meaningful for workforce templates, which create real records server-side and
   * can take a few seconds). Disables activation and swaps the button to a spinner
   * so a slow request can't be re-triggered by an impatient repeat click. */
  isBusy?: boolean;
  busyLabel?: string;
  /** True when a *different* workforce template is currently being created
   * (creatingWorkforceId is a single global lock, not per-card - see
   * templates/page.tsx). Blocks activation like isBusy, but without
   * claiming this card itself is the one in flight: no spinner, no label
   * swap, just a visibly disabled state instead of a click that silently
   * does nothing. Does NOT apply to a persona card routed via `onOpenPersona`:
   * opening the marketplace detail page is pure client-side navigation with
   * no server request to race against the in-flight workforce creation. */
  disabled?: boolean;
  /** Instantiates the template directly: a workforce (use-as-workforce) or,
   * as a back-compat fallback, an agent-type template with no persona (the
   * old "jump into the builder with this template prefilled" path). */
  onUse: (templateId: string) => void;
  onLike?: (templateId: string, event: MouseEvent<HTMLButtonElement>) => void;
  className?: string;
}

// Soft pill palette keyed by category, with dark-mode variants so pills work
// in both themes. Unknown categories hash into the palette so colors stay
// stable across renders. The colored dot inherits the text color via bg-current.
const PILL_PALETTE = [
  "bg-blue-50 text-blue-700 dark:bg-blue-950/50 dark:text-blue-300",
  "bg-pink-50 text-pink-700 dark:bg-pink-950/50 dark:text-pink-300",
  "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300",
  "bg-indigo-50 text-indigo-700 dark:bg-indigo-950/50 dark:text-indigo-300",
  "bg-orange-50 text-orange-700 dark:bg-orange-950/50 dark:text-orange-300",
];
const PILL_NEUTRAL = "bg-muted text-muted-foreground";
const PILL_KNOWN: Record<string, string> = {
  sales: PILL_PALETTE[2],
  marketing: PILL_PALETTE[1],
  support: PILL_PALETTE[0],
  research: PILL_PALETTE[3],
  productivity: PILL_PALETTE[4],
};

function pillClasses(category?: string): string {
  if (!category) return PILL_NEUTRAL;
  const key = category.toLowerCase();
  if (PILL_KNOWN[key]) return PILL_KNOWN[key];
  if (key === "general" || key === "others") return PILL_NEUTRAL;
  let hash = 0;
  for (let i = 0; i < key.length; i++) hash = (hash * 31 + key.charCodeAt(i)) >>> 0;
  return PILL_PALETTE[hash % PILL_PALETTE.length];
}

function LibraryConnections({ template }: { template: Template }) {
  const visibleConnections = template.connections?.slice(0, 4) || [];
  const remainingCount = Math.max((template.connections?.length || 0) - visibleConnections.length, 0);

  return (
    <div className="flex items-center gap-1.5">
      {visibleConnections.map((connection, index) => (
        <div
          key={`${connection.name}-${index}`}
          className="flex h-[26px] w-[26px] items-center justify-center overflow-hidden rounded-lg bg-muted text-muted-foreground"
        >
          {connection.logo ? (
            <img src={connection.logo} alt={connection.name} className="h-4 w-4 object-contain" />
          ) : (
            <span className="text-[11px] font-bold">
              {(connection.name || "").substring(0, 1).toUpperCase()}
            </span>
          )}
        </div>
      ))}
      {remainingCount > 0 ? (
        <div className="flex h-[26px] w-[26px] items-center justify-center rounded-lg bg-muted text-[10px] font-semibold text-muted-foreground">
          +{remainingCount}
        </div>
      ) : null}
    </div>
  );
}

export function LibraryTemplateCard({
  template,
  categoryLabel,
  useLabel,
  defaultSetupTime,
  workforceBadgeLabel,
  formatAgentsCount,
  onOpenPersona,
  isBusy,
  busyLabel,
  disabled,
  onUse,
  onLike,
  className,
}: LibraryTemplateCardProps) {
  const isWorkforce = template.type === "workforce";
  const persona = !isWorkforce ? template.persona : null;
  // Opening the detail page is pure client-side navigation - it has no
  // server request to race against a sibling workforce's in-flight
  // creation, so the cross-card `disabled` lock (see the prop doc above)
  // must not immobilize this card while that's happening.
  const isNavigationOnly = Boolean(persona && onOpenPersona);
  const isBlocked = isBusy || (disabled && !isNavigationOnly);
  const handleActivate = () => {
    if (isBlocked) return;
    if (persona && onOpenPersona) {
      onOpenPersona.onOpen(template.id);
      return;
    }
    onUse(template.id);
  };
  // Server-computed manager + workers total, matching what "Use" actually
  // creates.
  const totalAgentCount = template.agent_count || 0;

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (isNestedInteractiveElement(event.target, event.currentTarget)) {
      return;
    }

    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      handleActivate();
    }
  };

  const bullets = template.features && template.features.length > 0 ? template.features.slice(0, 3) : [];
  const pill = pillClasses(template.category);

  return (
    <div
      role="button"
      tabIndex={0}
      aria-disabled={isBlocked || undefined}
      aria-busy={isBusy || undefined}
      onClick={handleActivate}
      onKeyDown={handleKeyDown}
      className={cn(
        "group flex h-full cursor-pointer flex-col rounded-[18px] border border-border bg-card p-5 shadow-sm transition-all duration-300 ease-out hover:-translate-y-1 hover:border-transparent hover:shadow-[0_16px_40px_rgba(0,0,0,0.11)]",
        isBusy && "cursor-wait opacity-70",
        isBlocked && !isBusy && "cursor-not-allowed opacity-50",
        className
      )}
    >
      {/* Category pill + workforce badge + setup time */}
      <div className="mb-3.5 flex items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          <span
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full px-[9px] py-1 text-[11.5px] font-semibold",
              pill
            )}
          >
            <span className="h-[5px] w-[5px] rounded-full bg-current" />
            {categoryLabel || template.category}
          </span>
          {isWorkforce ? (
            <span className="inline-flex items-center gap-1 rounded-full bg-teal-50 px-[9px] py-1 text-[11.5px] font-semibold text-teal-700 dark:bg-teal-950/50 dark:text-teal-300">
              <Users className="h-3 w-3 flex-shrink-0" />
              {workforceBadgeLabel}
              {totalAgentCount > 0 && formatAgentsCount ? ` · ${formatAgentsCount(totalAgentCount)}` : ""}
            </span>
          ) : null}
        </div>
        {template.setup_time || defaultSetupTime ? (
          <span className="flex items-center gap-1 whitespace-nowrap text-[11.5px] font-medium text-muted-foreground">
            <Clock className="h-3 w-3 flex-shrink-0" />
            {template.setup_time || defaultSetupTime}
          </span>
        ) : null}
      </div>

      {persona ? (
        <div className="mb-2 flex items-center gap-3">
          <PersonaAvatar persona={persona} sizeClassName="h-11 w-11" textClassName="text-sm" />
          <div className="min-w-0">
            <h3 className="line-clamp-1 text-[16.5px] font-semibold leading-[1.25] tracking-[-0.015em] text-foreground">
              {persona.name}
            </h3>
            <p className="line-clamp-1 text-[12.5px] font-medium text-muted-foreground">
              {persona.role}
            </p>
          </div>
        </div>
      ) : (
        <h3 className="mb-2 line-clamp-2 text-[16.5px] font-semibold leading-[1.25] tracking-[-0.015em] text-foreground">
          {template.name}
        </h3>
      )}

      {bullets.length > 0 ? (
        <ul className="flex flex-1 flex-col gap-2 p-0">
          {bullets.map((item, index) => (
            <li
              key={`${template.id}-${index}`}
              className="flex gap-2 text-[13.5px] leading-[1.45] text-foreground/80"
            >
              <span className="flex-none text-muted-foreground/60">›</span>
              <span className="line-clamp-2">{item}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="flex-1 text-[13.5px] leading-[1.5] text-muted-foreground">{template.description}</p>
      )}

      {/* Footer: integrations + stats */}
      <div className="mt-[18px] flex items-center justify-between gap-2.5">
        <LibraryConnections template={template} />
        <div className="flex flex-shrink-0 items-center gap-3 text-muted-foreground">
          <span className="flex items-center gap-1">
            <Play className="h-2.5 w-2.5 flex-shrink-0 fill-current" />
            <span className="text-xs font-medium">{template.used_count ?? 0}</span>
          </span>
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onLike?.(template.id, event);
            }}
            className={cn(
              "flex items-center gap-1 border-none bg-transparent p-0",
              onLike ? "cursor-pointer" : "cursor-default"
            )}
          >
            <Heart className={cn("h-3 w-3 fill-current", template.is_liked ? "text-rose-500" : "text-rose-400/70")} />
            <span className="text-xs font-medium">{template.likes ?? 0}</span>
          </button>
        </div>
      </div>

      <button
        type="button"
        disabled={isBlocked}
        className={cn(
          "mt-4 flex h-[38px] items-center justify-center gap-1.5 rounded-[10px] bg-primary/10 text-[13.5px] font-semibold text-primary transition-all duration-300 hover:bg-primary hover:text-primary-foreground active:scale-[0.98] disabled:opacity-70 disabled:hover:bg-primary/10 disabled:hover:text-primary",
          isBusy ? "disabled:cursor-wait" : "disabled:cursor-not-allowed"
        )}
        onClick={(event) => {
          event.stopPropagation();
          handleActivate();
        }}
      >
        {isBusy ? (
          <>
            <Loader2 className="h-3.5 w-3.5 flex-shrink-0 animate-spin" />
            {busyLabel || useLabel}
          </>
        ) : persona && onOpenPersona ? (
          onOpenPersona.formatMeetLabel(persona.name)
        ) : (
          useLabel
        )}
      </button>
    </div>
  );
}
