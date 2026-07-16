"use client";

import { useState, type KeyboardEvent, type MouseEvent } from "react";
import { Clock, Heart, Play } from "lucide-react";
import type { Template } from "@/types/template";
import { cn } from "@/lib/utils";
import { isNestedInteractiveElement } from "./template-card-utils";

interface LibraryTemplateCardProps {
  template: Template;
  categoryLabel?: string;
  useLabel: string;
  defaultSetupTime: string;
  onUse: (templateId: string) => void;
  onLike?: (templateId: string, event: MouseEvent<HTMLButtonElement>) => void;
  className?: string;
}

function LibraryConnections({ template }: { template: Template }) {
  const visibleConnections = template.connections?.slice(0, 4) || [];
  const remainingCount = Math.max((template.connections?.length || 0) - visibleConnections.length, 0);

  return (
    <div className="flex items-center gap-1">
      {visibleConnections.map((connection, index) => (
        <div
          key={`${connection.name}-${index}`}
          className="flex h-6 w-6 items-center justify-center overflow-hidden rounded-md"
        >
          {connection.logo ? (
            <img src={connection.logo} alt={connection.name} className="h-4 w-4 object-contain" />
          ) : (
            <span className="text-[8px] font-bold text-white">
              {(connection.name || "").substring(0, 1).toUpperCase()}
            </span>
          )}
        </div>
      ))}
      {remainingCount > 0 ? (
        <div className="flex h-6 w-6 items-center justify-center rounded-md bg-[#EEF2FF] text-[9px] font-semibold text-[#64748B]">
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
  onUse,
  onLike,
  className,
}: LibraryTemplateCardProps) {
  const [isUseButtonHovered, setIsUseButtonHovered] = useState(false);
  const handleActivate = () => onUse(template.id);

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (isNestedInteractiveElement(event.target, event.currentTarget)) {
      return;
    }

    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      handleActivate();
    }
  };

  const items =
    template.features && template.features.length > 0
      ? template.features.slice(0, 3)
      : [template.description];

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={handleActivate}
      onKeyDown={handleKeyDown}
      className={cn(
        "group relative flex cursor-pointer flex-col overflow-hidden rounded-[14px] border border-[rgba(60,131,246,0.22)] bg-[hsl(217_85%_56%/0.03)] transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_10px_32px_rgba(0,0,0,0.11),0_2px_8px_rgba(0,0,0,0.06)]",
        className
      )}
    >
      {/* 3px blue top strip — always blue regardless of category */}
      <div className="h-[3px] flex-shrink-0 bg-[rgb(60,131,246)]" />

      <div className="flex flex-1 flex-col p-[18px_20px_16px]">
        {/* Category + setup time */}
        <div className="mb-[10px] flex items-center justify-between">
          <span className="text-[10px] font-bold uppercase tracking-[0.08em] text-[rgb(60,131,246)]">
            {categoryLabel || template.category}
          </span>
          <div className="flex items-center gap-[3px] whitespace-nowrap text-[10.5px] text-muted-foreground">
            <Clock className="h-[11px] w-[11px] flex-shrink-0" />
            <span>{template.setup_time || defaultSetupTime}</span>
          </div>
        </div>

        <h3 className="mb-[10px] text-[14px] font-bold leading-[1.3] tracking-[-0.02em] text-foreground">
          {template.name}
        </h3>

        {/* Bullet list */}
        <ul className="mb-[14px] flex flex-1 flex-col gap-[5px] p-0">
          {items.map((item, index) => (
            <li
              key={`${template.id}-${index}`}
              className="relative line-clamp-2 pl-3 text-[12px] leading-[1.45] text-muted-foreground"
            >
              <span className="absolute left-0 font-bold leading-[1.3] text-[rgb(47,121,238)]">›</span>
              {item}
            </li>
          ))}
        </ul>

        {/* Footer */}
        <div className="mt-auto border-t border-[rgba(60,131,246,0.08)] pt-[10px]">
          <div className="flex items-center justify-between gap-2">
            <LibraryConnections template={template} />
            <div className="flex flex-shrink-0 items-center gap-[10px]">
              <div className="flex items-center gap-1 text-[rgba(60,131,246,0.55)]">
                <Play className="h-[10px] w-[10px] flex-shrink-0 fill-current" />
                <span className="text-[11px] font-bold text-muted-foreground">{template.used_count ?? 0}</span>
              </div>
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  onLike?.(template.id, event);
                }}
                className={cn(
                  "flex items-center gap-1 border-none bg-transparent p-0",
                  onLike ? "cursor-pointer" : "cursor-default",
                  template.is_liked ? "text-pink-500" : "text-[rgba(60,131,246,0.55)]"
                )}
              >
                <Heart
                  className={cn(
                    "h-[11px] w-[11px] fill-current",
                    template.is_liked ? "text-rose-500" : "text-rose-400/70"
                  )}
                />
                <span className="text-[11px] font-bold text-muted-foreground">{template.likes ?? 0}</span>
              </button>
            </div>
          </div>

          <button
            type="button"
            className={cn(
              "mt-[14px] w-full rounded-lg py-[7px] text-[11.5px] font-semibold uppercase tracking-[0.04em] transition-all duration-200",
              isUseButtonHovered
                ? "border border-transparent bg-[linear-gradient(135deg,rgb(48,64,207),rgb(60,131,246))] text-white"
                : "border border-[rgba(60,131,246,0.28)] bg-transparent text-[rgb(60,131,246)]"
            )}
            onClick={(event) => {
              event.stopPropagation();
              handleActivate();
            }}
            onMouseEnter={() => setIsUseButtonHovered(true)}
            onMouseLeave={() => setIsUseButtonHovered(false)}
          >
            {useLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
