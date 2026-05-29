"use client";

import { useState, type KeyboardEvent, type MouseEvent } from "react";
import { ChevronRight, Clock, Heart, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Template } from "@/types/template";
import { cn } from "@/lib/utils";
import { isNestedInteractiveElement } from "./template-card-utils";

interface LibraryTemplateCardProps {
  template: Template;
  categoryLabel?: string;
  useLabel: string;
  defaultSetupTime: string;
  accentColorClassName?: string;
  accentSoftClassName?: string;
  accentHex?: string;
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
  accentColorClassName = "bg-[#5B67FF]",
  accentSoftClassName = "text-[#5B67FF]",
  accentHex = "#5B67FF",
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
        "group flex min-h-[250px] cursor-pointer flex-col overflow-hidden rounded-2xl border border-[#E7EAF3] bg-white shadow-[0_1px_2px_rgba(15,23,42,0.04)] transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md",
        className
      )}
    >
      <div className={cn("h-1 w-full", accentColorClassName)} />

      <div className="flex flex-1 flex-col p-4">
        <div className="flex items-center justify-between gap-3">
          <span className={cn("text-[10.5px] font-bold uppercase tracking-[0.06em]", accentSoftClassName)}>
            {categoryLabel || template.category}
          </span>
          <div className="flex items-center gap-1 text-[11px] text-[#6B7280]">
            <Clock className="h-[11px] w-[11px]" />
            <span>{template.setup_time || defaultSetupTime}</span>
          </div>
        </div>

        <h3 className="mt-[10px] text-[15px] font-bold leading-[1.3] text-[#111827]">{template.name}</h3>

        <div className="mt-3 flex-1 space-y-[6px]">
          {items.map((item, index) => (
            <div key={`${template.id}-${index}`} className="flex items-start gap-[7px] text-[12.5px] leading-[1.45] text-[#374151]">
              <ChevronRight className={cn("mt-1 h-[11px] w-[11px] shrink-0", accentSoftClassName)} strokeWidth={2.4} />
              <span className="line-clamp-2 flex-1">{item}</span>
            </div>
          ))}
        </div>

        <div className="mt-[14px] border-t border-[#E5E7EB] pt-3">
          <div className="flex items-center justify-between gap-[10px]">
            <LibraryConnections template={template} />
            <div className="flex items-center gap-[10px] text-[11.5px] text-[#6B7280]">
              <div className="flex items-center gap-1">
                <Play className="h-[10px] w-[10px] fill-current text-[#64748B]" />
                <span>{template.used_count ?? 0}</span>
              </div>
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  onLike?.(template.id, event);
                }}
                className={cn(
                  "flex items-center gap-1 transition-colors",
                  template.is_liked ? "text-pink-500" : onLike ? "hover:text-pink-500" : "cursor-default"
                )}
              >
                <Heart
                  className={cn(
                    "h-[11px] w-[11px] fill-current",
                    template.is_liked ? "text-rose-500" : "text-rose-400/70"
                  )}
                />
                <span>{template.likes ?? 0}</span>
              </button>
            </div>
          </div>

          <Button
            type="button"
            variant="outline"
            className="mt-3 h-9 w-full rounded-lg border bg-transparent text-[12px] font-bold uppercase tracking-[0.04em] hover:text-inherit"
            style={{
              borderColor: isUseButtonHovered ? accentHex : `${accentHex}40`,
              backgroundColor: isUseButtonHovered ? `${accentHex}10` : "transparent",
              color: accentHex,
            }}
            onClick={(event) => {
              event.stopPropagation();
              handleActivate();
            }}
            onMouseEnter={() => setIsUseButtonHovered(true)}
            onMouseLeave={() => setIsUseButtonHovered(false)}
          >
            {useLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}
