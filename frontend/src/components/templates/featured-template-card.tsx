"use client";

import type { KeyboardEvent, MouseEvent } from "react";
import { Heart, Play } from "lucide-react";
import type { Template } from "@/types/template";
import { cn } from "@/lib/utils";
import { isNestedInteractiveElement } from "./template-card-utils";

interface FeaturedTemplateCardProps {
  template: Template;
  categoryLabel?: string;
  onUse: (templateId: string) => void;
  onLike?: (templateId: string, event: MouseEvent<HTMLButtonElement>) => void;
  className?: string;
}

function FeaturedConnections({ template }: { template: Template }) {
  const visibleConnections = template.connections?.slice(0, 3) || [];

  return (
    <div className="flex items-center gap-1">
      {visibleConnections.map((connection, index) => (
        <div
          key={`${connection.name}-${index}`}
          className="flex h-5 w-5 items-center justify-center overflow-hidden rounded-md"
        >
          {connection.logo ? (
            <img src={connection.logo} alt={connection.name} className="h-3.5 w-3.5 object-contain" />
          ) : (
            <span className="text-[8px] font-bold text-white">
              {(connection.name || "").substring(0, 1).toUpperCase()}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

export function FeaturedTemplateCard({
  template,
  categoryLabel,
  onUse,
  onLike,
  className,
}: FeaturedTemplateCardProps) {
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

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={handleActivate}
      onKeyDown={handleKeyDown}
      className={cn(
        "group relative flex h-full min-w-[180px] flex-1 cursor-pointer flex-col justify-between gap-[10px] overflow-hidden rounded-[14px] bg-[rgb(243,244,252)] p-[18px_20px_16px] transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_8px_24px_rgba(60,131,246,0.18)]",
        className
      )}
    >
      {/* 3px gradient top strip */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[3px] rounded-t-[14px] bg-[linear-gradient(90deg,rgb(48,64,207),rgb(60,131,246),rgb(146,92,240))]" />

      <div className="relative z-[1] min-w-0 flex-1">
        <div className="mb-1.5 text-[9.5px] font-bold uppercase tracking-[0.08em] text-[rgb(60,131,246)]">
          {categoryLabel || template.category}
        </div>

        <h3 className="mb-[5px] line-clamp-2 text-[13px] font-bold leading-[1.3] text-foreground">
          {template.name}
        </h3>

        <p className="line-clamp-2 text-[11.5px] leading-[1.5] text-muted-foreground">
          {template.description}
        </p>
      </div>

      <div className="relative z-[1] mt-3 flex items-center gap-[10px] border-t border-border pt-[10px]">
        <FeaturedConnections template={template} />
        <div className="ml-auto flex items-center gap-[10px] text-[11px] font-semibold text-muted-foreground">
          <div className="flex items-center gap-1">
            <Play className="h-[10px] w-[10px] fill-current text-[rgba(60,131,246,0.55)]" />
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
              template.is_liked ? "text-pink-500" : "text-muted-foreground"
            )}
          >
            <Heart
              className={cn(
                "h-[11px] w-[11px] fill-current",
                template.is_liked ? "text-rose-500" : "text-rose-400/70"
              )}
            />
            <span className="text-[11px] font-bold">{template.likes ?? 0}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
