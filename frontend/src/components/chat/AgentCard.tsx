"use client"

import React from "react";
import { Bot, ChevronRight } from "lucide-react";
import { useRouter, usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

// Base styles for agent card
export const agentCardBaseClasses =
  "inline-flex flex-col items-start gap-2 bg-gradient-to-br from-primary/5 to-primary/10 border border-primary/20 rounded-lg p-3 my-2 max-w-sm hover:border-primary/40 hover:from-primary/10 hover:to-primary/15 transition-all cursor-pointer group";

export interface AgentCardProps {
  agentId: string;
  agentName: string;
  description?: string;
  status?: "draft" | "published";
  className?: string;
  onClick?: () => void;
}

/**
 * AgentCard component for displaying created agents in a card format.
 * Shows agent name, status badge, and optional description.
 * Automatically navigates to /agent/[id] when clicked.
 */
export function AgentCard({
  agentId,
  agentName,
  description,
  status = "draft",
  className,
  onClick,
}: AgentCardProps) {
  const router = useRouter();
  const pathname = usePathname();

  const handleClick = () => {
    if (onClick) {
      // Use custom onClick if provided
      onClick();
    } else {
      // Default: navigate based on agent status
      // DRAFT agents go to builder (edit), PUBLISHED agents go to chat
      const targetUrl = status === 'draft' ? `/build/${agentId}` : `/agent/${agentId}`;

      // If we are already on the target URL, do nothing to prevent reloading the page
      // and losing the current chat history or unsaved form changes
      if (pathname === targetUrl) {
        return;
      }

      router.push(targetUrl);
    }
  };

  const statusBadge =
    status === "draft" ? (
      <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-500/30 text-yellow-700 border border-yellow-500/50">
        DRAFT
      </span>
    ) : (
      <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-500/20 text-green-300 border border-green-500/30">
        PUBLISHED
      </span>
    );

  return (
    <div
      className={cn(agentCardBaseClasses, "cursor-pointer", className)}
      onClick={handleClick}
      data-agent-id={agentId}
    >
      {/* Header: Icon + Name + Status */}
      <div className="flex items-center gap-2 w-full">
        <div className="flex items-center justify-center w-8 h-8 rounded-md bg-primary/20 group-hover:bg-primary/30 transition-colors">
          <Bot className="w-4 h-4 text-primary" />
        </div>
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <span className="text-sm font-semibold text-foreground truncate">
            {agentName}
          </span>
          {statusBadge}
        </div>
        <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground transition-colors flex-shrink-0" />
      </div>

      {/* Description (if provided) */}
      {description && (
        <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed">
          {description}
        </p>
      )}

      {/* Agent ID (subtle) */}
      <div className="text-[10px] text-muted-foreground/60 font-mono">
        ID: {agentId}
      </div>
    </div>
  );
}

/**
 * Compact version of AgentCard for inline display
 * Also navigates to /agent/[id] when clicked.
 */
export interface AgentChipProps {
  agentId: string;
  agentName: string;
  className?: string;
  onClick?: () => void;
}

export function AgentChip({
  agentId,
  agentName,
  className,
  onClick,
}: AgentChipProps) {
  const router = useRouter();

  const handleClick = () => {
    if (onClick) {
      // Use custom onClick if provided
      onClick();
    } else {
      // Default: navigate based on agent status
      // DRAFT agents go to builder (edit), PUBLISHED agents go to chat
      if (status === 'draft') {
        router.push(`/build/${agentId}`);
      } else {
        router.push(`/agent/${agentId}`);
      }
    }
  };

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 bg-primary/10 border border-primary/30 rounded-md px-2 py-1 text-sm hover:bg-primary/15 hover:border-primary/50 transition-all cursor-pointer group",
        className
      )}
      onClick={handleClick}
      data-agent-id={agentId}
    >
      <Bot className="w-3.5 h-3.5 text-primary" />
      <span className="text-xs font-medium text-foreground truncate max-w-[150px]">
        {agentName}
      </span>
      <ChevronRight className="w-3 h-3 text-muted-foreground group-hover:text-foreground transition-colors flex-shrink-0" />
    </span>
  );
}
