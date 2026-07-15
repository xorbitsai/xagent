import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { FileTypeIcon } from "@/components/file/file-type-icon";

// Base styles shared between React component and HTML string
// Using the more detailed styles from ChatInput as the source of truth
export const fileChipBaseClasses =
  "relative top-[-2px] inline-flex items-center gap-1.5 align-middle bg-secondary/80 border border-border/50 rounded-md px-1.5 text-sm select-none group shadow-sm transition-all hover:bg-secondary hover:border-primary/30";

interface FileChipProps {
  path: string;
  filename?: string;
  mimeType?: string;
  className?: string;
  onDelete?: () => void;
  showDelete?: boolean;
  onClick?: () => void;
}

/**
 * React component for displaying a file chip.
 * Used in ChatMessage and other read-only views.
 */
export function FileChip({ path, filename, mimeType, className, onDelete, showDelete = false, onClick }: FileChipProps) {
  const displayFileName = filename || path.split('/').pop() || path;


  return (
    <span
      className={cn(fileChipBaseClasses, "mx-1 py-0.5", onClick && "cursor-pointer hover:bg-secondary/90", className)}
      onClick={onClick}
    >
      <FileTypeIcon
        filename={displayFileName}
        mimeType={mimeType}
        className="w-3.5 h-3.5 text-primary"
      />
      <span className="text-[11px] font-medium text-foreground/70 truncate max-w-[200px]">
        {displayFileName}
      </span>
      {showDelete && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onDelete?.();
          }}
          className="ml-1 text-muted-foreground hover:text-destructive transition-colors p-0.5 rounded-md hover:bg-destructive/10"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </span>
  );
}

/**
 * Generates HTML string for a file chip.
 * Used in contenteditable div in ChatInput.
 */
export const createFileChipHTML = (path: string, fileId?: string, filename?: string, className?: string, mimeType?: string) => {
  const displayFileName = filename || path.split('/').pop() || path;
  const iconHtml = renderToStaticMarkup(
    <FileTypeIcon
      filename={displayFileName}
      mimeType={mimeType}
      className="w-3.5 h-3.5 text-primary"
    />
  );
  const deleteIconHtml = renderToStaticMarkup(<X className="w-3.5 h-3.5 text-destructive cursor-pointer" />);

  // Note: specific styles for ChatInput context (relative positioning, margins)
  // These are slightly different from the React component default to work well in contenteditable
  const containerClass = cn(fileChipBaseClasses, "relative top-[-2px] mr-2 cursor-pointer hover:bg-secondary/90 file-chip-preview", className);

  return `
    <span contenteditable="false" data-file-path="${path}" data-file-id="${fileId || ''}" data-filename="${displayFileName}" class="${containerClass}">
      <span class="relative w-3.5 h-3.5 flex items-center justify-center">
        <span class="absolute inset-0 flex items-center justify-center transition-opacity group-hover:opacity-0">
          ${iconHtml}
        </span>
        <button type="button" class="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity file-chip-delete">
          ${deleteIconHtml}
        </button>
      </span>
      <span class="text-[11px] font-medium text-foreground/70 truncate max-w-[200px]">${displayFileName}</span>
    </span>&#8203;
  `.trim();
};
