import React from "react";
import { Loader2, File as FileIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { FileItem } from "@/hooks/use-file-mention";
import type { Translate } from "@/contexts/i18n-context"

interface FileMentionDropdownProps {
  show: boolean;
  isLoading: boolean;
  filteredFiles: FileItem[];
  selectedFileIndex: number;
  onInsert: (file: FileItem) => void;
  t: Translate;
  position?: { top?: number; bottom?: number; left: number } | null;
}

export function FileMentionDropdown({
  show,
  isLoading,
  filteredFiles,
  selectedFileIndex,
  onInsert,
  t,
  position
}: FileMentionDropdownProps) {
  if (!show) return null;

  const style = position ? {
    top: position.top !== undefined ? `${position.top}px` : 'auto',
    bottom: position.bottom !== undefined ? `${position.bottom}px` : 'auto',
    left: `${position.left}px`,
  } : undefined;

  return (
    <div
      className={cn(
        "absolute z-50 w-full max-w-sm rounded-lg border bg-popover shadow-md overflow-hidden",
        !position && "bottom-full left-0 mb-2"
      )}
      style={style}
    >
      {isLoading ? (
        <div className="p-4 flex items-center justify-center text-sm text-muted-foreground">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          {t("common.loading")}
        </div>
      ) : filteredFiles.length === 0 ? (
        <div className="p-4 text-sm text-muted-foreground text-center">
          {t("files.table.empty.noMatch")}
        </div>
      ) : (
        <div className="max-h-[200px] overflow-y-auto p-1">
          {filteredFiles.map((file, index) => (
            <div
              key={index}
              className={cn(
                "flex items-center gap-2 px-3 py-2 text-sm rounded-md cursor-pointer transition-colors overflow-scroll",
                index === selectedFileIndex ? "bg-accent text-accent-foreground" : "hover:bg-muted"
              )}
              onClick={() => onInsert(file)}
              onMouseDown={(e) => e.preventDefault()}
            >
              <FileIcon className="h-4 w-4 shrink-0 text-muted-foreground" />
              <div className="flex flex-col items-start overflow-auto">
                <span className="truncate font-medium">{file.filename}</span>
                {file.relative_path && file.relative_path !== file.filename && (
                  <span className="truncate text-xs text-muted-foreground">{file.relative_path}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
