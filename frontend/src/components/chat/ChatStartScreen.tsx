import React from "react";
import { Sparkles } from "lucide-react";
import { ChatInput } from "@/components/chat/ChatInput";
import { useI18n } from "@/contexts/i18n-context";

export interface PromptCard {
  icon?: any;
  title?: string;
  description?: string;
  prompt: string;
  color?: string;
  bg?: string;
}

interface ChatStartScreenProps {
  title: string;
  description?: string;
  prompts?: (PromptCard | string)[];
  onSend: (message: string, files: File[], config?: any) => void;
  isSending?: boolean;
  inputValue?: string;
  onInputChange?: (value: string) => void;
  files?: File[];
  onFilesChange?: (files: File[]) => void;
  showModeToggle?: boolean;
  readOnlyConfig?: boolean;
  taskConfig?: any;
  autoFocus?: boolean;
  inputMinHeightClass?: string;
}

export function ChatStartScreen({
  title,
  description,
  prompts,
  onSend,
  isSending = false,
  inputValue,
  onInputChange,
  files = [],
  onFilesChange,
  showModeToggle = false,
  readOnlyConfig = false,
  taskConfig,
  autoFocus = false,
  inputMinHeightClass
}: ChatStartScreenProps) {
  const { t } = useI18n();

  const handlePromptClick = (prompt: string) => {
    if (onInputChange) {
      onInputChange(prompt);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] py-16 text-center">
      <h2 className="text-3xl font-bold mb-3 text-blue-600 dark:text-blue-500">
        {title}
      </h2>
      {description && (
        <p className="text-base text-muted-foreground mb-10 max-w-md">{description}</p>
      )}

      <div className="w-full max-w-3xl mx-auto space-y-8">
        <div className="space-y-4">
          <ChatInput
            onSend={(msg, config) => onSend(msg, files, config)}
            isLoading={isSending}
            files={files}
            onFilesChange={onFilesChange || (() => { })}
            showModeToggle={showModeToggle}
            inputValue={inputValue}
            onInputChange={onInputChange}
            readOnlyConfig={readOnlyConfig}
            taskConfig={taskConfig}
            autoFocus={autoFocus}
            minHeightClass={inputMinHeightClass}
          />
        </div>

        {prompts && prompts.length > 0 && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 mt-4 px-1">
              <Sparkles className="w-3.5 h-3.5" />
              <span>{t("chatPage.sections.startingPrompts")}</span>
            </div>
            <div className={`grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4`}>
              {prompts.map((item, index) => {
                const isString = typeof item === 'string';
                const promptText = isString ? item : item.prompt;

                if (isString) {
                  return (
                    <div
                      key={index}
                      onClick={() => handlePromptClick(promptText)}
                      className="group relative p-4 h-28 rounded-xl border border-border bg-card hover:bg-muted/50 cursor-pointer transition-all duration-300 flex flex-col justify-center text-left"
                    >
                      <p className="text-sm text-foreground/90 line-clamp-3">{promptText}</p>
                    </div>
                  );
                }

                // Card style for Task Page
                return (
                  <div
                    key={index}
                    onClick={() => handlePromptClick(promptText)}
                    className="group relative px-4 py-3 min-h-[72px] rounded-xl border border-border bg-card hover:bg-muted/50 cursor-pointer transition-all duration-300 flex flex-row items-center text-left gap-4"
                  >
                    <div className="flex items-center justify-center shrink-0 h-10 w-10 rounded-lg bg-blue-50 dark:bg-blue-900/20 text-blue-500">
                      {item.icon && <item.icon className="w-5 h-5" />}
                    </div>
                    <h3 className="font-medium text-[14px] text-foreground/90 leading-snug">{item.title}</h3>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
