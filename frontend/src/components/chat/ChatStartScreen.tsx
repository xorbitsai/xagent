import React from "react";
import { Bot, Sparkles } from "lucide-react";
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
  icon?: React.ReactNode | string; // URL string or ReactNode
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
}

export function ChatStartScreen({
  title,
  description,
  icon,
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
  autoFocus = false
}: ChatStartScreenProps) {
  const { t } = useI18n();

  const handlePromptClick = (prompt: string) => {
    if (onInputChange) {
      onInputChange(prompt);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] py-16 text-center">
      <h2 className="text-3xl font-semibold mb-3 text-foreground">
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
          />
          <div className="text-xs text-muted-foreground/60 text-center">
            {t("chatPage.input.hintEnter")} · {t("chatPage.input.hintAt")}
          </div>
        </div>

        {prompts && prompts.length > 0 && (
          <div className="space-y-4 pt-4">
            <div className={`grid grid-cols-1 sm:grid-cols-2 ${prompts.length <= 3 ? 'lg:grid-cols-3' : 'lg:grid-cols-4'} gap-4`}>
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
                    className="group relative p-4 h-28 rounded-xl border border-border bg-card hover:bg-muted/50 cursor-pointer transition-all duration-300 flex flex-col justify-center items-start text-left gap-3"
                  >
                    <div className="flex items-center justify-center">
                      {item.icon && <item.icon className="w-5 h-5 text-muted-foreground" />}
                    </div>
                    <h3 className="font-medium text-sm text-foreground/90 leading-tight">{item.title}</h3>
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
