"use client";

import { useState, useEffect } from "react";
import { Presentation, Search, Smartphone, Wand2 } from "lucide-react";
import { useI18n } from "@/contexts/i18n-context";
import { useApp } from "@/contexts/app-context-chat";
import { ChatStartScreen } from "@/components/chat/ChatStartScreen";
import { FilePreviewDialog } from "@/components/file/file-preview-dialog";
import { getBrandingFromEnv } from "@/lib/branding";

function TaskHomePageContent() {
  const { t } = useI18n();
  const { sendMessage, state, dispatch, closeFilePreview } = useApp();
  const [files, setFiles] = useState<File[]>([]);
  const [inputValue, setInputValue] = useState("");
  const branding = getBrandingFromEnv();

  // Clear state on mount to ensure we are in "new task" mode
  useEffect(() => {
    dispatch({ type: "RESET_STATE" });
  }, [dispatch]);

  const samplePrompts = [
    {
      icon: Search,
      title: t("chatPage.cards.research.title"),
      prompt: t("chatPage.cards.research.prompt"),
    },
    {
      icon: Smartphone,
      title: t("chatPage.cards.linkedin.title"),
      prompt: t("chatPage.cards.linkedin.prompt"),
    },
    {
      icon: Wand2,
      title: t("chatPage.cards.poster.title"),
      prompt: t("chatPage.cards.poster.prompt"),
    },
    {
      icon: Search,
      title: t("chatPage.cards.compare.title"),
      prompt: t("chatPage.cards.compare.prompt"),
    },
    {
      icon: Wand2,
      title: t("chatPage.cards.visual.title"),
      prompt: t("chatPage.cards.visual.prompt"),
    },
    {
      icon: Presentation,
      title: t("chatPage.cards.presentation.title"),
      prompt: t("chatPage.cards.presentation.prompt"),
    }
  ];

  const handleSend = async (message: string, filesToSend: File[], config?: any) => {
    if (state.isProcessing) return;

    // Use sendMessage from AppContext - it will create task and send files via WebSocket
    await sendMessage(message, config, filesToSend || files);

    // Clear files after sending
    setFiles([]);
    setInputValue("");
  };

  return (
    <div className="h-full bg-background flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto">
        <main className="container max-w-4xl mx-auto px-4 py-8">
          <ChatStartScreen
            title={t("chatPage.page.emptyTitle", { appName: branding.appName })}
            description={t("chatPage.page.emptyDescription")}
            prompts={samplePrompts}
            onSend={handleSend}
            isSending={state.isProcessing}
            files={files}
            onFilesChange={setFiles}
            inputValue={inputValue}
            onInputChange={setInputValue}
            showModeToggle={true}
            autoFocus={true}
            inputMinHeightClass="min-h-[200px]"
          />
        </main>
      </div>

      {/* File Preview Modal */}
      <FilePreviewDialog
        open={state.filePreview.isOpen}
        onOpenChange={(open) => {
          if (!open) closeFilePreview()
        }}
      />
    </div>
  );
}

export default TaskHomePageContent;
