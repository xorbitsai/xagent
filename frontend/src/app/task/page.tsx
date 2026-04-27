"use client";

import { useState, useEffect } from "react";
import { Bot, Presentation, BarChart, Image as ImageIcon, Zap } from "lucide-react";
import { useI18n } from "@/contexts/i18n-context";
import { useApp } from "@/contexts/app-context-chat";
import { ChatStartScreen } from "@/components/chat/ChatStartScreen";
import { FilePreviewDialog } from "@/components/file/file-preview-dialog";

function TaskHomePageContent() {
  const { t } = useI18n();
  const { sendMessage, state, dispatch, closeFilePreview } = useApp();
  const [files, setFiles] = useState<File[]>([]);
  const [inputValue, setInputValue] = useState("");

  // Clear state on mount to ensure we are in "new task" mode
  useEffect(() => {
    dispatch({ type: "RESET_STATE" });
  }, [dispatch]);

  const samplePrompts = [
    {
      icon: Presentation,
      title: t("chatPage.cards.createPPT.title"),
      description: t("chatPage.cards.createPPT.description"),
      prompt: t("chatPage.cards.createPPT.prompt"),
      color: "text-orange-400",
      bg: "bg-orange-400/10"
    },
    {
      icon: BarChart,
      title: t("chatPage.cards.dataAnalysis.title"),
      description: t("chatPage.cards.dataAnalysis.description"),
      prompt: t("chatPage.cards.dataAnalysis.prompt"),
      color: "text-blue-400",
      bg: "bg-blue-400/10"
    },
    {
      icon: ImageIcon,
      title: t("chatPage.cards.designPoster.title"),
      description: t("chatPage.cards.designPoster.description"),
      prompt: t("chatPage.cards.designPoster.prompt"),
      color: "text-purple-400",
      bg: "bg-purple-400/10"
    },
    {
      icon: Zap,
      title: t("chatPage.cards.automatic.title"),
      description: t("chatPage.cards.automatic.description"),
      prompt: t("chatPage.cards.automatic.prompt"),
      color: "text-green-400",
      bg: "bg-green-400/10"
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
    <div className="h-screen bg-background flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto">
        <main className="container max-w-4xl mx-auto px-4 py-8">
          <ChatStartScreen
            title={t("chatPage.page.emptyTitle")}
            description={t("chatPage.page.emptyDescription", { appName: process.env.NEXT_PUBLIC_APP_NAME || "Xagent" })}
            icon={<Bot className="w-10 h-10 text-[hsl(var(--gradient-from))]" />}
            prompts={samplePrompts}
            onSend={handleSend}
            isSending={state.isProcessing}
            files={files}
            onFilesChange={setFiles}
            inputValue={inputValue}
            onInputChange={setInputValue}
            showModeToggle={true}
            autoFocus={true}
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
