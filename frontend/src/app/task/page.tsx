"use client";

import { useState, useEffect } from "react";
import { Bot, Presentation, BarChart, Image as ImageIcon, Zap, Sparkles } from "lucide-react";
import { ChatInput } from "@/components/chat/ChatInput";
import { useI18n } from "@/contexts/i18n-context";
import { useRouter } from "next/navigation";
import { useApp } from "@/contexts/app-context-chat";

function TaskHomePageContent() {
  const { t } = useI18n();
  const router = useRouter();
  const { sendMessage, state, dispatch } = useApp();
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

  const handleSend = async (message: string, config?: any, filesToSend?: File[]) => {
    if (state.isProcessing) return;

    // Use sendMessage from AppContext - it will create task and send files via WebSocket
    await sendMessage(message, config, filesToSend || files);

    // Clear files after sending
    setFiles([]);
    setInputValue("");
  };

  const handlePromptClick = (prompt: string) => {
    setInputValue(prompt);
    // Note: Don't auto-send here, let user review and click send
    // Or we can call handleSend but need to handle async properly
  };

  return (
    <div className="h-screen bg-background flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto">
        <main className="container max-w-4xl mx-auto px-4 py-8">
          <div className="flex flex-col items-center justify-center min-h-[80vh] py-16 text-center">
            <div className="relative mb-6">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-[hsl(var(--gradient-from))]/20 to-[hsl(var(--gradient-to))]/10 flex items-center justify-center animate-float">
                <Bot className="w-10 h-10 text-[hsl(var(--gradient-from))]" />
              </div>
              <div className="absolute -inset-4 rounded-3xl bg-gradient-to-br from-primary/5 via-accent/5 to-transparent blur-xl -z-10" />
            </div>
            <h2 className="text-2xl font-bold mb-2 gradient-text">
              {t("chatPage.page.emptyTitle", { appName: process.env.NEXT_PUBLIC_APP_NAME || "Xagent" })}
            </h2>
            <p className="text-xs text-muted-foreground/70 mb-8">{t("chatPage.page.emptyDescription")}</p>

            <div className="w-full max-w-4xl mx-auto space-y-8">
              <ChatInput
                onSend={handleSend}
                isLoading={state.isProcessing}
                files={files}
                onFilesChange={setFiles}
                showModeToggle={true}
                inputValue={inputValue}
                onInputChange={setInputValue}
              />

              <div className="space-y-4">
                <div className="flex items-center gap-2 text-sm text-muted-foreground/80 px-1">
                  <Sparkles className="w-4 h-4" />
                  <span>{t("chatPage.page.startWith")}</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  {samplePrompts.map((card, index) => (
                    <div
                      key={index}
                      onClick={() => handlePromptClick(card.prompt)}
                      className="group relative p-4 h-32 rounded-xl border border-border/40 bg-card/30 hover:bg-card hover:border-primary/50 cursor-pointer transition-all duration-300 hover:-translate-y-1 hover:shadow-lg flex flex-col justify-between items-start text-left"
                    >
                      <div className={`w-10 h-10 rounded-lg ${card.bg} flex items-center justify-center group-hover:scale-110 transition-transform`}>
                        <card.icon className={`w-5 h-5 ${card.color}`} />
                      </div>
                      <div>
                        <h3 className="font-medium text-sm text-foreground/90">{card.title}</h3>
                        <p className="text-xs text-muted-foreground/70 mt-1">{card.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default TaskHomePageContent;
