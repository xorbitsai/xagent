"use client";

import { useEffect, useState } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { useRouter } from "next/navigation";
import { useI18n } from "@/contexts/i18n-context";
import { getBrandingFromEnv } from "@/lib/branding";

export function WelcomeModal() {
    const branding = getBrandingFromEnv();
    const [open, setOpen] = useState(false);
    const router = useRouter();
    const { t } = useI18n();

    useEffect(() => {
        const hasVisited = localStorage.getItem("hasVisitedXagent");
        if (!hasVisited) {
            setOpen(true);
            localStorage.setItem("hasVisitedXagent", "true");
        }
    }, []);

    const handleClose = () => {
        setOpen(false);
    };

    const handleCardClick = (path: string) => {
        setOpen(false);
        router.push(path);
    };

    return (
        <Dialog open={open} onOpenChange={setOpen}>
            <DialogContent
                showCloseButton={false}
                onInteractOutside={(e) => e.preventDefault()}
                className="max-w-[90vw] md:max-w-[900px] lg:max-w-[1000px] p-8 md:p-12 rounded-3xl gap-0"
            >
                <div className="flex flex-col items-center text-center mb-10">
                    <span className="text-[11px] font-bold text-primary tracking-[0.2em] uppercase mb-4">
                        {t("dashboard.welcome.title", { appName: branding.appName.toUpperCase() })}
                    </span>
                    <h2 className="text-3xl md:text-4xl font-extrabold text-foreground mb-3 tracking-tight">
                        {t("dashboard.welcome.heading")}
                    </h2>
                    <p className="text-muted-foreground text-[15px]">
                        {t("dashboard.welcome.subtitle")}
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
                    {/* Card 1: Presentation Builder */}
                    <button
                        onClick={() => handleCardClick("/task")}
                        className="group flex flex-col text-left bg-card rounded-2xl border border-border/60 hover:border-primary/50 hover:shadow-[0_8px_30px_rgb(0,0,0,0.12)] dark:hover:shadow-[0_8px_30px_rgba(255,255,255,0.05)] transition-all duration-300 overflow-hidden"
                    >
                        <div className="h-[180px] w-full overflow-hidden">
                            <img src="/home_create_a_presentation.webp" alt="Presentation Builder" className="w-full h-full object-cover" />
                        </div>
                        <div className="p-6">
                            <h3 className="font-bold text-[16px] mb-2 text-foreground group-hover:text-primary transition-colors">
                                {t("dashboard.welcome.presentationBuilder.title")}
                            </h3>
                            <p className="text-[14px] text-muted-foreground leading-relaxed">
                                {t("dashboard.welcome.presentationBuilder.description")}
                            </p>
                        </div>
                    </button>

                    {/* Card 2: Build agents via Templates */}
                    <button
                        onClick={() => handleCardClick("/templates")}
                        className="group flex flex-col text-left bg-card rounded-2xl border border-border/60 hover:border-primary/50 hover:shadow-[0_8px_30px_rgb(0,0,0,0.12)] dark:hover:shadow-[0_8px_30px_rgba(255,255,255,0.05)] transition-all duration-300 overflow-hidden"
                    >
                        <div className="h-[180px] w-full overflow-hidden">
                            <img src="/home_chat_with_agents.webp" alt="Chat with Agents" className="w-full h-full object-cover" />
                        </div>
                        <div className="p-6">
                            <h3 className="font-bold text-[16px] mb-2 text-foreground group-hover:text-primary transition-colors">
                                {t("dashboard.welcome.buildAgents.title")}
                            </h3>
                            <p className="text-[14px] text-muted-foreground leading-relaxed">
                                {t("dashboard.welcome.buildAgents.description")}
                            </p>
                        </div>
                    </button>

                    {/* Card 3: Create Custom Agent */}
                    <button
                        onClick={() => handleCardClick("/build")}
                        className="group flex flex-col text-left bg-card rounded-2xl border border-border/60 hover:border-primary/50 hover:shadow-[0_8px_30px_rgb(0,0,0,0.12)] dark:hover:shadow-[0_8px_30px_rgba(255,255,255,0.05)] transition-all duration-300 overflow-hidden"
                    >
                        <div className="h-[180px] w-full overflow-hidden">
                            <img src="/home_build_your_own_agents.png" alt="Build Your Own Agents" className="w-full h-full object-cover" />
                        </div>
                        <div className="p-6">
                            <h3 className="font-bold text-[16px] mb-2 text-foreground group-hover:text-primary transition-colors">
                                {t("dashboard.welcome.createAgent.title")}
                            </h3>
                            <p className="text-[14px] text-muted-foreground leading-relaxed">
                                {t("dashboard.welcome.createAgent.description")}
                            </p>
                        </div>
                    </button>
                </div>

                <div className="flex justify-center">
                    <button
                        onClick={handleClose}
                        className="text-[14px] text-muted-foreground hover:text-foreground underline underline-offset-4 transition-colors"
                    >
                        {t("dashboard.welcome.skip")}
                    </button>
                </div>
            </DialogContent>
        </Dialog>
    );
}
