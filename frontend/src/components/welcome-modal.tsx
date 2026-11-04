"use client";

import { useEffect, useState } from "react";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
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
                className="max-h-[90vh] max-w-[90vw] overflow-y-auto rounded-2xl p-4 sm:rounded-3xl sm:p-6 md:max-w-[900px] md:p-12 lg:max-w-[1000px] gap-0"
            >
                <div className="mb-6 flex flex-col items-center text-center sm:mb-8 md:mb-10">
                    <span className="mb-3 text-[11px] font-bold uppercase tracking-[0.2em] text-primary sm:mb-4">
                        {t("dashboard.welcome.title", { appName: branding.appName.toUpperCase() })}
                    </span>
                    <DialogTitle className="mb-3 text-2xl font-extrabold tracking-tight text-foreground sm:text-3xl md:text-4xl">
                        {t("dashboard.welcome.heading")}
                    </DialogTitle>
                    <p className="text-[14px] text-muted-foreground sm:text-[15px]">
                        {t("dashboard.welcome.subtitle")}
                    </p>
                </div>

                <div className="mb-6 grid grid-cols-1 gap-4 sm:mb-8 sm:gap-5 md:mb-10 md:grid-cols-3 md:gap-6">
                    {/* Card 1: Presentation Builder */}
                    <button
                        onClick={() => handleCardClick("/task?starter=presentation")}
                        className="group flex flex-col text-left bg-card rounded-2xl border border-border/60 hover:border-primary/50 hover:shadow-[0_8px_30px_rgb(0,0,0,0.12)] dark:hover:shadow-[0_8px_30px_rgba(255,255,255,0.05)] transition-all duration-300 overflow-hidden"
                    >
                        <div className="h-[150px] w-full overflow-hidden sm:h-[180px]">
                            <img src="/home_create_a_presentation.webp" alt={t("dashboard.welcome.presentationBuilder.title")} className="w-full h-full object-cover" />
                        </div>
                        <div className="p-4 sm:p-6">
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
                        <div className="h-[150px] w-full overflow-hidden sm:h-[180px]">
                            <img src="/home_chat_with_agents.webp" alt={t("dashboard.welcome.buildAgents.title")} className="w-full h-full object-cover" />
                        </div>
                        <div className="p-4 sm:p-6">
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
                        onClick={() => handleCardClick("/build?create=true")}
                        className="group flex flex-col text-left bg-card rounded-2xl border border-border/60 hover:border-primary/50 hover:shadow-[0_8px_30px_rgb(0,0,0,0.12)] dark:hover:shadow-[0_8px_30px_rgba(255,255,255,0.05)] transition-all duration-300 overflow-hidden"
                    >
                        <div className="h-[150px] w-full overflow-hidden sm:h-[180px]">
                            <img src="/home_build_your_own_agents.png" alt={t("dashboard.welcome.createAgent.title")} className="w-full h-full object-cover" />
                        </div>
                        <div className="p-4 sm:p-6">
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
