"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import { Loader2 } from "lucide-react";

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    const checkModelsAndRedirect = async () => {
      try {
        const apiUrl = getApiUrl();
        const response = await apiRequest(`${apiUrl}/api/models/`);
        if (response.ok) {
          const models = await response.json();
          // Redirect to /models if there are no models available
          if (Array.isArray(models) && models.length === 0) {
            router.replace("/models");
            return;
          }
        }
      } catch (error) {
        console.error("Failed to fetch models for redirection:", error);
      }

      // Redirect to /task by default or if models exist
      router.replace("/task");
    };

    checkModelsAndRedirect();
  }, [router]);

  return (
    <div className="h-screen w-screen flex items-center justify-center bg-background">
      <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
    </div>
  );
}
