
import React, { useEffect, useState } from 'react';
import { getApiUrl } from '@/lib/utils';
import { apiRequest } from '@/lib/api-wrapper';
import { Sparkles } from 'lucide-react';
import { useI18n } from '@/contexts/i18n-context';

interface TokenUsageDisplayProps {
  taskId: number | null;
  isRunning: boolean;
  className?: string;
}

interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  llm_calls: number;
}

export function TokenUsageDisplay({ taskId, isRunning, className }: TokenUsageDisplayProps) {
  const [usage, setUsage] = useState<TokenUsage | null>(null);
  const { t } = useI18n();

  useEffect(() => {
    if (!taskId) return;

    let isMounted = true;
    let intervalId: NodeJS.Timeout;

    const fetchUsage = async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/chat/task/${taskId}`);
        if (response.ok && isMounted) {
          const data = await response.json();
          setUsage({
            input_tokens: data.input_tokens || 0,
            output_tokens: data.output_tokens || 0,
            total_tokens: data.total_tokens || 0,
            llm_calls: data.llm_calls || 0,
          });
        }
      } catch (error) {
        console.error('Failed to fetch token usage:', error);
      }
    };

    // Initial fetch
    fetchUsage();

    // Poll if running
    if (isRunning) {
      intervalId = setInterval(fetchUsage, 15000); // Poll every 15 seconds
    }

    return () => {
      isMounted = false;
      if (intervalId) clearInterval(intervalId);
    };
  }, [taskId, isRunning]);

  if (!usage) return null;

  return (
    <div className={`inline-flex flex-wrap items-center gap-x-3 gap-y-1 rounded-xl border bg-card/80 px-3 py-2 text-xs sm:text-sm ${className || ""}`}>
      <span className="flex items-center gap-1.5 whitespace-nowrap">
        <Sparkles className="w-4 h-4 text-indigo-500" />
        <span className="font-medium text-foreground">{usage.input_tokens.toLocaleString()}</span>
        <span className="text-muted-foreground">{t('chatPage.tokenUsage.input')}</span>
      </span>
      <span className="flex items-center gap-1.5 whitespace-nowrap">
        <span className="font-medium text-foreground">{usage.output_tokens.toLocaleString()}</span>
        <span className="text-muted-foreground">{t('chatPage.tokenUsage.output')}</span>
      </span>
    </div>
  );
}
