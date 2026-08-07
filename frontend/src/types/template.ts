// Template types

export interface AgentConfig {
  instructions: string;
  skills: string[];
  tool_categories: string[];
  execution_mode?: "flash" | "balanced" | "think";
}

export interface ConnectionInfo {
  name: string;
  logo?: string;
}

export interface SamplePrompt {
  title: string;
  prompt: string;
  highlights?: string[];
}

export type TemplateType = "agent" | "workforce";

export interface Template {
  id: string;
  name: string;
  category: string;
  featured?: boolean;
  description: string;
  features: string[];
  sample_prompts?: SamplePrompt[];
  connections: ConnectionInfo[];
  setup_time: string;
  tags: string[];
  author: string;
  version: string;
  views: number;
  likes: number;
  used_count: number;
  is_liked?: boolean;
  /** "agent" (default) for a single-agent template, "workforce" for a manager + worker-agents template. */
  type?: TemplateType;
  /** Total agents (manager + workers) a "workforce"-type template creates. 0 for "agent"-type templates. */
  agent_count?: number;
}

export interface TemplateDetail extends Template {
  /** Populated for an "agent"-type template. Null/absent for "workforce"-type templates,
   * which are configured via `workforce_config` instead. */
  agent_config?: AgentConfig | null;
  workforce_config?: Record<string, unknown> | null;
}

export type TemplateWithStats = Template;
