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

/** AI Team Marketplace card content for a template: display name, avatar,
 * and the chat-opening intro/questions a Hire flow seeds. Absent for
 * templates with no marketplace persona (e.g. a workforce-type template). */
export interface PersonaInfo {
  name: string;
  role: string;
  avatar?: string | null;
  intro: string;
  kickoff_questions: string[];
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
  persona?: PersonaInfo | null;
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
  /** Whether the current user already has a quick-access agent instance of this template. */
  hired?: boolean;
  /** ID of the current user's quick-access agent instance of this template, if `hired` is true. */
  hired_agent_id?: number | null;
}

export interface TemplateDetail extends Template {
  /** Populated for an "agent"-type template. Null/absent for "workforce"-type templates,
   * which are configured via `workforce_config` instead. */
  agent_config?: AgentConfig | null;
  workforce_config?: Record<string, unknown> | null;
}

export type TemplateWithStats = Template;
