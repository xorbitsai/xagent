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
}

export interface TemplateDetail extends Template {
  agent_config: AgentConfig;
}

export type TemplateWithStats = Template;
