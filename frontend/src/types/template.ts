// Template types

export interface AgentConfig {
  instructions: string;
  skills: string[];
  tool_categories: string[];
}

export interface Template {
  id: string;
  name: string;
  category: string;
  featured?: boolean;
  description: string;
  features: string[];
  setup_time: string;
  tags: string[];
  author: string;
  version: string;
  views: number;
  likes: number;
  used_count: number;
}

export interface TemplateDetail extends Template {
  agent_config: AgentConfig;
}

export interface TemplateWithStats extends Template {
  is_liked?: boolean;
}
