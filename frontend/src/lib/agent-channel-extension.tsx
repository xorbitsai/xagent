import type {
  AgentChannelHeaderActionComponent,
  AgentChannelOptionComponent,
  AgentChannelViewComponent,
} from "@/lib/page-extension-contracts"

// Extension point for an extra deployment channel. The OSS build has none, so
// every piece renders nothing and the Deploy dialog never enters the view.
export const AgentChannelHeaderAction: AgentChannelHeaderActionComponent = () => null

export const AgentChannelOption: AgentChannelOptionComponent = () => null

export const AgentChannelView: AgentChannelViewComponent = () => null
