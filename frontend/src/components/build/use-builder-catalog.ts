"use client"

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

// The agent builder's pickers are fed by independent endpoints. Each one loads
// and fails on its own (#3007): a slow or failing /api/tools/available must not
// hold the Model or Skills pickers, and a list may only say "no data" once it
// has actually loaded empty.

export type CatalogStatus = "loading" | "ready" | "error"

interface KnowledgeBase {
  name: string
  [key: string]: any
}

interface Skill {
  name: string
  description?: string
  when_to_use?: string
  tags?: string[]
  [key: string]: any
}

interface Tool {
  name: string
  description: string
  type: string
  category: string
  enabled: boolean
  always_available: boolean
  [key: string]: any
}

export interface Model {
  id: number
  model_id: string
  model_name: string
  model_provider: string
  category: string
  base_url?: string
  is_default?: boolean
  is_small_fast?: boolean
  is_visual?: boolean
  is_compact?: boolean
}

export interface AgentModelConfig {
  general: number | null
  small_fast: number | null
  visual: number | null
  compact: number | null
}

interface UseBuilderCatalogOptions {
  // Set by the builder's agent load when an admin views another user's agent;
  // the mount-time MCP fetch must then leave the owner-scoped list alone.
  ownerScopedMcpRef: MutableRefObject<boolean>
  // The builder's model config setter; used once, to seed a new agent from
  // the viewer's default models.
  setModelConfig: Dispatch<SetStateAction<AgentModelConfig>>
  isEditMode: boolean
}

// Every catalog endpoint returns a list; anything else is a malformed body,
// which must surface as an error rather than as an empty list.
function requireList<T>(value: unknown): T[] {
  if (!Array.isArray(value)) throw new Error("Expected a list in the response body")
  return value
}

// Runs one catalog request: `loading` while in flight, `ready` once applied,
// `error` on a non-OK status, a network failure or a malformed body.
async function loadList(
  path: string,
  setStatus: (status: CatalogStatus) => void,
  apply: (data: any) => void,
): Promise<void> {
  setStatus("loading")
  try {
    const res = await apiRequest(`${getApiUrl()}${path}`)
    if (!res.ok) throw new Error(`${path} returned ${res.status}`)
    apply(await res.json())
    setStatus("ready")
  } catch (error) {
    console.error(`Failed to load ${path}:`, error)
    setStatus("error")
  }
}

export function useBuilderCatalog({
  ownerScopedMcpRef,
  setModelConfig,
  isEditMode,
}: UseBuilderCatalogOptions) {
  const [models, setModels] = useState<Model[]>([])
  const [modelsStatus, setModelsStatus] = useState<CatalogStatus>("loading")
  const [kbs, setKbs] = useState<KnowledgeBase[]>([])
  const [kbsStatus, setKbsStatus] = useState<CatalogStatus>("loading")
  const [skills, setSkills] = useState<Skill[]>([])
  const [skillsStatus, setSkillsStatus] = useState<CatalogStatus>("loading")
  const [tools, setTools] = useState<Tool[]>([])
  const [intrinsicToolNames, setIntrinsicToolNames] = useState<string[]>([])
  const [skillLoaderTool, setSkillLoaderTool] = useState<string | null>(null)
  const [toolsStatus, setToolsStatus] = useState<CatalogStatus>("loading")
  const [mcpServers, setMcpServers] = useState<any[]>([])
  // The CURRENT user's general default (not necessarily the agent owner's).
  const [userDefaultGeneral, setUserDefaultGeneral] = useState<number | null>(null)
  // Latched once the mount-time model list and user default have both landed
  // (either way), so the edit-mode seed can tell an unset slot from a loading one.
  const [modelDefaultsSettled, setModelDefaultsSettled] = useState(false)
  // Latched once the mount-time MCP fetch has finished, either way.
  const [mcpSettled, setMcpSettled] = useState(false)

  // Mount-time values, as the single mount fetch below has always read them.
  const mountOptionsRef = useRef({ setModelConfig, isEditMode })

  const loadModels = useCallback(async (): Promise<Model[]> => {
    let loaded: Model[] = []
    await loadList("/api/models/?category=llm", setModelsStatus, (data) => {
      loaded = requireList<Model>(data)
      setModels(loaded)
    })
    return loaded
  }, [])

  const loadKbs = useCallback(async () => {
    await loadList("/api/kb/collections", setKbsStatus, (data) => {
      const collections = requireList<KnowledgeBase>(data?.collections)
      // A failed listing still answers 200, with status "error"; team KBs
      // merged in afterwards can make that list non-empty, which still shows.
      if (data?.status === "error" && collections.length === 0) {
        throw new Error(data?.message || "Knowledge base listing failed")
      }
      setKbs(collections)
    })
  }, [])

  const loadSkills = useCallback(async () => {
    await loadList("/api/skills/", setSkillsStatus, (data) => {
      setSkills(requireList<Skill>(data))
    })
  }, [])

  const loadTools = useCallback(async () => {
    await loadList("/api/tools/available?include_usage=false", setToolsStatus, (data) => {
      const all = requireList<Tool>(data?.tools)
      setTools(all.filter((t) => t.enabled))
      // Unfiltered: the runtime injects these without reading ToolConfig.enabled.
      setIntrinsicToolNames(all.filter((t) => t.always_available).map((t) => t.name))
      const loader = data?.skill_loader_tool
      setSkillLoaderTool(typeof loader === "string" && loader.trim() ? loader.trim() : null)
    })
  }, [])

  useEffect(() => {
    const loadModelDefaults = async () => {
      const modelsLoad = loadModels()
      try {
        const res = await apiRequest(`${getApiUrl()}/api/models/user-default`)
        if (!res.ok) return
        const userDefaults = await res.json()

        // The shape guards are defensive: a malformed entry is skipped rather
        // than failing the whole default.
        const config: AgentModelConfig = {
          general: null,
          small_fast: null,
          visual: null,
          compact: null,
        }
        for (const m of Array.isArray(userDefaults) ? userDefaults : []) {
          const id = m?.model?.id
          if (!id) continue
          if (m.config_type === "general") config.general = id
          else if (m.config_type === "small_fast") config.small_fast = id
          else if (m.config_type === "visual") config.visual = id
          else if (m.config_type === "compact") config.compact = id
        }
        setUserDefaultGeneral(config.general)

        // The fallback needs the model list, so a new agent's seed waits for it.
        const availableModels = await modelsLoad
        const { isEditMode: editing, setModelConfig: seed } = mountOptionsRef.current
        if (!editing) {
          // Fallback: no general default set, pick the first available LLM
          // (the models endpoint is queried with ?category=llm).
          if (!config.general && availableModels.length > 0) {
            config.general = availableModels[0].id
          }
          // The model list renders before the default lands, so fill only
          // the slots the user has not picked in the meantime.
          seed((prev) => ({
            general: prev.general ?? config.general,
            small_fast: prev.small_fast ?? config.small_fast,
            visual: prev.visual ?? config.visual,
            compact: prev.compact ?? config.compact,
          }))
        }
      } catch (error) {
        console.error("Failed to load default models:", error)
      } finally {
        await modelsLoad
        setModelDefaultsSettled(true)
      }
    }

    // MCP has no picker of its own: selected connectors render from it, so the
    // builder only needs to know when the mount fetch has finished.
    const loadMcp = async () => {
      try {
        const res = await apiRequest(`${getApiUrl()}/api/mcp/servers`)
        // Skip when an admin cross-user view has taken over the MCP list
        // (the agent load fetches the owner's servers); otherwise this mount
        // fetch could resolve last and clobber the owner list with the admin's own.
        if (res.ok && !ownerScopedMcpRef.current) {
          const data = await res.json()
          // Re-check after the await: the agent load may have claimed the
          // owner-scoped list during the JSON parse.
          // A non-list body would break every connector lookup at render.
          if (!ownerScopedMcpRef.current && Array.isArray(data)) setMcpServers(data)
        }
      } catch (error) {
        console.error("Failed to load MCP servers:", error)
      } finally {
        setMcpSettled(true)
      }
    }

    void loadKbs()
    void loadSkills()
    void loadTools()
    void loadModelDefaults()
    void loadMcp()
  }, [loadKbs, loadSkills, loadTools, loadModels, ownerScopedMcpRef])

  return {
    models,
    modelsStatus,
    retryModels: loadModels,
    userDefaultGeneral,
    modelDefaultsSettled,
    kbs,
    kbsStatus,
    reloadKbs: loadKbs,
    skills,
    skillsStatus,
    retrySkills: loadSkills,
    tools,
    intrinsicToolNames,
    skillLoaderTool,
    toolsStatus,
    retryTools: loadTools,
    mcpServers,
    setMcpServers,
    mcpSettled,
  }
}
