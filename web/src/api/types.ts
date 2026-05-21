export type ConsoleView = 'chat' | 'sessions' | 'models' | 'tools' | 'skills' | 'diagnostics' | 'logs' | 'analytics' | 'docs' | 'workspace' | 'environment' | 'settings'

export type SessionInfo = {
  session_id: string
  created_at: number
  updated_at: number
  provider: string
  model: string
  base_url?: string | null
  system_prompt?: string | null
  message_count: number
  summary?: string | null
  mode?: string | null
}

export type TranscriptToolCall = {
  id: string
  name: string
  arguments: Record<string, unknown>
}

export type TranscriptAttachment = {
  type: 'file' | 'image' | 'text'
  name?: string | null
  path?: string | null
  url?: string | null
  mime_type?: string | null
  mimeType?: string | null
  data?: string | null
  is_directory?: boolean | null
  isDirectory?: boolean | null
  line_start?: number | null
  lineStart?: number | null
  line_end?: number | null
  lineEnd?: number | null
  note?: string | null
  quote?: string | null
}

export type TranscriptMessage = {
  role: 'user' | 'assistant' | 'system' | 'tool'
  text?: string | null
  name?: string | null
  tool_call_id?: string | null
  tool_calls?: TranscriptToolCall[]
  attachments?: TranscriptAttachment[] | null
  is_error?: boolean
  metadata?: Record<string, unknown> | null
}

export type CredentialStatus = {
  source: string
  name: string
  configured: boolean
  redacted?: string
}

export type ProviderSummary = {
  name: string
  display_name: string
  requires_api_key: boolean
  default_base_url?: string | null
}

export type ModelSummary = {
  id: string
  display_name: string
  context_window?: number | null
}

export type ProviderModelList = {
  models: ModelSummary[]
  discovery: {
    kind: string
    source?: string | null
    reason?: string | null
    error?: string | null
    count?: number | null
    base_url?: string | null
  }
}

export type ProviderRuntimeStatus = {
  family: string
  provider_name: string
  model: string
  base_url?: string | null
  api_key_env_names: string[]
  model_env_names: string[]
  base_url_env_names: string[]
  source: string
  credential?: CredentialStatus | null
}

export type ProviderSelectionResult = {
  provider: string
  family: string
  model: string
  base_url?: string | null
  ready: boolean
  missing_credentials: string[]
  credential?: CredentialStatus | null
}

export type ToolSummary = {
  name: string
  description: string
  parameters: Record<string, unknown>
  required: string[]
  enabled: boolean
}

export type ToolGroup = {
  name: string
  tools: ToolSummary[]
}

export type SkillSummary = {
  name: string
  description: string
  when_to_use: string
  source: {
    source: string
    path?: string | null
  }
  version?: string | null
}

export type ServiceStatus = {
  name: string
  available: boolean
  status: string
  detail?: string | null
}

export type HealthStatus = {
  status: string
  runtime: {
    python_version: string
    platform: string
    implementation: string
  }
  services: ServiceStatus[]
  diagnostics?: {
    enabled: boolean
    pending_count: number
  } | null
}

export type ConfigPaths = {
  aether_home: string
  sessions_dir: string
  prefs_file: string
}

export type EffectiveConfig = {
  values: Record<string, unknown>
}

export type PrefMutationResult = {
  ok: boolean
  key: string
  value?: unknown
  deleted?: boolean
}

export type StatusResponse = {
  ok: boolean
  name: string
  version: string
  web: { enabled: boolean }
}

export type RunSocketFrame = {
  type: string
  id?: string | number | null
  transport_sequence?: number
  payload?: Record<string, unknown>
}

export type SlashCommandInfo = {
  name: string
  description: string
  category?: string | null
}

export type CommandCatalog = {
  commands: SlashCommandInfo[]
}


export type LogFileSummary = {
  key: string
  name: string
  path: string
  exists: boolean
  size_bytes: number
}

export type LogReadResult = {
  file: string
  path: string
  exists: boolean
  lines: string[]
  available_files: LogFileSummary[]
}


export type EnvVarSummary = {
  key: string
  is_set: boolean
  source: 'file' | 'process' | 'missing'
  redacted_value?: string | null
  description: string
  category: string
  is_secret: boolean
  advanced: boolean
  url?: string | null
}

export type EnvCatalog = {
  env_path: string
  variables: EnvVarSummary[]
}

export type EnvMutationResult = {
  ok: boolean
  key: string
  env_path: string
}

export type EnvRevealResult = {
  key: string
  value: string
  source: 'file' | 'process' | 'missing'
}

export type EnvRevealAuditEntry = {
  key: string
  source: "file" | "process" | "missing"
  revealed_at: number
}


export type TokenUsageSummary = {
  input_tokens: number
  output_tokens: number
  cache_read_tokens: number
  cache_write_tokens: number
  reasoning_tokens: number
  total_tokens: number
}

export type AnalyticsSummary = {
  session_count: number
  message_count: number
  assistant_message_count: number
  tool_call_count: number
  usage: TokenUsageSummary
}

export type AnalyticsDailyEntry = {
  day: string
  sessions: number
  messages: number
  tool_calls: number
  usage: TokenUsageSummary
}

export type AnalyticsModelEntry = {
  provider: string
  model: string
  sessions: number
  messages: number
  tool_calls: number
  usage: TokenUsageSummary
}

export type AnalyticsSessionEntry = {
  session_id: string
  summary?: string | null
  provider: string
  model: string
  updated_at: number
  messages: number
  tool_calls: number
  usage: TokenUsageSummary
}

export type AnalyticsReport = {
  days: number
  summary: AnalyticsSummary
  daily: AnalyticsDailyEntry[]
  models: AnalyticsModelEntry[]
  top_sessions: AnalyticsSessionEntry[]
}


export type DocSummary = {
  path: string
  title: string
  size_bytes: number
  updated_at: number
}

export type DocIndex = {
  root: string
  default_path?: string | null
  documents: DocSummary[]
}

export type DocContent = {
  path: string
  title: string
  content: string
  size_bytes: number
  updated_at: number
}


export type WorkspaceEntry = {
  path: string
  name: string
  kind: 'file' | 'directory'
  size_bytes?: number | null
  updated_at?: number | null
}

export type WorkspaceTree = {
  root: string
  path: string
  parent_path?: string | null
  entries: WorkspaceEntry[]
}

export type WorkspaceFile = {
  root: string
  path: string
  name: string
  content: string
  size_bytes: number
  updated_at: number
  language: string
  truncated: boolean
  binary: boolean
}

export type WorkspaceSearchResult = {
  root: string
  query: string
  entries: WorkspaceEntry[]
}
