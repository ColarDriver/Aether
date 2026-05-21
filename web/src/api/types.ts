export type ConsoleView = 'chat' | 'models' | 'tools' | 'skills' | 'diagnostics' | 'logs' | 'environment' | 'settings'

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

export type TranscriptMessage = {
  role: 'user' | 'assistant' | 'system' | 'tool'
  text?: string | null
  name?: string | null
  tool_call_id?: string | null
  tool_calls?: TranscriptToolCall[]
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
