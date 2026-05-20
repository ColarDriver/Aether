export type ConsoleView = 'chat' | 'models' | 'tools' | 'skills' | 'diagnostics' | 'settings'

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

export type TranscriptMessage = {
  role: 'user' | 'assistant' | 'system' | 'tool'
  text?: string | null
  name?: string | null
  tool_call_id?: string | null
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
