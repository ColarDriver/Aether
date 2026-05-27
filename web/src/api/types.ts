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

export type SessionDetail = {
  session_id: string
  info: SessionInfo
  messages: TranscriptMessage[]
}

export type SessionForkResult = {
  source_session_id: string
  forked_from_index: number
  messages_copied: number
  info: SessionInfo
  messages: TranscriptMessage[]
}

export type SessionRewindResult = {
  session_id: string
  rewound_to_index: number
  messages_kept: number
  messages_removed: number
  info: SessionInfo
  messages: TranscriptMessage[]
}

export type SessionTurnTarget = {
  target_user_message_id: string
  user_message_index: number
  user_message_count: number
  message_index: number
  content?: string | null
}

export type SessionTurnCodeSnapshot = {
  available: boolean
  files_changed: string[]
  insertions: number
  deletions: number
  checkpoint_id?: string | null
  reason?: string | null
}

export type SessionTurnCheckpoint = {
  target: SessionTurnTarget
  code: SessionTurnCodeSnapshot
  work_dir?: string | null
  conversation?: Record<string, unknown> | null
}

export type SessionTurnCheckpointsResult = {
  session_id: string
  checkpoints: SessionTurnCheckpoint[]
}

export type SessionMessageAction = {
  name: string
  supported: boolean
  label: string
  reason?: string | null
  checkpoint_id?: string | null
  destructive: boolean
}

export type SessionMessageActionsResult = {
  session_id: string
  message_index: number
  role: string
  target_user_message_id?: string | null
  user_message_index?: number | null
  actions: SessionMessageAction[]
}

export type SessionCheckpointActionBody = {
  message_index?: number | null
  target_user_message_id?: string | null
  user_message_index?: number | null
  expected_content?: string | null
  checkpoint_id?: string | null
  paths?: string[] | null
  new_session_id?: string | null
}

export type SessionCheckpointActionResult = {
  action: string
  restore?: Record<string, unknown> | null
  result: SessionRewindResult | SessionForkResult
}

export type SessionExportResult = {
  session_id: string
  data: Record<string, unknown>
}

export type SessionImportResult = {
  source_session_id?: string | null
  overwritten: boolean
  info: SessionInfo
  messages: TranscriptMessage[]
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
    base_url_source?: string | null
    url?: string | null
    suggested_base_url?: string | null
    warning?: string | null
    body_preview?: string | null
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
  extra?: Record<string, unknown>
}

export type ProviderPreflightStatus = {
  family: string
  provider_name: string
  model: string
  base_url?: string | null
  chat_completions_url?: string | null
  models_url?: string | null
  status: 'ready' | 'warning' | 'error' | string
  ready: boolean
  credential?: CredentialStatus | null
  discovery?: ProviderModelList['discovery'] | null
  issues: string[]
  suggestions: string[]
  extra?: Record<string, unknown>
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

export type McpImportedTool = {
  name: string
  server: string
  local_name: string
  description: string
  enabled: boolean
}

export type McpServerSummary = {
  name: string
  status: string
  tools_count: number
  resources_count: number
  credential_status: string
}

export type McpConfiguredServer = {
  name: string
  enabled: boolean
  transport: string
  command?: string | null
  args: string[]
  url?: string | null
  env_keys: string[]
  header_keys: string[]
  timeout?: number | null
  connect_timeout?: number | null
  source: string
}

export type McpConfigList = {
  config_path: string
  exists: boolean
  servers: McpConfiguredServer[]
}

export type McpConfigMutationResult = {
  ok: boolean
  config_path: string
  message: string
  server?: McpConfiguredServer | null
}

export type McpStatus = {
  enabled: boolean
  status: string
  message: string
  servers: McpServerSummary[]
  imported_tools: McpImportedTool[]
}

export type McpResourceSummary = {
  server: string
  uri: string
  name: string
  mime_type?: string | null
  description: string
}

export type McpResourceList = {
  enabled: boolean
  status: string
  message: string
  resources: McpResourceSummary[]
}

export type McpResourceContent = {
  type: string
  text?: string | null
  blob?: string | null
  mime_type?: string | null
  uri?: string | null
}

export type McpResourceReadResult = {
  enabled: boolean
  status: string
  message: string
  server: string
  uri: string
  name?: string | null
  mime_type?: string | null
  contents: McpResourceContent[]
}

export type WebSearchStatus = {
  enabled: boolean
  provider: string
  supported_providers: string[]
  api_key_configured: boolean
  credential_name: string
  api_key_source?: string | null
  status: string
  message: string
}

export type WebSearchTestResult = {
  ok: boolean
  provider: string
  query: string
  result_count: number
  message: string
  content_preview: string
  error?: string | null
}

export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'interrupted' | 'killed' | string

export type TaskSummary = {
  task_id: string
  parent_session_id: string
  subagent_type: string
  prompt: string
  status: TaskStatus
  started_at: number
  finished_at?: number | null
  last_heartbeat: number
  model?: string | null
  isolation?: string | null
  worktree_path?: string | null
  parent_task_id?: string | null
  child_depth: number
  background: boolean
  tool_use_count: number
  input_tokens: number
  output_tokens: number
  iterations: number
  summary?: string | null
  error?: string | null
  result_path?: string | null
  output_tail?: string | null
  metadata?: Record<string, unknown> | null
}

export type TaskMessage = {
  index: number
  role: string
  content?: string | null
  name?: string | null
  tool_call_id?: string | null
  is_error?: boolean
  iteration?: number | null
  elapsed_ms?: number | null
  error?: string | null
  raw?: Record<string, unknown> | null
}

export type TaskPendingMessage = {
  index: number
  message: string
  ts?: number | null
  raw?: Record<string, unknown> | null
}

export type TaskDeliveredMessage = {
  index: number
  message: string
  ts?: number | null
  delivered_at?: number | null
  raw?: Record<string, unknown> | null
}

export type TaskMessagesResult = {
  task_id: string
  messages: TaskMessage[]
  pending_messages: TaskPendingMessage[]
  delivered_messages: TaskDeliveredMessage[]
  total_count: number
  truncated: boolean
}

export type TaskChildMessageStream = {
  task: TaskSummary
  messages: TaskMessage[]
  pending_messages: TaskPendingMessage[]
  delivered_messages: TaskDeliveredMessage[]
  total_count: number
  truncated: boolean
}

export type TaskChildMessagesResult = {
  task_id: string
  streams: TaskChildMessageStream[]
  total_count: number
  truncated: boolean
}

export type TaskResultArtifact = {
  task_id: string
  result_path?: string | null
  result: Record<string, unknown>
}

export type TaskListResult = {
  tasks: TaskSummary[]
  active_count: number
  total_count: number
}

export type TaskStopResult = {
  task_id: string
  delivered: boolean
  status: string
  message: string
}

export type TaskSendMessageResult = {
  task_id: string
  queued: boolean
  status: string
  message: string
  queued_chars: number
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

export type RunAttachment = {
  type: 'file' | 'image' | 'text'
  name?: string
  path?: string
  url?: string
  mimeType?: string
  data?: string
  isDirectory?: boolean
  lineStart?: number
  lineEnd?: number
  note?: string
  quote?: string
}

export type PlanCurrent = {
  session_id: string
  mode: 'agent' | 'plan'
  plan_path?: string | null
  has_plan: boolean
  plan_content?: string | null
  info?: SessionInfo
}


export type ContextStatus = {
  session_id: string
  context_engine: string
  compression_count: number
  last_compression?: Record<string, unknown> | null
  message_count: number
  token_estimate: number
  provider?: string | null
  model?: string | null
  context_window?: number | null
  prompt_tokens?: number
  transcript_tokens?: number
  system_tokens?: number
  memory_tokens?: number
  attachment_tokens?: number
  tool_result_tokens?: number
  pressure_level?: string
  next_action?: string
  breakdown?: Array<{
    label: string
    tokens: number
    detail?: string | null
  }>
  status?: string | null
  error?: string | null
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

export type WorkspaceRootInfo = {
  root: string
  name: string
  exists: boolean
  readable: boolean
  git_root?: string | null
  is_git: boolean
  recent_roots: string[]
  message?: string | null
}

export type WorkspaceFile = {
  root: string
  path: string
  name: string
  content: string
  size_bytes: number
  updated_at: number
  language: string
  mime_type?: string | null
  truncated: boolean
  binary: boolean
}

export type WorkspaceSearchResult = {
  root: string
  query: string
  entries: WorkspaceEntry[]
}

export type WorkspaceGitFile = {
  path: string
  status: string
  index_status: string
  worktree_status: string
  staged: boolean
  unstaged: boolean
  untracked: boolean
}

export type WorkspaceGitStatus = {
  root: string
  git_root?: string | null
  available: boolean
  branch?: string | null
  upstream?: string | null
  ahead: number
  behind: number
  clean: boolean
  files: WorkspaceGitFile[]
  message?: string | null
}

export type WorkspaceGitDiff = {
  root: string
  path?: string | null
  diff: string
  staged: boolean
  truncated: boolean
}

export type WorkspaceChange = {
  change_id: string
  path: string
  status: string
  source: string
  staged: boolean
  unstaged: boolean
  untracked: boolean
  binary: boolean
  accepted: boolean
  rejected: boolean
  conflict: boolean
  checkpoint_available: boolean
  additions: number
  removals: number
  hunks: number
  current_hash?: string | null
}

export type WorkspaceChangeList = {
  root: string
  git_root?: string | null
  available: boolean
  changes: WorkspaceChange[]
  message?: string | null
}

export type WorkspaceChangeActionResult = {
  root: string
  action: string
  paths: string[]
  status: WorkspaceGitStatus
  checkpoint_id?: string | null
  message?: string | null
}

export type WorkspaceChangeVerificationResult = {
  root: string
  paths: string[]
  status: string
  command: string[]
  exit_code?: number | null
  stdout: string
  stderr: string
  message?: string | null
}

export type WorkspaceCheckpointFile = {
  path: string
  exists: boolean
  size_bytes: number
  binary: boolean
}

export type WorkspaceCheckpoint = {
  checkpoint_id: string
  label?: string | null
  created_at: number
  root: string
  files: WorkspaceCheckpointFile[]
}

export type WorkspaceCheckpointList = {
  root: string
  checkpoints: WorkspaceCheckpoint[]
}
