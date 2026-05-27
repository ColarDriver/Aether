import type {
  AnalyticsReport,
  CommandCatalog,
  ConfigPaths,
  ContextStatus,
  DocContent,
  DocIndex,
  EffectiveConfig,
  EnvCatalog,
  EnvMutationResult,
  EnvRevealAuditEntry,
  EnvRevealResult,
  HealthStatus,
  LogFileSummary,
  LogReadResult,
  McpConfigList,
  McpConfigMutationResult,
  McpResourceList,
  McpResourceReadResult,
  McpStatus,
  PlanCurrent,
  PrefMutationResult,
  ProviderModelList,
  ProviderPreflightStatus,
  ProviderRuntimeStatus,
  ProviderSelectionResult,
  ProviderSummary,
  SessionDetail,
  SessionCheckpointActionBody,
  SessionCheckpointActionResult,
  SessionExportResult,
  SessionForkResult,
  SessionImportResult,
  SessionInfo,
  SessionMessageActionsResult,
  SessionRewindResult,
  SessionTurnCheckpointsResult,
  SkillSummary,
  StatusResponse,
  TaskChildMessagesResult,
  TaskListResult,
  TaskMessagesResult,
  TaskResultArtifact,
  TaskSendMessageResult,
  TaskStopResult,
  TaskSummary,
  ToolGroup,
  ToolSummary,
  TranscriptMessage,
  WebSearchStatus,
  WebSearchTestResult,
  WorkspaceChangeActionResult,
  WorkspaceChangeList,
  WorkspaceChangeVerificationResult,
  WorkspaceCheckpoint,
  WorkspaceCheckpointList,
  WorkspaceEntry,
  WorkspaceFile,
  WorkspaceGitDiff,
  WorkspaceGitStatus,
  WorkspaceRootInfo,
  WorkspaceSearchResult,
  WorkspaceTree,
} from './types'

const DEFAULT_BASE_URL =
  typeof import.meta !== 'undefined' && typeof import.meta.env.VITE_AETHER_WEB_URL === 'string'
    ? import.meta.env.VITE_AETHER_WEB_URL
    : ''

let baseUrl = normalizeBaseUrl(window.__AETHER_BASE_PATH__ || DEFAULT_BASE_URL)
let sessionToken: string | null = window.__AETHER_SESSION_TOKEN__ || null

export class ApiError extends Error {
  constructor(
    public status: number,
    public body: unknown,
  ) {
    super(errorMessage(status, body))
    this.name = 'ApiError'
  }
}

export function setBaseUrl(next: string) {
  baseUrl = normalizeBaseUrl(next)
}

export function getBaseUrl() {
  return baseUrl
}

export function setSessionToken(token: string | null) {
  const trimmed = token?.trim() ?? ''
  sessionToken = trimmed.length > 0 ? trimmed : null
}

export function getSessionToken() {
  return sessionToken
}

export async function refreshSessionTokenFromBootstrapDocument(): Promise<string | null> {
  const apiToken = await refreshSessionTokenFromBootstrapApi().catch(() => null)
  if (apiToken) return apiToken

  const url = bootstrapDocumentUrl()
  const response = await fetch(url, {
    method: 'GET',
    cache: 'no-store',
    headers: { Accept: 'text/html' },
  })
  if (!response.ok) return null
  const html = await response.text()
  return applySessionToken(extractSessionToken(html))
}

async function refreshSessionTokenFromBootstrapApi(): Promise<string | null> {
  const response = await fetch(baseUrl + '/api/bootstrap', {
    method: 'GET',
    cache: 'no-store',
    headers: { Accept: 'application/json' },
  })
  if (!response.ok) return null
  const payload = await response.json().catch(() => null)
  if (!payload || typeof payload !== 'object') return null
  return applySessionToken((payload as { session_token?: unknown }).session_token)
}

function applySessionToken(token: unknown): string | null {
  if (typeof token !== 'string') return null
  const trimmed = token.trim()
  if (!trimmed || trimmed === sessionToken) return null
  setSessionToken(trimmed)
  window.__AETHER_SESSION_TOKEN__ = trimmed
  return trimmed
}

function bootstrapDocumentUrl(): string {
  const base = normalizeBaseUrl(baseUrl || window.location.origin) || window.location.origin
  const url = new URL(base, window.location.origin)
  if (!url.pathname || url.pathname === '/') {
    url.pathname = window.location.pathname || '/'
  }
  url.search = ''
  url.hash = ''
  return url.toString()
}

function extractSessionToken(html: string): string | null {
  const marker = 'window.__AETHER_SESSION_TOKEN__='
  const index = html.indexOf(marker)
  if (index < 0) return null
  const start = index + marker.length
  const quote = html[start]
  if (quote !== '\'' && quote !== '"') return null
  const end = html.indexOf(quote, start + 1)
  if (end < 0) return null
  return html.slice(start + 1, end)
}

export async function request<T>(method: string, path: string, body?: unknown, options?: { timeoutMs?: number }): Promise<T> {
  return requestWithTokenRetry<T>(method, path, body, options, false)
}

export async function requestBlob(method: string, path: string, options?: { timeoutMs?: number }): Promise<Blob> {
  return requestBlobWithTokenRetry(method, path, options, false)
}

async function requestWithTokenRetry<T>(method: string, path: string, body: unknown, options: { timeoutMs?: number } | undefined, didRefreshToken: boolean): Promise<T> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), options?.timeoutMs ?? 30_000)
  try {
    const response = await fetch(baseUrl + path, {
      method,
      headers: buildHeaders(),
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    })
    clearTimeout(timeout)
    if (!response.ok) {
      if (response.status === 401 && !didRefreshToken) {
        const refreshed = await refreshSessionTokenFromBootstrapDocument().catch(() => null)
        if (refreshed) return requestWithTokenRetry<T>(method, path, body, options, true)
      }
      throw await buildApiError(response)
    }
    if (response.status === 204) return undefined as T
    return response.json() as Promise<T>
  } catch (error) {
    clearTimeout(timeout)
    if (controller.signal.aborted) {
      throw new Error('Request timed out after ' + Math.round((options?.timeoutMs ?? 30_000) / 1000) + 's')
    }
    throw error
  }
}

async function requestBlobWithTokenRetry(method: string, path: string, options: { timeoutMs?: number } | undefined, didRefreshToken: boolean): Promise<Blob> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), options?.timeoutMs ?? 30_000)
  try {
    const response = await fetch(baseUrl + path, {
      method,
      headers: buildHeaders(),
      signal: controller.signal,
    })
    clearTimeout(timeout)
    if (!response.ok) {
      if (response.status === 401 && !didRefreshToken) {
        const refreshed = await refreshSessionTokenFromBootstrapDocument().catch(() => null)
        if (refreshed) return requestBlobWithTokenRetry(method, path, options, true)
      }
      throw await buildApiError(response)
    }
    return response.blob()
  } catch (error) {
    clearTimeout(timeout)
    if (controller.signal.aborted) {
      throw new Error('Request timed out after ' + Math.round((options?.timeoutMs ?? 30_000) / 1000) + 's')
    }
    throw error
  }
}

async function buildApiError(response: Response): Promise<ApiError> {
  const raw = await response.text().catch(() => '')
  let errorBody: unknown = raw
  try {
    errorBody = JSON.parse(raw)
  } catch {
    // body is not JSON; keep the raw text
  }
  return new ApiError(response.status, errorBody)
}

function buildHeaders(): Record<string, string> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (sessionToken) headers['X-Aether-Session-Token'] = sessionToken
  return headers
}

function errorMessage(status: number, body: unknown) {
  if (body && typeof body === 'object' && 'error' in body) {
    const error = (body as { error?: { message?: unknown } }).error
    if (typeof error?.message === 'string') return error.message
  }
  return 'API error ' + status
}

function normalizeBaseUrl(value: string) {
  return value.replace(/\/$/, '')
}

export const api = {
  status: () => request<StatusResponse>('GET', '/api/status'),
  health: () => request<HealthStatus>('GET', '/api/health'),
  commands: () => request<CommandCatalog>('GET', '/api/commands'),
  analytics: (params: { days?: number; limit?: number } = {}) => {
    const query = new URLSearchParams()
    if (params.days) query.set('days', String(params.days))
    if (params.limit) query.set('limit', String(params.limit))
    const suffix = query.toString() ? '?' + query.toString() : ''
    return request<AnalyticsReport>('GET', '/api/analytics' + suffix)
  },
  docs: () => request<DocIndex>('GET', '/api/docs'),
  doc: (path: string) => request<DocContent>('GET', '/api/docs/' + encodePathSegments(path)),
  workspaceTree: (path = '') => {
    const suffix = path ? '?path=' + encodeURIComponent(path) : ''
    return request<WorkspaceTree>('GET', '/api/workspace/tree' + suffix)
  },
  workspaceRoot: () => request<WorkspaceRootInfo>('GET', '/api/workspace/root'),
  switchWorkspaceRoot: (body: { path: string; session_id?: string | null; remember?: boolean }) =>
    request<WorkspaceRootInfo>('PUT', '/api/workspace/root', body),
  workspaceFile: (path: string) => request<WorkspaceFile>('GET', '/api/workspace/file?path=' + encodeURIComponent(path)),
  workspaceFileBlob: (path: string) => requestBlob('GET', '/api/workspace/raw?path=' + encodeURIComponent(path)),
  workspaceSaveFile: (path: string, content: string) => request<WorkspaceFile>('PUT', '/api/workspace/file', { path, content }),
  workspaceCreateFile: (path: string, content = '') => request<WorkspaceFile>('POST', '/api/workspace/file', { path, content }),
  workspaceCreateDirectory: (path: string) => request<WorkspaceEntry>('POST', '/api/workspace/directory', { path }),
  workspaceRenamePath: (path: string, newPath: string) => request<WorkspaceEntry>('PATCH', '/api/workspace/path', { path, new_path: newPath }),
  workspaceDeletePath: (path: string, recursive = false) => {
    const query = new URLSearchParams({ path, recursive: recursive ? 'true' : 'false' })
    return request<void>('DELETE', '/api/workspace/path?' + query.toString())
  },
  workspaceSearch: (q: string, limit = 100) => {
    const query = new URLSearchParams({ q, limit: String(limit) })
    return request<WorkspaceSearchResult>('GET', '/api/workspace/search?' + query.toString())
  },
  workspaceGitStatus: () => request<WorkspaceGitStatus>('GET', '/api/workspace/git/status'),
  workspaceGitDiff: (path?: string | null, staged = false) => {
    const query = new URLSearchParams()
    if (path) query.set('path', path)
    if (staged) query.set('staged', 'true')
    const suffix = query.toString() ? '?' + query.toString() : ''
    return request<WorkspaceGitDiff>('GET', '/api/workspace/git/diff' + suffix)
  },
  workspaceGitRestore: (path: string) => request<WorkspaceGitStatus>('POST', '/api/workspace/git/restore', { path }),
  workspaceChanges: () => request<WorkspaceChangeList>('GET', '/api/workspace/changes'),
  acceptWorkspaceChanges: (paths: string[]) =>
    request<WorkspaceChangeActionResult>('POST', '/api/workspace/changes/accept', { paths }),
  rejectWorkspaceChanges: (body: { paths: string[]; checkpoint_id?: string | null; expected_hashes?: Record<string, string> | null }) =>
    request<WorkspaceChangeActionResult>('POST', '/api/workspace/changes/reject', body),
  verifyWorkspaceChanges: (body: { paths: string[]; command?: string[] | null; timeout_seconds?: number }) =>
    request<WorkspaceChangeVerificationResult>('POST', '/api/workspace/changes/verify', body),
  workspaceCheckpoints: () => request<WorkspaceCheckpointList>('GET', '/api/workspace/checkpoints'),
  createWorkspaceCheckpoint: (body: { label?: string | null } = {}) =>
    request<WorkspaceCheckpoint>('POST', '/api/workspace/checkpoints', body),
  restoreWorkspaceCheckpoint: (checkpointId: string) =>
    request<WorkspaceCheckpoint>('POST', '/api/workspace/checkpoints/' + encodeURIComponent(checkpointId) + '/restore'),
  restoreWorkspaceCheckpointPaths: (checkpointId: string, paths: string[]) =>
    request<WorkspaceGitStatus>('POST', '/api/workspace/checkpoints/' + encodeURIComponent(checkpointId) + '/restore-paths', { paths }),
  sessions: () => request<{ sessions: SessionInfo[] }>('GET', '/api/sessions'),
  createSession: (body: { provider: string; model: string; base_url?: string | null; system_prompt?: string | null }) =>
    request<SessionInfo>('POST', '/api/sessions', body),
  updateSession: (sessionId: string, body: { provider?: string | null; model?: string | null; base_url?: string | null; system_prompt?: string | null; update_base_url?: boolean; update_system_prompt?: boolean }) =>
    request<SessionInfo>('PATCH', '/api/sessions/' + encodeURIComponent(sessionId), body),
  searchSessions: (query: string, limit = 50) =>
    request<{ sessions: SessionInfo[] }>('GET', '/api/sessions/search?q=' + encodeURIComponent(query) + '&limit=' + encodeURIComponent(String(limit))),
  sessionDetail: (sessionId: string) =>
    request<SessionDetail>('GET', '/api/sessions/' + encodeURIComponent(sessionId)),
  resumeSession: (sessionId: string) =>
    request<SessionDetail>('POST', '/api/sessions/' + encodeURIComponent(sessionId) + '/resume'),
  forkSession: (sessionId: string, body: { message_index: number; new_session_id?: string | null }) =>
    request<SessionForkResult>('POST', '/api/sessions/' + encodeURIComponent(sessionId) + '/fork', body),
  rewindSession: (sessionId: string, body: { message_index?: number | null; target_user_message_id?: string | null; user_message_index?: number | null; expected_content?: string | null }) =>
    request<SessionRewindResult>('POST', '/api/sessions/' + encodeURIComponent(sessionId) + '/rewind', body),
  sessionTurnCheckpoints: (sessionId: string) =>
    request<SessionTurnCheckpointsResult>('GET', '/api/sessions/' + encodeURIComponent(sessionId) + '/turn-checkpoints'),
  sessionMessageActions: (sessionId: string, messageIndex: number) =>
    request<SessionMessageActionsResult>('GET', '/api/sessions/' + encodeURIComponent(sessionId) + '/message-actions/' + encodeURIComponent(String(messageIndex))),
  sessionActionFork: (sessionId: string, body: SessionCheckpointActionBody) =>
    request<SessionForkResult>('POST', '/api/sessions/' + encodeURIComponent(sessionId) + '/actions/fork', body),
  sessionActionRewind: (sessionId: string, body: SessionCheckpointActionBody) =>
    request<SessionCheckpointActionResult>('POST', '/api/sessions/' + encodeURIComponent(sessionId) + '/actions/rewind', body),
  sessionActionUndoRun: (sessionId: string, body: SessionCheckpointActionBody) =>
    request<SessionCheckpointActionResult>('POST', '/api/sessions/' + encodeURIComponent(sessionId) + '/actions/undo-run', body),
  sessionActionRetry: (sessionId: string, body: SessionCheckpointActionBody) =>
    request<SessionCheckpointActionResult>('POST', '/api/sessions/' + encodeURIComponent(sessionId) + '/actions/retry', body),
  renameSession: (sessionId: string, newSessionId: string) =>
    request<SessionInfo>('POST', '/api/sessions/' + encodeURIComponent(sessionId) + '/rename', { new_session_id: newSessionId }),
  exportSession: (sessionId: string) =>
    request<SessionExportResult>('GET', '/api/sessions/' + encodeURIComponent(sessionId) + '/export'),
  importSession: (body: { data: Record<string, unknown>; new_session_id?: string | null; overwrite?: boolean; make_current?: boolean }) =>
    request<SessionImportResult>('POST', '/api/sessions/import', body),
  sessionMessages: (sessionId: string) =>
    request<{ session_id: string; messages: TranscriptMessage[] }>('GET', '/api/sessions/' + encodeURIComponent(sessionId) + '/messages'),
  deleteSession: (sessionId: string) => request<void>("DELETE", "/api/sessions/" + encodeURIComponent(sessionId)),
  planCurrent: (sessionId: string) =>
    request<PlanCurrent>('GET', '/api/plan/' + encodeURIComponent(sessionId)),
  setPlanMode: (sessionId: string, mode: 'agent' | 'plan') =>
    request<PlanCurrent>('PUT', '/api/plan/' + encodeURIComponent(sessionId) + '/mode', { mode }),
  clearPlan: (sessionId: string) =>
    request<PlanCurrent>('POST', '/api/plan/' + encodeURIComponent(sessionId) + '/clear'),
  contextStatus: (sessionId: string) =>
    request<ContextStatus>('GET', '/api/context/' + encodeURIComponent(sessionId) + '/status'),
  estimateContext: (sessionId: string, body: { draft?: string; attachments?: Array<Record<string, unknown>> } = {}) =>
    request<ContextStatus>('POST', '/api/context/' + encodeURIComponent(sessionId) + '/estimate', body),
  compressContext: (sessionId: string, body: { focus?: string | null; force?: boolean } = {}) =>
    request<ContextStatus>('POST', '/api/context/' + encodeURIComponent(sessionId) + '/compress', body),
  providers: () => request<{ providers: ProviderSummary[] }>('GET', '/api/providers'),
  currentProvider: () => request<ProviderRuntimeStatus>('GET', '/api/providers/current'),
  providerPreflight: (params: { provider?: string | null; model?: string | null; baseUrl?: string | null } = {}) => {
    const query = new URLSearchParams()
    if (params.provider) query.set('provider', params.provider)
    if (params.model) query.set('model', params.model)
    if (params.baseUrl) query.set('base_url', params.baseUrl)
    const suffix = query.toString() ? '?' + query.toString() : ''
    return request<ProviderPreflightStatus>('GET', '/api/providers/preflight' + suffix)
  },
  providerModels: (provider: string) => request<ProviderModelList>('GET', '/api/providers/' + encodeURIComponent(provider) + '/models'),
  selectModel: (body: { provider: string; model: string; persist_last_model?: boolean }) =>
    request<ProviderSelectionResult>('POST', '/api/model/select', body),
  toolGroups: () => request<{ groups: ToolGroup[] }>('GET', '/api/tools/groups'),
  tools: () => request<{ tools: ToolSummary[] }>('GET', '/api/tools'),
  mcpStatus: () => request<McpStatus>('GET', '/api/mcp/status'),
  mcpConfig: () => request<McpConfigList>('GET', '/api/mcp/config'),
  upsertMcpServer: (body: { name: string; command?: string | null; args?: string[]; env?: Record<string, string>; url?: string | null; headers?: Record<string, string>; transport?: string | null; timeout?: number | null; connect_timeout?: number | null; enabled?: boolean }) =>
    request<McpConfigMutationResult>('PUT', '/api/mcp/servers', body),
  deleteMcpServer: (name: string) => request<McpConfigMutationResult>('DELETE', '/api/mcp/servers/' + encodeURIComponent(name)),
  refreshMcp: () => request<McpStatus>('POST', '/api/mcp/refresh'),
  mcpResources: (server?: string | null) => {
    const suffix = server ? '?server=' + encodeURIComponent(server) : ''
    return request<McpResourceList>('GET', '/api/mcp/resources' + suffix)
  },
  mcpResourceRead: (server: string, uri: string) => {
    const query = new URLSearchParams({ server, uri })
    return request<McpResourceReadResult>('GET', '/api/mcp/resources/read?' + query.toString())
  },
  webSearchStatus: () => request<WebSearchStatus>('GET', '/api/web-search/status'),
  testWebSearch: (body: { query: string; max_results?: number }) => request<WebSearchTestResult>('POST', '/api/web-search/test', body),
  skills: () => request<{ skills: SkillSummary[] }>('GET', '/api/skills'),
  tasks: (params: { sessionId?: string; activeOnly?: boolean; includeOutputTail?: boolean; limit?: number } = {}) => {
    const query = taskQuery(params)
    return request<TaskListResult>('GET', '/api/tasks' + query)
  },
  sessionTasks: (sessionId: string, params: { activeOnly?: boolean; includeOutputTail?: boolean; limit?: number } = {}) => {
    const query = taskQuery(params)
    return request<TaskListResult>('GET', '/api/sessions/' + encodeURIComponent(sessionId) + '/tasks' + query)
  },
  taskDetail: (taskId: string) => request<TaskSummary>('GET', '/api/tasks/' + encodeURIComponent(taskId)),
  taskMessages: (taskId: string, params: { limit?: number } = {}) => {
    const query = params.limit ? '?limit=' + encodeURIComponent(String(params.limit)) : ''
    return request<TaskMessagesResult>('GET', '/api/tasks/' + encodeURIComponent(taskId) + '/messages' + query)
  },
  taskChildMessages: (taskId: string, params: { limit?: number; perTaskLimit?: number } = {}) => {
    const query = new URLSearchParams()
    if (params.limit) query.set('limit', String(params.limit))
    if (params.perTaskLimit) query.set('per_task_limit', String(params.perTaskLimit))
    const suffix = query.toString() ? '?' + query.toString() : ''
    return request<TaskChildMessagesResult>('GET', '/api/tasks/' + encodeURIComponent(taskId) + '/children/messages' + suffix)
  },
  taskResult: (taskId: string) => request<TaskResultArtifact>('GET', '/api/tasks/' + encodeURIComponent(taskId) + '/result'),
  sendTaskMessage: (taskId: string, body: { message: string; summary?: string | null }) =>
    request<TaskSendMessageResult>('POST', '/api/tasks/' + encodeURIComponent(taskId) + '/messages', body),
  stopTask: (taskId: string) => request<TaskStopResult>('POST', '/api/tasks/' + encodeURIComponent(taskId) + '/stop'),
  diagnostics: () => request<HealthStatus>('GET', '/api/health'),
  logFiles: () => request<{ files: LogFileSummary[] }>('GET', '/api/logs/files'),
  logs: (params: { file?: string; lines?: number; level?: string; component?: string; search?: string }) => {
    const query = new URLSearchParams()
    if (params.file) query.set('file', params.file)
    if (params.lines) query.set('lines', String(params.lines))
    if (params.level && params.level !== 'ALL') query.set('level', params.level)
    if (params.component && params.component !== 'all') query.set('component', params.component)
    if (params.search) query.set('search', params.search)
    const suffix = query.toString() ? '?' + query.toString() : ''
    return request<LogReadResult>('GET', '/api/logs' + suffix)
  },
  config: () => request<EffectiveConfig>('GET', '/api/config'),
  configPaths: () => request<ConfigPaths>('GET', '/api/config/paths'),
  prefs: () => request<Record<string, unknown>>("GET", "/api/prefs"),
  pref: (key: string) => request<{ key: string; value: unknown }>("GET", "/api/prefs/" + encodeURIComponent(key)),
  setPref: (body: { key: string; value: unknown }) => request<PrefMutationResult>("PUT", "/api/prefs", body),
  deletePref: (key: string) => request<PrefMutationResult>("DELETE", "/api/prefs", { key }),
  env: () => request<EnvCatalog>('GET', '/api/env'),
  setEnvVar: (body: { key: string; value: string }) => request<EnvMutationResult>('PUT', '/api/env', body),
  deleteEnvVar: (key: string) => request<EnvMutationResult>('DELETE', '/api/env', { key }),
  revealEnvVar: (key: string) => request<EnvRevealResult>("POST", "/api/env/reveal", { key }),
  revealEnvAudit: () => request<{ events: EnvRevealAuditEntry[] }>("GET", "/api/env/reveal-audit"),
}

function encodePathSegments(path: string) {
  return path.split('/').map((part) => encodeURIComponent(part)).join('/')
}

function taskQuery(params: { sessionId?: string; activeOnly?: boolean; includeOutputTail?: boolean; limit?: number }) {
  const query = new URLSearchParams()
  if (params.sessionId) query.set('session_id', params.sessionId)
  if (params.activeOnly) query.set('active_only', 'true')
  if (params.includeOutputTail) query.set('include_output_tail', 'true')
  if (params.limit) query.set('limit', String(params.limit))
  const serialized = query.toString()
  return serialized ? '?' + serialized : ''
}
