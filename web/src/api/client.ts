import type {
  AnalyticsReport,
  CommandCatalog,
  ConfigPaths,
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
  PlanCurrent,
  PrefMutationResult,
  ProviderModelList,
  ProviderRuntimeStatus,
  ProviderSelectionResult,
  ProviderSummary,
  SessionInfo,
  SkillSummary,
  StatusResponse,
  TaskListResult,
  TaskSummary,
  ToolGroup,
  ToolSummary,
  TranscriptMessage,
  WorkspaceFile,
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

export async function request<T>(method: string, path: string, body?: unknown, options?: { timeoutMs?: number }): Promise<T> {
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
      const errorBody = await response.json().catch(() => response.text())
      throw new ApiError(response.status, errorBody)
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
  workspaceFile: (path: string) => request<WorkspaceFile>('GET', '/api/workspace/file?path=' + encodeURIComponent(path)),
  workspaceSearch: (q: string, limit = 100) => {
    const query = new URLSearchParams({ q, limit: String(limit) })
    return request<WorkspaceSearchResult>('GET', '/api/workspace/search?' + query.toString())
  },
  sessions: () => request<{ sessions: SessionInfo[] }>('GET', '/api/sessions'),
  createSession: (body: { provider: string; model: string; base_url?: string | null; system_prompt?: string | null }) =>
    request<SessionInfo>('POST', '/api/sessions', body),
  updateSession: (sessionId: string, body: { provider?: string | null; model?: string | null; base_url?: string | null; system_prompt?: string | null; update_base_url?: boolean; update_system_prompt?: boolean }) =>
    request<SessionInfo>('PATCH', '/api/sessions/' + encodeURIComponent(sessionId), body),
  searchSessions: (query: string, limit = 50) =>
    request<{ sessions: SessionInfo[] }>('GET', '/api/sessions/search?q=' + encodeURIComponent(query) + '&limit=' + encodeURIComponent(String(limit))),
  sessionDetail: (sessionId: string) =>
    request<{ session_id: string; info: SessionInfo; messages: TranscriptMessage[] }>('GET', '/api/sessions/' + encodeURIComponent(sessionId)),
  resumeSession: (sessionId: string) =>
    request<{ session_id: string; info: SessionInfo; messages: TranscriptMessage[] }>('POST', '/api/sessions/' + encodeURIComponent(sessionId) + '/resume'),
  sessionMessages: (sessionId: string) =>
    request<{ session_id: string; messages: TranscriptMessage[] }>('GET', '/api/sessions/' + encodeURIComponent(sessionId) + '/messages'),
  deleteSession: (sessionId: string) => request<void>("DELETE", "/api/sessions/" + encodeURIComponent(sessionId)),
  planCurrent: (sessionId: string) =>
    request<PlanCurrent>('GET', '/api/plan/' + encodeURIComponent(sessionId)),
  setPlanMode: (sessionId: string, mode: 'agent' | 'plan') =>
    request<PlanCurrent>('PUT', '/api/plan/' + encodeURIComponent(sessionId) + '/mode', { mode }),
  providers: () => request<{ providers: ProviderSummary[] }>('GET', '/api/providers'),
  currentProvider: () => request<ProviderRuntimeStatus>('GET', '/api/providers/current'),
  providerModels: (provider: string) => request<ProviderModelList>('GET', '/api/providers/' + encodeURIComponent(provider) + '/models'),
  selectModel: (body: { provider: string; model: string; persist_last_model?: boolean }) =>
    request<ProviderSelectionResult>('POST', '/api/model/select', body),
  toolGroups: () => request<{ groups: ToolGroup[] }>('GET', '/api/tools/groups'),
  tools: () => request<{ tools: ToolSummary[] }>('GET', '/api/tools'),
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
