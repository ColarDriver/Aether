import type {
  ConfigPaths,
  EffectiveConfig,
  HealthStatus,
  ProviderModelList,
  ProviderRuntimeStatus,
  ProviderSelectionResult,
  ProviderSummary,
  SessionInfo,
  SkillSummary,
  StatusResponse,
  ToolGroup,
  ToolSummary,
  TranscriptMessage,
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
  sessions: () => request<{ sessions: SessionInfo[] }>('GET', '/api/sessions'),
  createSession: (body: { provider: string; model: string; base_url?: string | null; system_prompt?: string | null }) =>
    request<SessionInfo>('POST', '/api/sessions', body),
  resumeSession: (sessionId: string) =>
    request<{ session_id: string; info: SessionInfo; messages: TranscriptMessage[] }>('POST', '/api/sessions/' + encodeURIComponent(sessionId) + '/resume'),
  sessionMessages: (sessionId: string) =>
    request<{ session_id: string; messages: TranscriptMessage[] }>('GET', '/api/sessions/' + encodeURIComponent(sessionId) + '/messages'),
  providers: () => request<{ providers: ProviderSummary[] }>('GET', '/api/providers'),
  currentProvider: () => request<ProviderRuntimeStatus>('GET', '/api/providers/current'),
  providerModels: (provider: string) => request<ProviderModelList>('GET', '/api/providers/' + encodeURIComponent(provider) + '/models'),
  selectModel: (body: { provider: string; model: string; persist_last_model?: boolean }) =>
    request<ProviderSelectionResult>('POST', '/api/model/select', body),
  toolGroups: () => request<{ groups: ToolGroup[] }>('GET', '/api/tools/groups'),
  tools: () => request<{ tools: ToolSummary[] }>('GET', '/api/tools'),
  skills: () => request<{ skills: SkillSummary[] }>('GET', '/api/skills'),
  diagnostics: () => request<HealthStatus>('GET', '/api/health'),
  config: () => request<EffectiveConfig>('GET', '/api/config'),
  configPaths: () => request<ConfigPaths>('GET', '/api/config/paths'),
  prefs: () => request<Record<string, unknown>>('GET', '/api/prefs'),
}
