import { readFileSync } from 'node:fs'

import { atom } from 'nanostores'

import type { GatewayReady, SessionInfo, SlashCommandInfo } from '../gatewayTypes.js'
import { envBaseUrl } from '../lib/sessionBaseUrl.js'

export type RunStatus = 'starting' | 'idle' | 'thinking' | 'responding' | 'tool_use'

export type SessionState = {
  sessionId: string | null
  provider: string
  model: string
  recentSessions: SessionInfo[]
  bannerTools: string[]
  bannerToolCount: number
  bannerSkills: string[]
  bannerSkillCount: number
  status: RunStatus
  statusDetail: string | null
  usage: {
    input: number
    output: number
    cacheRead: number
    cacheWrite: number
  }
  systemOverride: string | null
  gatewayReady: GatewayReady | null
  catalog: SlashCommandInfo[]
  /**
   * Runtime knobs sourced from CLI flags / env vars at boot. Forwarded into
   * every `agent.run` request so a `--temperature 0.2` invocation stays
   * sticky across turns (matches Python `aether --temperature 0.2`).
   */
  maxIterations: number | null
  temperature: number | null
  maxTokens: number | null
  disableBuiltinTools: boolean
  apiKey: string | null
  baseUrl: string | null
}

function readPositiveInt(envKey: string): number | null {
  const raw = process.env[envKey]
  if (!raw) {
    return null
  }
  const value = Number.parseInt(raw, 10)
  return Number.isFinite(value) && value > 0 ? value : null
}

function readFloat(envKey: string): number | null {
  const raw = process.env[envKey]
  if (!raw) {
    return null
  }
  const value = Number.parseFloat(raw)
  return Number.isFinite(value) ? value : null
}

function readSystemOverride(): string | null {
  if (process.env.AETHER_SYSTEM) {
    return process.env.AETHER_SYSTEM
  }
  const path = process.env.AETHER_SYSTEM_FILE
  if (!path) {
    return null
  }
  try {
    // Synchronously load the file at boot — small operator override, not a
    // hot path. Failure here falls back to the gateway default.
    return readFileSync(path, 'utf8').trim()
  } catch {
    return null
  }
}

const initialProvider = process.env.AETHER_PROVIDER || 'openai'

const initialState: SessionState = {
  sessionId: null,
  provider: initialProvider,
  model: process.env.AETHER_MODEL || '',
  recentSessions: [],
  bannerTools: [],
  bannerToolCount: 0,
  bannerSkills: [],
  bannerSkillCount: 0,
  status: 'starting',
  statusDetail: null,
  usage: {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0
  },
  systemOverride: readSystemOverride(),
  gatewayReady: null,
  catalog: [],
  maxIterations: readPositiveInt('AETHER_MAX_ITERATIONS'),
  temperature: readFloat('AETHER_TEMPERATURE'),
  maxTokens: readPositiveInt('AETHER_MAX_TOKENS'),
  disableBuiltinTools: process.env.AETHER_NO_BUILTIN_TOOLS === '1',
  apiKey: process.env.AETHER_API_KEY || null,
  baseUrl: envBaseUrl(initialProvider)
}

export const sessionState = atom<SessionState>(initialState)

export const sessionActions = {
  resetForTests(): void {
    sessionState.set(initialState)
  },

  setGatewayReady(ready: GatewayReady): void {
    sessionState.set({ ...sessionState.get(), gatewayReady: ready })
  },

  setCatalog(catalog: SlashCommandInfo[]): void {
    sessionState.set({ ...sessionState.get(), catalog })
  },

  setBannerData(input: {
    recentSessions?: SessionInfo[]
    bannerTools?: string[]
    bannerToolCount?: number
    bannerSkills?: string[]
    bannerSkillCount?: number
  }): void {
    const current = sessionState.get()
    sessionState.set({
      ...current,
      recentSessions: input.recentSessions ?? current.recentSessions,
      bannerTools: input.bannerTools ?? current.bannerTools,
      bannerToolCount: input.bannerToolCount ?? current.bannerToolCount,
      bannerSkills: input.bannerSkills ?? current.bannerSkills,
      bannerSkillCount: input.bannerSkillCount ?? current.bannerSkillCount
    })
  },

  setSession(info: SessionInfo | null, sessionId?: string | null): void {
    const current = sessionState.get()
    if (!info) {
      sessionState.set({ ...current, sessionId: sessionId ?? null })
      return
    }
    sessionState.set({
      ...current,
      sessionId: info.session_id || sessionId || current.sessionId,
      provider: info.provider || current.provider,
      model: info.model || current.model,
      baseUrl: info.base_url ?? current.baseUrl,
      systemOverride: null
    })
  },

  setProviderModel(provider: string, model: string, sessionId?: string): void {
    const current = sessionState.get()
    sessionState.set({
      ...current,
      provider,
      model,
      sessionId: sessionId ?? current.sessionId,
      systemOverride: null
    })
  },

  setStatus(status: RunStatus, detail: string | null = null): void {
    sessionState.set({ ...sessionState.get(), status, statusDetail: detail })
  },

  addUsage(input: {
    inputTokens: number
    outputTokens: number
    cacheReadTokens?: number
    cacheWriteTokens?: number
  }): void {
    const current = sessionState.get()
    sessionState.set({
      ...current,
      usage: {
        input: current.usage.input + input.inputTokens,
        output: current.usage.output + input.outputTokens,
        cacheRead: current.usage.cacheRead + (input.cacheReadTokens ?? 0),
        cacheWrite: current.usage.cacheWrite + (input.cacheWriteTokens ?? 0)
      }
    })
  },

  setSystemOverride(systemOverride: string | null): void {
    sessionState.set({ ...sessionState.get(), systemOverride })
  }
}
