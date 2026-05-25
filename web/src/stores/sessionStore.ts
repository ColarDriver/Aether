import { create } from 'zustand'
import { api } from '../api/client'
import type { SessionInfo } from '../api/types'

type CreateSessionInput = {
  provider: string
  model: string
}

type SessionState = {
  sessions: SessionInfo[]
  activeSessionId: string | null
  isLoading: boolean
  error: string | null
  loadSessions: () => Promise<void>
  createSession: (input: CreateSessionInput) => Promise<SessionInfo>
  resumeSession: (sessionId: string) => Promise<SessionInfo>
  updateSession: (sessionId: string, updates: Partial<Pick<SessionInfo, 'provider' | 'model' | 'base_url' | 'system_prompt'>>) => Promise<SessionInfo>
  deleteSession: (sessionId: string) => Promise<void>
  setActiveSession: (sessionId: string | null) => void
  setSessionMode: (sessionId: string, mode: 'agent' | 'plan') => void
}

export const useSessionStore = create<SessionState>((set, get) => ({
  sessions: [],
  activeSessionId: null,
  isLoading: false,
  error: null,
  loadSessions: async () => {
    set({ isLoading: true, error: null })
    try {
      const { sessions } = await api.sessions()
      set((state) => {
        const activeStillExists = sessions.some((session) => session.session_id === state.activeSessionId)
        return {
          sessions,
          activeSessionId: activeStillExists ? state.activeSessionId : sessions[0]?.session_id ?? null,
          isLoading: false,
        }
      })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error), isLoading: false })
    }
  },
  createSession: async (input) => {
    const created = await api.createSession({ provider: input.provider, model: input.model })
    set((state) => ({
      sessions: [created, ...state.sessions.filter((session) => session.session_id !== created.session_id)],
      activeSessionId: created.session_id,
    }))
    void get().loadSessions()
    return created
  },
  resumeSession: async (sessionId) => {
    const resumed = await api.resumeSession(sessionId)
    const info = resumed.info
    set((state) => ({
      sessions: [info, ...state.sessions.filter((session) => session.session_id !== info.session_id)],
      activeSessionId: info.session_id,
    }))
    void get().loadSessions()
    return info
  },
  updateSession: async (sessionId, updates) => {
    const updated = await api.updateSession(sessionId, {
      provider: updates.provider,
      model: updates.model,
      base_url: updates.base_url,
      system_prompt: updates.system_prompt,
      update_base_url: Object.prototype.hasOwnProperty.call(updates, 'base_url'),
      update_system_prompt: Object.prototype.hasOwnProperty.call(updates, 'system_prompt'),
    })
    set((state) => ({
      sessions: state.sessions.map((session) => (
        session.session_id === updated.session_id ? updated : session
      )),
      activeSessionId: state.activeSessionId === sessionId ? updated.session_id : state.activeSessionId,
    }))
    return updated
  },
  deleteSession: async (sessionId) => {
    await api.deleteSession(sessionId)
    set((state) => {
      const sessions = state.sessions.filter((session) => session.session_id !== sessionId)
      return {
        sessions,
        activeSessionId: state.activeSessionId === sessionId ? sessions[0]?.session_id ?? null : state.activeSessionId,
      }
    })
  },
  setActiveSession: (activeSessionId) => set({ activeSessionId }),
  setSessionMode: (sessionId, mode) => set((state) => ({
    sessions: state.sessions.map((session) => (
      session.session_id === sessionId ? { ...session, mode } : session
    )),
  })),
}))
