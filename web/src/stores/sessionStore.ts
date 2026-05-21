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
  createSession: (input: CreateSessionInput) => Promise<void>
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
      set((state) => ({
        sessions,
        activeSessionId: state.activeSessionId ?? sessions[0]?.session_id ?? null,
        isLoading: false,
      }))
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
  },
  setActiveSession: (activeSessionId) => set({ activeSessionId }),
  setSessionMode: (sessionId, mode) => set((state) => ({
    sessions: state.sessions.map((session) => (
      session.session_id === sessionId ? { ...session, mode } : session
    )),
  })),
}))
