import { create } from 'zustand'
import { api } from '../api/client'
import type { SessionImportResult, SessionInfo, SessionRewindResult } from '../api/types'

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
  forkSession: (sessionId: string, messageIndex: number) => Promise<SessionInfo>
  rewindSession: (sessionId: string, messageIndex: number) => Promise<SessionRewindResult>
  renameSession: (sessionId: string, newSessionId: string) => Promise<SessionInfo>
  importSession: (input: { data: Record<string, unknown>; newSessionId?: string | null; overwrite?: boolean; makeCurrent?: boolean }) => Promise<SessionImportResult>
  updateSession: (sessionId: string, updates: Partial<Pick<SessionInfo, 'provider' | 'model' | 'base_url' | 'system_prompt'>>) => Promise<SessionInfo>
  deleteSession: (sessionId: string) => Promise<void>
  setActiveSession: (sessionId: string | null) => void
  setSessionMode: (sessionId: string, mode: 'agent' | 'plan') => void
}

export const useSessionStore = create<SessionState>((set) => ({
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
    void refreshSessionsAfterMutation(set, created.session_id)
    return created
  },
  resumeSession: async (sessionId) => {
    const resumed = await api.resumeSession(sessionId)
    const info = resumed.info
    set((state) => ({
      sessions: [info, ...state.sessions.filter((session) => session.session_id !== info.session_id)],
      activeSessionId: info.session_id,
    }))
    void refreshSessionsAfterMutation(set, info.session_id)
    return info
  },
  forkSession: async (sessionId, messageIndex) => {
    const forked = await api.forkSession(sessionId, { message_index: messageIndex })
    const info = forked.info
    set((state) => ({
      sessions: [info, ...state.sessions.filter((session) => session.session_id !== info.session_id)],
      activeSessionId: info.session_id,
    }))
    void refreshSessionsAfterMutation(set, info.session_id)
    return info
  },
  rewindSession: async (sessionId, messageIndex) => {
    const rewound = await api.rewindSession(sessionId, { message_index: messageIndex })
    const info = rewound.info
    set((state) => ({
      sessions: state.sessions.map((session) => (
        session.session_id === info.session_id ? info : session
      )),
      activeSessionId: info.session_id,
    }))
    void refreshSessionsAfterMutation(set, info.session_id)
    return rewound
  },
  renameSession: async (sessionId, newSessionId) => {
    const renamed = await api.renameSession(sessionId, newSessionId)
    set((state) => ({
      sessions: [renamed, ...state.sessions.filter((session) => session.session_id !== sessionId && session.session_id !== renamed.session_id)],
      activeSessionId: state.activeSessionId === sessionId ? renamed.session_id : state.activeSessionId,
    }))
    void refreshSessionsAfterMutation(set, renamed.session_id)
    return renamed
  },
  importSession: async (input) => {
    const imported = await api.importSession({
      data: input.data,
      new_session_id: input.newSessionId,
      overwrite: input.overwrite,
      make_current: input.makeCurrent,
    })
    const info = imported.info
    set((state) => ({
      sessions: [info, ...state.sessions.filter((session) => session.session_id !== info.session_id)],
      activeSessionId: input.makeCurrent === false ? state.activeSessionId : info.session_id,
    }))
    void refreshSessionsAfterMutation(set, info.session_id)
    return imported
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

async function refreshSessionsAfterMutation(
  set: (partial: Partial<SessionState> | ((state: SessionState) => Partial<SessionState>)) => void,
  expectedActiveSessionId: string,
): Promise<void> {
  try {
    const { sessions } = await api.sessions()
    set((state) => mergeSessionsAfterMutation(state, sessions, expectedActiveSessionId))
  } catch {
    // The mutation already succeeded and updated local state. A background
    // reconciliation failure should not make the selected session jump away.
  }
}

function mergeSessionsAfterMutation(
  state: SessionState,
  serverSessions: SessionInfo[],
  expectedActiveSessionId: string,
): Partial<SessionState> {
  const preserveIds = [expectedActiveSessionId, state.activeSessionId].filter((item): item is string => Boolean(item))
  const preserved = preserveIds
    .map((sessionId) => state.sessions.find((session) => session.session_id === sessionId))
    .filter((session): session is SessionInfo => Boolean(session))
    .filter((session, index, sessions) => sessions.findIndex((item) => item.session_id === session.session_id) === index)
    .filter((session) => !serverSessions.some((serverSession) => serverSession.session_id === session.session_id))
  const sessions = [...preserved, ...serverSessions.filter((session) => !preserved.some((item) => item.session_id === session.session_id))]
  const activeStillExists = sessions.some((session) => session.session_id === state.activeSessionId)

  return {
    sessions,
    activeSessionId: activeStillExists ? state.activeSessionId : sessions[0]?.session_id ?? null,
  }
}
