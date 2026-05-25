// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import { useSessionStore } from './sessionStore'

const baseSession = {
  session_id: 'session-1',
  created_at: 1,
  updated_at: 2,
  provider: 'openai',
  model: 'gpt-5.4',
  message_count: 0,
  mode: 'agent',
}

afterEach(() => {
  vi.restoreAllMocks()
  useSessionStore.setState({
    sessions: [],
    activeSessionId: null,
    isLoading: false,
    error: null,
  })
})

describe('sessionStore', () => {
  it('drops a stale active session when loading from the server', async () => {
    vi.spyOn(api, 'sessions').mockResolvedValue({ sessions: [{ ...baseSession, session_id: 'session-2' }] })
    useSessionStore.setState({
      sessions: [baseSession],
      activeSessionId: 'session-1',
      isLoading: false,
      error: null,
    })

    await useSessionStore.getState().loadSessions()

    expect(useSessionStore.getState().sessions.map((session) => session.session_id)).toEqual(['session-2'])
    expect(useSessionStore.getState().activeSessionId).toBe('session-2')
  })

  it('creates, resumes, updates, and deletes sessions while keeping local state current', async () => {
    vi.spyOn(api, 'createSession').mockResolvedValue(baseSession)
    vi.spyOn(api, 'sessions').mockResolvedValue({ sessions: [{ ...baseSession, session_id: 'session-2' }, baseSession] })
    vi.spyOn(api, 'resumeSession').mockResolvedValue({
      session_id: 'session-2',
      info: { ...baseSession, session_id: 'session-2' },
      messages: [],
    })
    vi.spyOn(api, 'updateSession').mockResolvedValue({ ...baseSession, model: 'gpt-5.4-mini' })
    vi.spyOn(api, 'deleteSession').mockResolvedValue(undefined)

    const created = await useSessionStore.getState().createSession({ provider: 'openai', model: 'gpt-5.4' })
    expect(created.session_id).toBe('session-1')
    expect(useSessionStore.getState().activeSessionId).toBe('session-1')

    const resumed = await useSessionStore.getState().resumeSession('session-2')
    expect(resumed.session_id).toBe('session-2')
    expect(useSessionStore.getState().activeSessionId).toBe('session-2')

    const updated = await useSessionStore.getState().updateSession('session-1', { model: 'gpt-5.4-mini' })
    expect(updated.model).toBe('gpt-5.4-mini')
    expect(api.updateSession).toHaveBeenCalledWith('session-1', {
      provider: undefined,
      model: 'gpt-5.4-mini',
      base_url: undefined,
      system_prompt: undefined,
      update_base_url: false,
      update_system_prompt: false,
    })

    await useSessionStore.getState().deleteSession('session-2')
    expect(api.deleteSession).toHaveBeenCalledWith('session-2')
    expect(useSessionStore.getState().sessions.map((session) => session.session_id)).toEqual(['session-1'])
    expect(useSessionStore.getState().activeSessionId).toBe('session-1')
  })
})
