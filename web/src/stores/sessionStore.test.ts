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
    vi.spyOn(api, 'forkSession').mockResolvedValue({
      source_session_id: 'session-2',
      forked_from_index: 1,
      messages_copied: 2,
      info: { ...baseSession, session_id: 'session-fork' },
      messages: [],
    })
    vi.spyOn(api, 'rewindSession').mockResolvedValue({
      session_id: 'session-fork',
      rewound_to_index: 0,
      messages_kept: 1,
      messages_removed: 1,
      info: { ...baseSession, session_id: 'session-fork', message_count: 1 },
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

    const forked = await useSessionStore.getState().forkSession('session-2', 1)
    expect(forked.session_id).toBe('session-fork')
    expect(api.forkSession).toHaveBeenCalledWith('session-2', { message_index: 1 })
    expect(useSessionStore.getState().activeSessionId).toBe('session-fork')

    const rewound = await useSessionStore.getState().rewindSession('session-fork', 0)
    expect(rewound.messages_kept).toBe(1)
    expect(api.rewindSession).toHaveBeenCalledWith('session-fork', { message_index: 0 })
    expect(useSessionStore.getState().sessions.find((session) => session.session_id === 'session-fork')?.message_count).toBe(1)
    expect(useSessionStore.getState().activeSessionId).toBe('session-fork')

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
    expect(useSessionStore.getState().sessions.map((session) => session.session_id)).toEqual(['session-fork', 'session-1'])
    expect(useSessionStore.getState().activeSessionId).toBe('session-fork')
  })

  it('renames and imports sessions while preserving active selection semantics', async () => {
    vi.spyOn(api, 'sessions').mockResolvedValue({ sessions: [] })
    vi.spyOn(api, 'renameSession').mockResolvedValue({ ...baseSession, session_id: 'session-renamed' })
    vi.spyOn(api, 'importSession').mockResolvedValue({
      source_session_id: 'session-exported',
      overwritten: false,
      info: { ...baseSession, session_id: 'session-imported' },
      messages: [],
    })
    useSessionStore.setState({
      sessions: [baseSession],
      activeSessionId: 'session-1',
      isLoading: false,
      error: null,
    })

    const renamed = await useSessionStore.getState().renameSession('session-1', 'session-renamed')
    expect(renamed.session_id).toBe('session-renamed')
    expect(api.renameSession).toHaveBeenCalledWith('session-1', 'session-renamed')
    expect(useSessionStore.getState().activeSessionId).toBe('session-renamed')
    expect(useSessionStore.getState().sessions.map((session) => session.session_id)).toContain('session-renamed')

    const imported = await useSessionStore.getState().importSession({ data: { session_id: 'session-exported' }, newSessionId: 'session-imported' })
    expect(imported.info.session_id).toBe('session-imported')
    expect(api.importSession).toHaveBeenCalledWith({
      data: { session_id: 'session-exported' },
      new_session_id: 'session-imported',
      overwrite: undefined,
      make_current: undefined,
    })
    expect(useSessionStore.getState().activeSessionId).toBe('session-imported')

    vi.mocked(api.importSession).mockResolvedValueOnce({
      source_session_id: 'session-exported',
      overwritten: false,
      info: { ...baseSession, session_id: 'session-imported-passive' },
      messages: [],
    })
    await useSessionStore.getState().importSession({
      data: { session_id: 'session-exported' },
      newSessionId: 'session-imported-passive',
      makeCurrent: false,
    })
    expect(api.importSession).toHaveBeenLastCalledWith({
      data: { session_id: 'session-exported' },
      new_session_id: 'session-imported-passive',
      overwrite: undefined,
      make_current: false,
    })
    expect(useSessionStore.getState().activeSessionId).toBe('session-imported')
  })
})
