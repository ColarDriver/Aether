// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { SessionInfo } from '../../api/types'
import { useProviderStore } from '../../stores/providerStore'
import { useSessionStore } from '../../stores/sessionStore'
import { ProviderSettings } from './ProviderSettings'

const initialProviderState = useProviderStore.getState()
const initialSessionState = useSessionStore.getState()

afterEach(() => {
  cleanup()
  useProviderStore.setState({
    ...initialProviderState,
    providers: [],
    current: null,
    preflight: null,
    modelsByProvider: {},
    discoveryByProvider: {},
    isLoading: false,
    error: null,
    loadProviders: vi.fn(),
    loadModels: vi.fn(),
    loadPreflight: vi.fn(),
    selectModel: vi.fn(),
  }, true)
  useSessionStore.setState({
    ...initialSessionState,
    sessions: [],
    activeSessionId: null,
    isLoading: false,
    error: null,
  }, true)
})

describe('ProviderSettings', () => {
  it('renders current provider, credential state, and loaded models', () => {
    useProviderStore.setState({
      providers: [{ name: 'codex', display_name: 'Codex', requires_api_key: true }],
      current: {
        family: 'openai',
        provider_name: 'codex',
        model: 'gpt-5.4',
        api_key_env_names: ['CODEX_API_KEY'],
        model_env_names: [],
        base_url_env_names: [],
        source: 'env',
        credential: { source: 'env', name: 'CODEX_API_KEY', configured: true },
      },
      preflight: {
        family: 'openai-compatible',
        provider_name: 'codex',
        model: 'gpt-5.4',
        status: 'ready',
        ready: true,
        chat_completions_url: null,
        models_url: 'https://api.example/v1/models',
        issues: [],
        suggestions: [],
      },
      modelsByProvider: { codex: [{ id: 'gpt-5.4', display_name: 'GPT-5.4' }] },
      discoveryByProvider: { codex: { kind: 'live', count: 1, url: 'https://api.example/v1/models' } },
      loadProviders: vi.fn(),
      loadModels: vi.fn(),
      loadPreflight: vi.fn(),
      selectModel: vi.fn(),
    })

    render(<ProviderSettings />)

    expect(screen.getByRole('heading', { name: 'Provider and model' })).toBeTruthy()
    expect(screen.getAllByText('codex').length).toBeGreaterThan(0)
    expect(screen.getAllByText('CODEX_API_KEY').length).toBeGreaterThan(0)
    expect(screen.getByText('live · 1 models')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'GPT-5.4' })).toBeTruthy()
  })

  it('applies model selection to the current session', async () => {
    const activeSession: SessionInfo = {
      session_id: 'session-current',
      created_at: 1,
      updated_at: 2,
      provider: 'openai',
      model: 'gpt-5.4',
      base_url: 'https://old.example/v1',
      message_count: 3,
      mode: 'agent',
    }
    const selected = {
      provider: 'codex',
      family: 'openai',
      model: 'gpt-5.5',
      base_url: 'https://codex.example/v1',
      ready: true,
      missing_credentials: [],
      credential: { source: 'env', name: 'CODEX_API_KEY', configured: true },
    }
    const selectModel = vi.fn().mockResolvedValue(selected)
    const updateSession = vi.fn(async (sessionId: string, updates: Partial<Pick<SessionInfo, 'provider' | 'model' | 'base_url' | 'system_prompt'>>) => {
      const updated = { ...activeSession, ...updates, session_id: sessionId }
      useSessionStore.setState({ sessions: [updated], activeSessionId: sessionId })
      return updated
    })

    useProviderStore.setState({
      providers: [{ name: 'codex', display_name: 'Codex', requires_api_key: true }],
      current: {
        family: 'openai',
        provider_name: 'codex',
        model: 'gpt-5.4',
        api_key_env_names: ['CODEX_API_KEY'],
        model_env_names: [],
        base_url_env_names: [],
        source: 'env',
        credential: { source: 'env', name: 'CODEX_API_KEY', configured: true },
      },
      preflight: null,
      modelsByProvider: { codex: [{ id: 'gpt-5.5', display_name: 'GPT-5.5' }] },
      discoveryByProvider: { codex: { kind: 'live', count: 1 } },
      loadProviders: vi.fn(),
      loadModels: vi.fn(),
      loadPreflight: vi.fn(),
      selectModel,
    })
    useSessionStore.setState({
      sessions: [activeSession],
      activeSessionId: activeSession.session_id,
      updateSession,
    })

    render(<ProviderSettings />)
    fireEvent.click(screen.getByRole('button', { name: 'GPT-5.5' }))

    await waitFor(() => expect(selectModel).toHaveBeenCalledWith('codex', 'gpt-5.5'))
    await waitFor(() => expect(updateSession).toHaveBeenCalledWith('session-current', {
      provider: 'codex',
      model: 'gpt-5.5',
      base_url: 'https://codex.example/v1',
    }))
    expect(await screen.findByText(/Updated current session `session-/)).toBeTruthy()
    expect(screen.getByText('codex/gpt-5.5')).toBeTruthy()
  })

  it('does not patch a session when no current session is selected', async () => {
    const selectModel = vi.fn().mockResolvedValue({
      provider: 'codex',
      family: 'openai',
      model: 'gpt-5.5',
      base_url: null,
      ready: true,
      missing_credentials: [],
      credential: null,
    })
    const updateSession = vi.fn()

    useProviderStore.setState({
      providers: [{ name: 'codex', display_name: 'Codex', requires_api_key: true }],
      current: {
        family: 'openai',
        provider_name: 'codex',
        model: 'gpt-5.4',
        api_key_env_names: ['CODEX_API_KEY'],
        model_env_names: [],
        base_url_env_names: [],
        source: 'env',
        credential: null,
      },
      preflight: {
        family: 'openai-compatible',
        provider_name: 'codex',
        model: 'gpt-5.4',
        status: 'warning',
        ready: true,
        chat_completions_url: null,
        models_url: null,
        issues: ['Model discovery failed: HTTP 404'],
        suggestions: ['Check the provider base URL.'],
      },
      modelsByProvider: { codex: [{ id: 'gpt-5.5', display_name: 'GPT-5.5' }] },
      discoveryByProvider: { codex: { kind: 'live', count: 1 } },
      loadProviders: vi.fn(),
      loadModels: vi.fn(),
      loadPreflight: vi.fn(),
      selectModel,
    })
    useSessionStore.setState({
      sessions: [],
      activeSessionId: null,
      updateSession,
    })

    render(<ProviderSettings />)
    fireEvent.click(screen.getByRole('button', { name: 'GPT-5.5' }))

    await waitFor(() => expect(selectModel).toHaveBeenCalledWith('codex', 'gpt-5.5'))
    expect(updateSession).not.toHaveBeenCalled()
    expect(await screen.findByText('Saved default model `codex/gpt-5.5` for new sessions.')).toBeTruthy()
  })
})
