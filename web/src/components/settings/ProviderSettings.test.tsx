// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useProviderStore } from '../../stores/providerStore'
import { ProviderSettings } from './ProviderSettings'

afterEach(() => {
  cleanup()
  useProviderStore.setState({
    providers: [],
    current: null,
    modelsByProvider: {},
    isLoading: false,
    error: null,
    loadProviders: vi.fn(),
    loadModels: vi.fn(),
    selectModel: vi.fn(),
  })
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
      modelsByProvider: { codex: [{ id: 'gpt-5.4', display_name: 'GPT-5.4' }] },
      loadProviders: vi.fn(),
      loadModels: vi.fn(),
      selectModel: vi.fn(),
    })

    render(<ProviderSettings />)

    expect(screen.getByRole('heading', { name: 'Provider and model' })).toBeTruthy()
    expect(screen.getAllByText('codex').length).toBeGreaterThan(0)
    expect(screen.getByText('configured')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'GPT-5.4' })).toBeTruthy()
  })
})
