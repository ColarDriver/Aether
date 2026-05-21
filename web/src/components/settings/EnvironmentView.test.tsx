// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import { EnvironmentView } from './EnvironmentView'

const catalog = {
  env_path: '/workspace/Aether/.env',
  variables: [
    {
      key: 'OPENAI_API_KEY',
      is_set: true,
      source: 'file' as const,
      redacted_value: 'sk-t...cret',
      description: 'OpenAI-compatible API key.',
      category: 'provider',
      is_secret: true,
      advanced: false,
      url: null,
    },
    {
      key: 'WEB_SEARCH_PROVIDER',
      is_set: false,
      source: 'missing' as const,
      redacted_value: null,
      description: 'Local web search backend.',
      category: 'tools',
      is_secret: false,
      advanced: false,
      url: null,
    },
  ],
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('EnvironmentView', () => {
  it('renders redacted values and supports reveal/set/delete actions', async () => {
    vi.spyOn(api, 'env').mockResolvedValue(catalog)
    const reveal = vi.spyOn(api, 'revealEnvVar').mockResolvedValue({ key: 'OPENAI_API_KEY', value: 'sk-test-secret', source: 'file' })
    const setEnv = vi.spyOn(api, 'setEnvVar').mockResolvedValue({ ok: true, key: 'WEB_SEARCH_PROVIDER', env_path: catalog.env_path })
    const deleteEnv = vi.spyOn(api, 'deleteEnvVar').mockResolvedValue({ ok: true, key: 'OPENAI_API_KEY', env_path: catalog.env_path })

    render(<EnvironmentView />)

    expect(await screen.findByText('OPENAI_API_KEY')).toBeTruthy()
    expect(screen.getByText('sk-t...cret')).toBeTruthy()

    fireEvent.click(screen.getByTitle('Reveal value'))
    expect(await screen.findByText('sk-test-secret')).toBeTruthy()
    expect(reveal).toHaveBeenCalledWith('OPENAI_API_KEY')

    fireEvent.click(screen.getByText('Set'))
    fireEvent.change(screen.getByLabelText('Value for WEB_SEARCH_PROVIDER'), { target: { value: 'brave' } })
    fireEvent.click(screen.getByTitle('Save'))
    await waitFor(() => expect(setEnv).toHaveBeenCalledWith({ key: 'WEB_SEARCH_PROVIDER', value: 'brave' }))

    fireEvent.click(screen.getByTitle('Delete'))
    await waitFor(() => expect(deleteEnv).toHaveBeenCalledWith('OPENAI_API_KEY'))
  })
})
