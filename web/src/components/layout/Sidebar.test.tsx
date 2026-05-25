// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Sidebar } from './Sidebar'

const session = {
  session_id: 'session-12345678',
  created_at: 1,
  updated_at: 2,
  provider: 'codex',
  model: 'gpt-5.4',
  message_count: 3,
  summary: 'Plan auth flow',
  mode: 'plan',
}

afterEach(cleanup)

describe('Sidebar', () => {
  it('renders console navigation and session actions', () => {
    const onSelectSession = vi.fn()
    const onSelectView = vi.fn()
    const onNewSession = vi.fn()
    const onDeleteSession = vi.fn()

    render(
      <Sidebar
        sessions={[session]}
        activeSessionId="session-12345678"
        activeView="chat"
        onSelectSession={onSelectSession}
        onSelectView={onSelectView}
        onNewSession={onNewSession}
        onDeleteSession={onDeleteSession}
      />,
    )

    expect(screen.getByText('Aether')).toBeTruthy()
    expect(screen.getByRole('button', { name: /Models/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Settings' })).toBeTruthy()
    expect(screen.getByText('Plan auth flow')).toBeTruthy()
    expect(screen.getByText('Older')).toBeTruthy()
    expect(screen.getByText('plan')).toBeTruthy()
    expect(screen.getByText('3 msgs')).toBeTruthy()

    fireEvent.change(screen.getByPlaceholderText('Search sessions'), { target: { value: 'auth' } })
    expect(screen.getByText('Plan auth flow')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /Models/ }))
    fireEvent.click(screen.getByRole('button', { name: /Plan auth flow.*gpt-5.4.*plan.*3 msgs/ }))
    fireEvent.click(screen.getByRole('button', { name: 'New session' }))

    fireEvent.click(screen.getByRole('button', { name: 'Delete session Plan auth flow' }))
    expect(screen.getByRole('dialog', { name: 'Delete session' })).toBeTruthy()
    expect(screen.getByText('Delete session "Plan auth flow"? This removes its conversation context.')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }))

    expect(onSelectView).toHaveBeenCalledWith('models')
    expect(onSelectSession).toHaveBeenCalledWith('session-12345678')
    expect(onNewSession).toHaveBeenCalledOnce()
    expect(onDeleteSession).toHaveBeenCalledWith('session-12345678')
  })
})
