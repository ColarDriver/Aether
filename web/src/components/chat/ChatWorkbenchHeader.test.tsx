// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ChatWorkbenchHeader } from './ChatWorkbenchHeader'

afterEach(cleanup)

describe('ChatWorkbenchHeader', () => {
  it('renders session metadata and toggles the workspace panel', () => {
    const onToggleWorkspaceRail = vi.fn()
    const onSwapPanels = vi.fn()

    render(
      <ChatWorkbenchHeader
        session={{
          session_id: 'session-abcdef12',
          created_at: 1,
          updated_at: 2,
          provider: 'codex',
          model: 'gpt-5.4',
          message_count: 4,
          summary: 'Refine web UI',
          mode: 'plan',
        }}
        online
        socketState="reconnecting"
        socketDetail="Reconnecting run stream"
        workspaceRailOpen
        onToggleWorkspaceRail={onToggleWorkspaceRail}
        onSwapPanels={onSwapPanels}
      />,
    )

    expect(screen.getByText('Refine web UI')).toBeTruthy()
    expect(screen.getByText('4 messages')).toBeTruthy()
    expect(screen.getByText('online')).toBeTruthy()
    expect(screen.getByText('reconnecting')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Swap sessions and workspace panels' }))
    expect(onSwapPanels).toHaveBeenCalledOnce()

    fireEvent.click(screen.getByRole('button', { name: 'Hide workspace panel' }))
    expect(onToggleWorkspaceRail).toHaveBeenCalledOnce()
  })
})
