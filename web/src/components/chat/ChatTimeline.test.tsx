// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ChatBlock } from '../../chat-rendering'
import { ChatTimeline } from './ChatTimeline'

const base = {
  sessionId: 's1',
  runId: 'r1',
  timestamp: 1,
  source: 'live' as const,
}

afterEach(cleanup)

describe('ChatTimeline', () => {
  it('renders user, assistant, thinking, tool, result, and diff blocks in one timeline', () => {
    const blocks: ChatBlock[] = [
      { ...base, id: 'u', kind: 'user_message', content: 'hello' },
      { ...base, id: 'a', kind: 'assistant_message', content: 'I will read.' },
      { ...base, id: 't', kind: 'thinking', content: 'Need inspect', isActive: true },
      { ...base, id: 'tc', kind: 'tool_call', toolCallId: 'call-1', toolName: 'read_file', arguments: { path: 'README.md' }, status: 'finished' },
      { ...base, id: 'tr', kind: 'tool_result', toolCallId: 'call-1', toolName: 'read_file', content: 'contents', isError: false, metadata: {} },
    ]

    render(<ChatTimeline blocks={blocks} />)

    expect(screen.getByText('hello')).toBeTruthy()
    expect(screen.getByText('I will read.')).toBeTruthy()
    expect(screen.getByText(/thinking/)).toBeTruthy()
    expect(screen.getByText('read_file')).toBeTruthy()
    expect(screen.getByText('contents')).toBeTruthy()
  })

  it('renders prompt blocks with actions', () => {
    const onPermission = vi.fn()
    const onApproval = vi.fn()
    const blocks: ChatBlock[] = [
      { ...base, id: 'p', kind: 'permission_request', promptId: 'p1', toolName: 'write_file', arguments: {}, allowSession: true, state: 'pending' },
      { ...base, id: 'ap', kind: 'approval_request', promptId: 'a1', approvalKind: 'plan', planText: '# Plan', questions: [], state: 'pending' },
    ]

    render(<ChatTimeline blocks={blocks} onRespondPermission={onPermission} onRespondApproval={onApproval} />)

    fireEvent.click(screen.getByRole('button', { name: 'Allow once' }))
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }))

    expect(onPermission).toHaveBeenCalledWith({ type: 'allow_once' })
    expect(onApproval).toHaveBeenCalledWith({ confirmed: true })
  })

  it('only renders allow-session when the permission block allows it', () => {
    render(
      <ChatTimeline
        blocks={[
          { ...base, id: 'p', kind: 'permission_request', promptId: 'p1', toolName: 'shell', arguments: {}, state: 'pending' },
        ]}
        onRespondPermission={() => undefined}
      />,
    )

    expect(screen.queryByRole('button', { name: 'Allow session' })).toBeNull()
  })
})
