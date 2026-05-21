// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { useChatStore } from './chatStore'

const runSocketMock = vi.hoisted(() => ({
  connect: vi.fn(),
  startRun: vi.fn(() => 'run-attach'),
  cancelRun: vi.fn(),
  onFrame: vi.fn(() => () => undefined),
  respondPermission: vi.fn(),
  respondApproval: vi.fn(),
}))

vi.mock('../api/runSocket', () => ({
  runSocket: runSocketMock,
}))

afterEach(() => {
  vi.clearAllMocks()
  useChatStore.setState({
    connected: false,
    frames: [],
    activeRunId: null,
    blocksBySession: {},
    tokenUsageByRun: {},
    statusByRun: {},
    pendingPermission: null,
    pendingApproval: null,
  })
})

describe('chatStore local timeline blocks', () => {
  it('appends local notices and errors into the session timeline', () => {
    useChatStore.getState().appendLocalNotice('s1', '# Help')
    useChatStore.getState().appendLocalError('s1', 'bad command')

    expect(useChatStore.getState().blocksBySession.s1?.map((block) => block.kind)).toEqual([
      'system_notice',
      'error',
    ])
    expect(useChatStore.getState().blocksBySession.s1?.[0]).toMatchObject({ content: '# Help' })
    expect(useChatStore.getState().blocksBySession.s1?.[1]).toMatchObject({ message: 'bad command' })
  })

  it('adds optimistic attachments and sends them through the run socket', () => {
    const attachments = [{ type: 'file' as const, name: 'notes.md', path: 'notes.md' }]

    const runId = useChatStore.getState().startRun('s1', 'read this', attachments)

    expect(runId).toBe('run-attach')
    expect(runSocketMock.startRun).toHaveBeenCalledWith('s1', 'read this', attachments)
    expect(useChatStore.getState().blocksBySession.s1?.[0]).toMatchObject({
      kind: 'user_message',
      content: 'read this',
      attachments,
    })
  })
})
