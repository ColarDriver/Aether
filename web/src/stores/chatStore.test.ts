// @vitest-environment jsdom

import { afterEach, describe, expect, it } from 'vitest'
import { useChatStore } from './chatStore'

afterEach(() => {
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
})
