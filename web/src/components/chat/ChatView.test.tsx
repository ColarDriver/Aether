// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import { useChatStore } from '../../stores/chatStore'
import { useTaskStore } from '../../stores/taskStore'
import { ChatView, isNearChatBottom } from './ChatView'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
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
  useTaskStore.setState({
    tasksBySession: {},
    isLoadingBySession: {},
    errorBySession: {},
  })
})

describe('isNearChatBottom', () => {
  it('treats the scroll position as bottom only within the follow threshold', () => {
    expect(isNearChatBottom({ scrollHeight: 1000, scrollTop: 540, clientHeight: 420 })).toBe(true)
    expect(isNearChatBottom({ scrollHeight: 1000, scrollTop: 500, clientHeight: 420 })).toBe(false)
  })
})

describe('ChatView', () => {
  it('renders an empty session without unstable store-selector fallbacks', async () => {
    vi.spyOn(api, 'sessionMessages').mockResolvedValue({ session_id: 'session-1', messages: [] })
    vi.spyOn(api, 'sessionTasks').mockResolvedValue({ tasks: [], active_count: 0, total_count: 0 })

    render(
      <ChatView
        session={{
          session_id: 'session-1',
          created_at: 1,
          updated_at: 1,
          provider: 'openai',
          model: 'gpt-5.4',
          message_count: 0,
        }}
      />,
    )

    expect(screen.getByText('session-1')).toBeTruthy()
    await waitFor(() => expect(api.sessionMessages).toHaveBeenCalledWith('session-1'))
    expect(api.sessionTasks).toHaveBeenCalledWith('session-1', { limit: 100 })
  })
})
