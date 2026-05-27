// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import { useChatStore } from '../../stores/chatStore'
import { useTaskStore } from '../../stores/taskStore'
import { ChatView, isNearChatBottom, restoredChatScrollTop } from './ChatView'

const runSocketMock = vi.hoisted(() => ({
  connect: vi.fn(),
  onFrame: vi.fn(() => () => undefined),
  startRun: vi.fn(() => 'run-chat-view'),
  cancelRun: vi.fn(),
  respondPermission: vi.fn(),
  respondApproval: vi.fn(),
}))

vi.mock('../../api/runSocket', () => ({
  runSocket: runSocketMock,
}))

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  runSocketMock.connect.mockClear()
  runSocketMock.onFrame.mockClear()
  runSocketMock.startRun.mockClear()
  runSocketMock.cancelRun.mockClear()
  runSocketMock.respondPermission.mockClear()
  runSocketMock.respondApproval.mockClear()
  useChatStore.setState({
    connected: false,
    socketState: 'idle',
    socketDetail: null,
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

  it('clamps restored chat scroll snapshots to the current content height', () => {
    expect(restoredChatScrollTop({ scrollTop: 500, atBottom: false }, { scrollHeight: 1000, clientHeight: 400 })).toBe(500)
    expect(restoredChatScrollTop({ scrollTop: 900, atBottom: false }, { scrollHeight: 1000, clientHeight: 400 })).toBe(600)
    expect(restoredChatScrollTop({ scrollTop: -10, atBottom: false }, { scrollHeight: 1000, clientHeight: 400 })).toBe(0)
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

  it('keeps the latest completed run tokens visible in the composer', () => {
    vi.spyOn(api, 'sessionMessages').mockResolvedValue({ session_id: 'session-1', messages: [] })
    vi.spyOn(api, 'sessionTasks').mockResolvedValue({ tasks: [], active_count: 0, total_count: 0 })
    useChatStore.setState({
      activeRunId: null,
      blocksBySession: {
        'session-1': [{
          id: 'assistant-r1',
          sessionId: 'session-1',
          runId: 'r1',
          timestamp: 1,
          source: 'live',
          kind: 'assistant_message',
          content: 'Done.',
          isStreaming: false,
        }],
      },
      statusByRun: {
        r1: {
          runId: 'r1',
          sessionId: 'session-1',
          state: 'idle',
          tokens: { input_tokens: 100, output_tokens: 23, total_tokens: 123 },
        },
      },
      tokenUsageByRun: {
        r1: { input_tokens: 100, output_tokens: 23, total_tokens: 123 },
      },
    })

    render(
      <ChatView
        session={{
          session_id: 'session-1',
          created_at: 1,
          updated_at: 1,
          provider: 'openai',
          model: 'gpt-5.4',
          message_count: 1,
        }}
      />,
    )

    expect(screen.getByLabelText(/123 tokens/)).toBeTruthy()
  })

  it('keeps the composer visible when no session is selected', () => {
    render(<ChatView session={null} />)

    expect(screen.getByText('Aether')).toBeTruthy()
    expect(screen.getByRole('button', { name: /Inspect project/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Review UI/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Plan edit/ })).toBeTruthy()
    expect(screen.getByRole('textbox')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Send message' })).toBeTruthy()
  })

  it('retries persisted prompts through checkpoint-aware session actions', async () => {
    const session = {
      session_id: 'session-1',
      created_at: 1,
      updated_at: 1,
      provider: 'openai',
      model: 'gpt-5.4',
      message_count: 2,
    }
    vi.spyOn(api, 'sessionMessages').mockResolvedValue({
      session_id: 'session-1',
      messages: [
        { role: 'user', text: 'Patch auth' },
        { role: 'assistant', text: 'Done' },
      ],
    })
    vi.spyOn(api, 'sessionTasks').mockResolvedValue({ tasks: [], active_count: 0, total_count: 0 })
    vi.spyOn(api, 'sessionTurnCheckpoints').mockResolvedValue({
      session_id: 'session-1',
      checkpoints: [{
        target: {
          target_user_message_id: 'turn-1',
          user_message_index: 0,
          user_message_count: 1,
          message_index: 0,
          content: 'Patch auth',
        },
        code: {
          available: true,
          files_changed: ['src/auth.ts'],
          insertions: 2,
          deletions: 1,
          checkpoint_id: 'cp-1',
        },
      }],
    })
    vi.spyOn(api, 'sessionActionRetry').mockResolvedValue({
      action: 'retry_prepared',
      restore: { checkpoint_id: 'cp-1', paths: ['src/auth.ts'] },
      result: {
        session_id: 'session-1',
        rewound_to_index: -1,
        messages_kept: 0,
        messages_removed: 2,
        info: { ...session, message_count: 0 },
        messages: [],
      },
    })
    vi.spyOn(api, 'sessions').mockResolvedValue({ sessions: [session] })

    render(<ChatView session={session} />)

    expect(await screen.findByText('Patch auth')).toBeTruthy()
    await waitFor(() => expect(api.sessionTurnCheckpoints).toHaveBeenCalledWith('session-1'))

    fireEvent.click(screen.getByRole('button', { name: 'Retry prompt' }))

    await waitFor(() => expect(api.sessionActionRetry).toHaveBeenCalledWith('session-1', {
      user_message_index: 0,
      expected_content: 'Patch auth',
      checkpoint_id: 'cp-1',
      paths: ['src/auth.ts'],
    }))
    expect(runSocketMock.startRun).toHaveBeenCalledWith('session-1', 'Patch auth', [])
    expect(await screen.findByText(/Restored 1 file from checkpoint `cp-1`/)).toBeTruthy()
  })

  it('rewinds assistant replies by message index while user prompts use stable turn targets', async () => {
    const session = {
      session_id: 'session-1',
      created_at: 1,
      updated_at: 1,
      provider: 'openai',
      model: 'gpt-5.4',
      message_count: 2,
    }
    vi.spyOn(api, 'sessionMessages').mockResolvedValue({
      session_id: 'session-1',
      messages: [
        { role: 'user', text: 'Patch auth' },
        { role: 'assistant', text: 'Done' },
      ],
    })
    vi.spyOn(api, 'sessionTasks').mockResolvedValue({ tasks: [], active_count: 0, total_count: 0 })
    vi.spyOn(api, 'sessionTurnCheckpoints').mockResolvedValue({ session_id: 'session-1', checkpoints: [] })
    vi.spyOn(api, 'sessionActionRewind').mockResolvedValue({
      action: 'rewind',
      restore: null,
      result: {
        session_id: 'session-1',
        rewound_to_index: 1,
        messages_kept: 2,
        messages_removed: 0,
        info: session,
        messages: [
          { role: 'user', text: 'Patch auth' },
          { role: 'assistant', text: 'Done' },
        ],
      },
    })
    vi.spyOn(api, 'sessions').mockResolvedValue({ sessions: [session] })

    render(<ChatView session={session} />)

    expect(await screen.findByText('Done')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Rewind to reply' }))

    await waitFor(() => expect(api.sessionActionRewind).toHaveBeenCalledWith('session-1', { message_index: 1 }))
  })

  it('stops an active subagent task from the session task bar', async () => {
    const task = {
      task_id: 'task-1',
      parent_session_id: 'session-1',
      subagent_type: 'explorer',
      prompt: 'Inspect renderer',
      status: 'running',
      started_at: 10,
      finished_at: null,
      last_heartbeat: 11,
      model: 'gpt-5.4',
      isolation: null,
      worktree_path: null,
      parent_task_id: null,
      child_depth: 1,
      background: true,
      tool_use_count: 2,
      input_tokens: 100,
      output_tokens: 25,
      iterations: 1,
      summary: null,
      error: null,
      result_path: null,
      output_tail: null,
      metadata: null,
    } as const
    vi.spyOn(api, 'sessionMessages').mockResolvedValue({ session_id: 'session-1', messages: [] })
    vi.spyOn(api, 'sessionTasks').mockResolvedValue({ tasks: [task], active_count: 1, total_count: 1 })
    vi.spyOn(api, 'stopTask').mockResolvedValue({
      task_id: 'task-1',
      delivered: true,
      status: 'running',
      message: 'Stop signal sent to running task.',
    })
    useTaskStore.setState({
      tasksBySession: { 'session-1': [task] },
      isLoadingBySession: {},
      errorBySession: {},
    })

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

    fireEvent.click(screen.getByRole('button', { name: 'Stop task task-1' }))

    await waitFor(() => expect(api.stopTask).toHaveBeenCalledWith('task-1'))
    await waitFor(() => expect(api.sessionTasks).toHaveBeenCalledWith('session-1', { limit: 100 }))
    expect(await screen.findByText('Stop signal sent to running task.')).toBeTruthy()
  })
})
