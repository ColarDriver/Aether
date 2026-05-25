// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest'
import type { RunSocketFrame } from '../api/types'
import { useChatStore } from './chatStore'

const socketHarness = vi.hoisted(() => ({
  handler: null as ((frame: RunSocketFrame) => void) | null,
}))

const runSocketMock = vi.hoisted(() => ({
  connect: vi.fn(),
  startRun: vi.fn(() => 'run-attach'),
  cancelRun: vi.fn(),
  onFrame: vi.fn((handler: (frame: RunSocketFrame) => void) => {
    socketHarness.handler = handler
    return () => {
      socketHarness.handler = null
    }
  }),
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
    pendingPermissionQueue: [],
    pendingApprovalQueue: [],
  })
})

describe('chatStore prompt queue', () => {
  it('keeps multiple permission prompts queued and advances after a response', () => {
    dispatchFrame(permissionFrame('perm-1', 's1', { tool_name: 'write_file', arguments: { path: 'a.py' } }))
    dispatchFrame(permissionFrame('perm-2', 's1', { tool_name: 'shell', arguments: { command: 'pytest' } }))

    expect(useChatStore.getState().pendingPermission?.promptId).toBe('perm-1')
    expect(useChatStore.getState().pendingPermissionQueue.map((prompt) => prompt.promptId)).toEqual([
      'perm-1',
      'perm-2',
    ])

    useChatStore.getState().respondPermission({ type: 'allow_once' })

    expect(runSocketMock.respondPermission).toHaveBeenCalledWith('perm-1', { type: 'allow_once' })
    expect(useChatStore.getState().pendingPermission?.promptId).toBe('perm-2')
    expect(useChatStore.getState().pendingPermissionQueue.map((prompt) => prompt.promptId)).toEqual(['perm-2'])
    expect(useChatStore.getState().blocksBySession.s1?.[0]).toMatchObject({
      kind: 'permission_request',
      promptId: 'perm-1',
      state: 'allowed',
    })
  })

  it('resolves only the matching prompt and preserves unrelated prompts', () => {
    dispatchFrame(permissionFrame('perm-1', 's1', { tool_name: 'write_file' }))
    dispatchFrame(permissionFrame('perm-2', 's1', { tool_name: 'shell' }))
    dispatchFrame(approvalFrame('approval-1', 's1'))

    dispatchFrame(frame('prompt.resolved', { prompt_id: 'perm-2', result: { decision: { type: 'deny' } } }))

    expect(useChatStore.getState().pendingPermission?.promptId).toBe('perm-1')
    expect(useChatStore.getState().pendingApproval?.promptId).toBe('approval-1')
    expect(useChatStore.getState().pendingPermissionQueue.map((prompt) => prompt.promptId)).toEqual(['perm-1'])
    expect(useChatStore.getState().pendingApprovalQueue.map((prompt) => prompt.promptId)).toEqual(['approval-1'])
    expect(useChatStore.getState().blocksBySession.s1?.find((block) => block.id === 'permission-perm-2')).toMatchObject({
      kind: 'permission_request',
      state: 'denied',
    })

    dispatchFrame(frame('prompt.resolved', { prompt_id: 'approval-1', result: { confirmed: true } }))

    expect(useChatStore.getState().pendingPermission?.promptId).toBe('perm-1')
    expect(useChatStore.getState().pendingApproval).toBeNull()
    expect(useChatStore.getState().pendingApprovalQueue).toEqual([])
  })

  it('replaces duplicate replayed prompt ids instead of duplicating queue entries', () => {
    dispatchFrame(permissionFrame('perm-1', 's1', { tool_name: 'write_file', reason: 'old' }))
    dispatchFrame(permissionFrame('perm-1', 's1', { tool_name: 'write_file', reason: 'new' }))

    expect(useChatStore.getState().pendingPermissionQueue).toHaveLength(1)
    expect(useChatStore.getState().pendingPermission).toMatchObject({
      promptId: 'perm-1',
      request: { reason: 'new' },
    })
    expect(useChatStore.getState().blocksBySession.s1?.filter((block) => block.id === 'permission-perm-1')).toHaveLength(1)
  })

  it('clears queued prompts only for the cleared session', () => {
    dispatchFrame(permissionFrame('perm-s1', 's1', { tool_name: 'write_file' }))
    dispatchFrame(permissionFrame('perm-s2', 's2', { tool_name: 'shell' }))
    dispatchFrame(approvalFrame('approval-s1', 's1'))
    dispatchFrame(approvalFrame('approval-s2', 's2'))

    useChatStore.getState().clearSession('s1')

    expect(useChatStore.getState().pendingPermission?.promptId).toBe('perm-s2')
    expect(useChatStore.getState().pendingApproval?.promptId).toBe('approval-s2')
    expect(useChatStore.getState().pendingPermissionQueue.map((prompt) => prompt.promptId)).toEqual(['perm-s2'])
    expect(useChatStore.getState().pendingApprovalQueue.map((prompt) => prompt.promptId)).toEqual(['approval-s2'])
    expect(useChatStore.getState().blocksBySession.s1).toEqual([])
    expect(useChatStore.getState().blocksBySession.s2?.map((block) => block.kind)).toEqual([
      'permission_request',
      'approval_request',
    ])
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

  it('clears the visible session timeline and pending prompts', () => {
    useChatStore.getState().appendLocalNotice('s1', 'old content')
    const permissionPrompt = {
      promptId: 'perm-1',
      sessionId: 's1',
      runId: 'run-1',
      request: { tool_name: 'write_file' },
    }
    const approvalPrompt = {
      promptId: 'approval-1',
      kind: 'plan',
      sessionId: 's1',
      runId: 'run-1',
      questions: [],
    }
    useChatStore.setState({
      activeRunId: 'run-1',
      pendingPermission: permissionPrompt,
      pendingApproval: approvalPrompt,
      pendingPermissionQueue: [permissionPrompt],
      pendingApprovalQueue: [approvalPrompt],
    })

    useChatStore.getState().clearSession('s1')

    expect(useChatStore.getState().blocksBySession.s1).toEqual([])
    expect(useChatStore.getState().activeRunId).toBeNull()
    expect(useChatStore.getState().pendingPermission).toBeNull()
    expect(useChatStore.getState().pendingApproval).toBeNull()
    expect(useChatStore.getState().pendingPermissionQueue).toEqual([])
    expect(useChatStore.getState().pendingApprovalQueue).toEqual([])
  })
})

function dispatchFrame(nextFrame: RunSocketFrame) {
  if (!socketHarness.handler) useChatStore.getState().connect()
  expect(socketHarness.handler).toBeTypeOf('function')
  socketHarness.handler?.(nextFrame)
}

function permissionFrame(promptId: string, sessionId: string, request: Record<string, unknown>): RunSocketFrame {
  return frame('permission.requested', {
    session_id: sessionId,
    run_id: 'run-' + sessionId,
    prompt_id: promptId,
    request: {
      session_id: sessionId,
      ...request,
    },
  })
}

function approvalFrame(promptId: string, sessionId: string): RunSocketFrame {
  return frame('approval.requested', {
    session_id: sessionId,
    run_id: 'run-' + sessionId,
    prompt_id: promptId,
    kind: 'plan',
    plan_text: '# Plan',
  })
}

function frame(type: string, payload: Record<string, unknown>): RunSocketFrame {
  return { type, payload, transport_sequence: 1 }
}
