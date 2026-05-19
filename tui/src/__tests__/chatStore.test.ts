import { beforeEach, describe, expect, it } from 'vitest'

import { applyGatewayEvent, resetGatewayEventDedupeForTests } from '../hooks/useGatewayEvents.js'
import { activityActions, activityState } from '../store/activityStore.js'
import { chatActions, chatItems } from '../store/chatStore.js'
import { sessionActions, sessionState } from '../store/sessionStore.js'

describe('chat store event mapping', () => {
  beforeEach(() => {
    chatActions.resetForTests()
    activityActions.resetForTests()
    sessionActions.resetForTests()
    resetGatewayEventDedupeForTests()
  })

  it('appends streaming text deltas and marks done', () => {
    applyGatewayEvent({
      type: 'text.delta',
      session_id: 's1',
      run_id: 'r1',
      text: 'hel',
      sequence: 0
    })
    applyGatewayEvent({
      type: 'text.delta',
      session_id: 's1',
      run_id: 'r1',
      text: 'lo',
      sequence: 1
    })
    applyGatewayEvent({
      type: 'done',
      session_id: 's1',
      run_id: 'r1',
      final_text: 'hello',
      exit_reason: 'done'
    })

    const items = chatItems.get()
    expect(items[0]).toMatchObject({
      kind: 'assistant',
      runId: 'r1',
      text: 'hello',
      streaming: false
    })
    // The done event also appends the per-turn footer note (icon + verb).
    expect(
      items.some((item) => item.kind === 'note' && /^[✓⏹✗] done\b/.test(item.text))
    ).toBe(true)
    expect(sessionState.get().status).toBe('idle')
  })

  it('counts silent stream progress without appending assistant text', () => {
    applyGatewayEvent({
      type: 'stream.progress',
      session_id: 's1',
      run_id: 'r1',
      chars: 24,
      sequence: 0
    })

    expect(activityState.get().responseChars).toBe(24)
    expect(chatItems.get()).toEqual([])
  })

  it('keeps later assistant streaming below interleaved shell output', () => {
    applyGatewayEvent({
      type: 'text.delta',
      session_id: 's1',
      run_id: 'r1',
      text: 'before ',
      sequence: 0
    })
    applyGatewayEvent({
      type: 'tool.call',
      session_id: 's1',
      run_id: 'r1',
      tool_call_id: 'tc1',
      tool_name: 'shell',
      arguments: { command: 'echo hi' },
      iteration: 1
    })
    applyGatewayEvent({
      type: 'tool.result',
      session_id: 's1',
      run_id: 'r1',
      tool_call_id: 'tc1',
      tool_name: 'shell',
      content: 'hi',
      is_error: false,
      iteration: 1
    })
    applyGatewayEvent({
      type: 'text.delta',
      session_id: 's1',
      run_id: 'r1',
      text: 'after',
      sequence: 1
    })
    applyGatewayEvent({
      type: 'done',
      session_id: 's1',
      run_id: 'r1',
      final_text: 'before after',
      exit_reason: 'done'
    })

    const items = chatItems.get()
    expect(items[0]).toMatchObject({
      kind: 'assistant',
      runId: 'r1',
      text: 'before ',
      streaming: false
    })
    expect(items[1]).toMatchObject({
      kind: 'tool-call',
      toolCallId: 'tc1'
    })
    expect(items[2]).toMatchObject({
      kind: 'assistant',
      runId: 'r1',
      text: 'after',
      streaming: false
    })
  })

  it('updates activity todos from todo_write tool calls', () => {
    applyGatewayEvent({
      type: 'tool.call',
      session_id: 's1',
      run_id: 'r1',
      tool_call_id: 'todo1',
      tool_name: 'todo_write',
      arguments: {
        todos: [
          { id: '1', content: 'Implement registry', status: 'in_progress' },
          { id: '2', content: 'Wire gateway', status: 'pending' }
        ]
      },
      iteration: 1
    })

    expect(activityState.get().todos).toEqual([
      { id: '1', content: 'Implement registry', status: 'in_progress' },
      { id: '2', content: 'Wire gateway', status: 'pending' }
    ])

    applyGatewayEvent({
      type: 'tool.call',
      session_id: 's1',
      run_id: 'r1',
      tool_call_id: 'todo2',
      tool_name: 'todo_write',
      arguments: {
        todos: [
          { id: '1', content: 'Implement registry', status: 'completed' },
          { id: '2', content: 'Wire gateway', status: 'completed' }
        ]
      },
      iteration: 1
    })

    expect(activityState.get().todos).toEqual([])
  })

  it('appends the final assistant suffix below interleaved shell output even without a post-tool delta', () => {
    applyGatewayEvent({
      type: 'text.delta',
      session_id: 's1',
      run_id: 'r1',
      text: 'before ',
      sequence: 0
    })
    applyGatewayEvent({
      type: 'tool.call',
      session_id: 's1',
      run_id: 'r1',
      tool_call_id: 'tc1',
      tool_name: 'shell',
      arguments: { command: 'echo hi' },
      iteration: 1
    })
    applyGatewayEvent({
      type: 'tool.result',
      session_id: 's1',
      run_id: 'r1',
      tool_call_id: 'tc1',
      tool_name: 'shell',
      content: 'hi',
      is_error: false,
      iteration: 1
    })
    applyGatewayEvent({
      type: 'done',
      session_id: 's1',
      run_id: 'r1',
      final_text: 'before after',
      exit_reason: 'done'
    })

    const items = chatItems.get()
    expect(items[0]).toMatchObject({
      kind: 'assistant',
      runId: 'r1',
      text: 'before ',
      streaming: false
    })
    expect(items[1]).toMatchObject({
      kind: 'tool-call',
      toolCallId: 'tc1'
    })
    expect(items[2]).toMatchObject({
      kind: 'assistant',
      runId: 'r1',
      text: 'after',
      streaming: false
    })
  })

  it('maps status and usage events into session state', () => {
    applyGatewayEvent({
      type: 'status',
      session_id: 's1',
      run_id: 'r1',
      kind: 'thinking',
      detail: null
    })
    applyGatewayEvent({
      type: 'usage',
      session_id: 's1',
      run_id: 'r1',
      input_tokens: 10,
      output_tokens: 3,
      cache_read_tokens: 2,
      cache_write_tokens: 1
    })

    expect(sessionState.get().status).toBe('thinking')
    expect(sessionState.get().usage).toEqual({
      input: 10,
      output: 3,
      cacheRead: 2,
      cacheWrite: 1
    })
  })

  it('syncs session mode from tool result metadata', () => {
    sessionActions.setMode('plan')
    applyGatewayEvent({
      type: 'tool.result',
      session_id: 's1',
      run_id: 'r1',
      tool_call_id: 'tc1',
      tool_name: 'exit_plan_mode',
      content: 'Plan approved. Returning to agent mode.',
      is_error: false,
      iteration: 1,
      metadata: { new_mode: 'agent' }
    })

    expect(sessionState.get().mode).toBe('agent')
  })
})

describe('chatActions.attachPermissionPreview', () => {
  beforeEach(() => {
    chatActions.resetForTests()
    activityActions.resetForTests()
    sessionActions.resetForTests()
    resetGatewayEventDedupeForTests()
  })

  const SAMPLE_DIFF = [
    '--- src/foo.ts',
    '+++ src/foo.ts',
    '@@ -1,2 +1,3 @@',
    ' const a = 1',
    '+const b = 2',
    ' const c = 3'
  ].join('\n')

  it('hydrates an existing tool-call row with summary + pending status', () => {
    chatActions.pushToolCall({
      id: 'tc1',
      toolName: 'file_edit',
      args: { file_path: 'src/foo.ts' },
      iteration: 1,
      coalesce: true
    })

    chatActions.attachPermissionPreview({
      toolCallId: 'tc1',
      toolName: 'file_edit',
      preview: { diff: SAMPLE_DIFF, path: 'src/foo.ts' }
    })

    const items = chatItems.get()
    expect(items).toHaveLength(1)
    const item = items[0]
    expect(item?.kind).toBe('tool-call')
    if (item?.kind === 'tool-call') {
      expect(item.previewStatus).toBe('pending')
      expect(item.diffOpen).toBe(true)
      expect(item.summary?.path).toBe('src/foo.ts')
      expect(item.summary?.linesAdded).toBe(1)
      expect(item.summary?.linesRemoved).toBe(0)
      expect(item.summary?.diff).toBe(SAMPLE_DIFF)
    }
  })

  it('inserts a placeholder when the tool.call has not arrived yet, and pushToolCall is idempotent', () => {
    chatActions.attachPermissionPreview({
      toolCallId: 'tc2',
      toolName: 'write_file',
      preview: { diff: SAMPLE_DIFF, path: 'src/bar.ts' }
    })
    chatActions.pushToolCall({
      id: 'tc2',
      toolName: 'write_file',
      args: { file_path: 'src/bar.ts', contents: 'x' },
      iteration: 2,
      coalesce: true
    })

    const items = chatItems.get()
    expect(items).toHaveLength(1)
    const item = items[0]
    if (item?.kind === 'tool-call') {
      // The summary survives the late tool.call event.
      expect(item.previewStatus).toBe('pending')
      expect(item.diffOpen).toBe(true)
      expect(item.summary?.diff).toBe(SAMPLE_DIFF)
      // The args from pushToolCall hydrated the placeholder.
      expect(item.args).toMatchObject({ file_path: 'src/bar.ts', contents: 'x' })
      expect(item.iteration).toBe(2)
    }
  })

  it('skips preview synthesis when the preview lacks a path or diff', () => {
    chatActions.pushToolCall({
      id: 'tc3',
      toolName: 'file_edit',
      args: {},
      iteration: 1,
      coalesce: true
    })

    chatActions.attachPermissionPreview({
      toolCallId: 'tc3',
      toolName: 'file_edit',
      preview: { diff: '', path: 'src/baz.ts' }
    })
    chatActions.attachPermissionPreview({
      toolCallId: 'tc3',
      toolName: 'file_edit',
      preview: { diff: SAMPLE_DIFF, path: null }
    })

    const item = chatItems.get()[0]
    if (item?.kind === 'tool-call') {
      expect(item.summary).toBeUndefined()
      expect(item.previewStatus).toBeUndefined()
      expect(item.diffOpen).toBeUndefined()
    }
  })
})

describe('chatActions.resolvePermissionPreview', () => {
  beforeEach(() => {
    chatActions.resetForTests()
  })

  function seedPending(toolCallId: string): void {
    chatActions.pushToolCall({
      id: toolCallId,
      toolName: 'file_edit',
      args: {},
      iteration: 1,
      coalesce: true
    })
    chatActions.attachPermissionPreview({
      toolCallId,
      toolName: 'file_edit',
      preview: {
        diff: '--- a\n+++ a\n@@ -1 +1 @@\n-old\n+new\n',
        path: 'src/q.ts'
      }
    })
  }

  it("'approved' clears the pending label but keeps the diff open", () => {
    seedPending('tcA')
    chatActions.resolvePermissionPreview('tcA', 'approved')
    const item = chatItems.get()[0]
    if (item?.kind === 'tool-call') {
      expect(item.previewStatus).toBeUndefined()
      expect(item.diffOpen).toBe(true)
      expect(item.summary?.path).toBe('src/q.ts')
    }
  })

  it("'rejected' flips the previewStatus", () => {
    seedPending('tcB')
    chatActions.resolvePermissionPreview('tcB', 'rejected')
    const item = chatItems.get()[0]
    if (item?.kind === 'tool-call') {
      expect(item.previewStatus).toBe('rejected')
      expect(item.diffOpen).toBe(true)
    }
  })

  it('preserves the preview diff when the approved tool result has summary metadata without a diff', () => {
    seedPending('tcD')
    chatActions.resolvePermissionPreview('tcD', 'approved')
    chatActions.pushToolResult({
      toolCallId: 'tcD',
      toolName: 'file_edit',
      text: 'ok',
      isError: false,
      metadata: {
        path: 'src/q.ts',
        lines_added: 1,
        lines_removed: 1
      }
    })
    const item = chatItems.get()[0]
    if (item?.kind === 'tool-call') {
      expect(item.previewStatus).toBeUndefined()
      expect(item.diffOpen).toBe(true)
      expect(item.summary?.diff).toContain('-old')
      expect(item.summary?.diff).toContain('+new')
    }
  })

  it("'aborted' removes the preview row entirely", () => {
    seedPending('tcC')
    chatActions.resolvePermissionPreview('tcC', 'aborted')
    expect(chatItems.get()).toHaveLength(0)
  })
})
