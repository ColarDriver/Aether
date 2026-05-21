import { describe, expect, it } from 'vitest'
import type { RunSocketFrame } from '../api/types'
import { reduceRunFrame, resolvePromptInBlocks } from './blockReducer'
import { createChatRenderState } from './runState'

describe('reduceRunFrame', () => {
  it('merges assistant deltas into a streaming assistant block', () => {
    const state = createChatRenderState()
    const first = reduceRunFrame(state, frame('assistant.delta', { session_id: 's1', run_id: 'r1', text: 'Hel' }))
    const second = reduceRunFrame(first, frame('assistant.delta', { session_id: 's1', run_id: 'r1', text: 'lo' }))

    expect(second.blocksBySession.s1).toHaveLength(1)
    expect(second.blocksBySession.s1?.[0]).toMatchObject({
      kind: 'assistant_message',
      content: 'Hello',
      isStreaming: true,
    })
  })

  it('merges reasoning deltas into an active thinking block', () => {
    const state = reduceRunFrame(
      createChatRenderState(),
      frame('reasoning.delta', { session_id: 's1', run_id: 'r1', text: 'I should inspect' }),
    )

    expect(state.blocksBySession.s1?.[0]).toMatchObject({
      kind: 'thinking',
      content: 'I should inspect',
      isActive: true,
    })
  })

  it('updates tool call status and creates result and diff blocks', () => {
    const started = reduceRunFrame(
      createChatRenderState(),
      frame('tool.started', {
        session_id: 's1',
        run_id: 'r1',
        tool_call_id: 'call-1',
        tool_name: 'file_edit',
        arguments: { path: 'app.py' },
      }),
    )
    const finished = reduceRunFrame(
      started,
      frame('tool.finished', {
        session_id: 's1',
        run_id: 'r1',
        tool_call_id: 'call-1',
        tool_name: 'file_edit',
        content: 'updated',
        metadata: { diff: '-old\n+new' },
      }),
    )

    expect(finished.blocksBySession.s1?.map((block) => block.kind)).toEqual([
      'tool_call',
      'tool_result',
      'diff',
    ])
    expect(finished.blocksBySession.s1?.[0]).toMatchObject({ status: 'finished' })
  })

  it('creates prompt blocks and exposes pending prompt state', () => {
    const withPermission = reduceRunFrame(
      createChatRenderState(),
      frame('permission.requested', {
        session_id: 's1',
        run_id: 'r1',
        prompt_id: 'permission-1',
        request: {
          tool_name: 'write_file',
          arguments: { path: 'app.py' },
          preview: { diff: '-old\n+new' },
        },
      }),
    )
    const withApproval = reduceRunFrame(
      withPermission,
      frame('approval.requested', {
        session_id: 's1',
        run_id: 'r1',
        prompt_id: 'approval-1',
        kind: 'plan',
        plan_text: '# Plan',
      }),
    )

    expect(withApproval.pendingPermissionBlock).toMatchObject({ promptId: 'permission-1' })
    expect(withApproval.pendingApprovalBlock).toMatchObject({ promptId: 'approval-1' })
    expect(withApproval.blocksBySession.s1?.map((block) => block.kind)).toEqual([
      'permission_request',
      'approval_request',
    ])
  })

  it('clears streaming flags on finish and creates visible errors on failure', () => {
    const streaming = reduceRunFrame(
      createChatRenderState(),
      frame('assistant.delta', { session_id: 's1', run_id: 'r1', text: 'partial' }),
    )
    const failed = reduceRunFrame(
      streaming,
      frame('run.failed', { session_id: 's1', run_id: 'r1', message: 'boom' }),
    )

    expect(failed.blocksBySession.s1?.[0]).toMatchObject({ kind: 'assistant_message', isStreaming: false, isError: true })
    expect(failed.blocksBySession.s1?.[1]).toMatchObject({ kind: 'error', message: 'boom' })
  })
})

describe('resolvePromptInBlocks', () => {
  it('optimistically marks prompt blocks as resolved', () => {
    const state = reduceRunFrame(
      createChatRenderState(),
      frame('approval.requested', {
        session_id: 's1',
        run_id: 'r1',
        prompt_id: 'approval-1',
        kind: 'plan',
      }),
    )

    const resolved = resolvePromptInBlocks(state.blocksBySession, 'approval-1', { confirmed: false })

    expect(resolved.s1?.[0]).toMatchObject({ kind: 'approval_request', state: 'rejected' })
  })
})

function frame(type: string, payload: Record<string, unknown>): RunSocketFrame {
  return { type, payload, transport_sequence: 1 }
}
