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

  it('estimates active-run tokens from streamed text before provider usage arrives', () => {
    const state = createChatRenderState()
    const streamed = reduceRunFrame(state, frame('assistant.delta', { session_id: 's1', run_id: 'r1', text: 'Streaming token count should grow.' }))

    expect(streamed.tokenUsageByRun.r1?.output_tokens).toBeGreaterThan(0)
    expect(streamed.statusByRun.r1?.tokens?.output_tokens).toBe(streamed.tokenUsageByRun.r1?.output_tokens)
  })

  it('renders run.result final text when no assistant delta was streamed', () => {
    const accepted = reduceRunFrame(
      createChatRenderState(),
      frame('run.accepted', { session_id: 's1', run_id: 'r1' }),
    )

    const state = reduceRunFrame(
      accepted,
      frame('run.result', {
        session_id: 's1',
        run_id: 'r1',
        final_text: 'Final non-streamed answer',
        usage: { input_tokens: 4, output_tokens: 6, total_tokens: 10 },
        metadata: { hosted_web_search: { provider: 'codex', source_count: 1 } },
      }),
    )

    expect(state.activeRunId).toBeNull()
    expect(state.blocksBySession.s1?.[0]).toMatchObject({
      kind: 'assistant_message',
      content: 'Final non-streamed answer',
      isStreaming: false,
      metadata: { hosted_web_search: { provider: 'codex', source_count: 1 } },
    })
    expect(state.tokenUsageByRun.r1).toMatchObject({ total_tokens: 10 })
  })

  it('uses run.result to complete a partial streamed assistant block', () => {
    const partial = reduceRunFrame(
      createChatRenderState(),
      frame('assistant.delta', { session_id: 's1', run_id: 'r1', text: 'Hel' }),
    )

    const state = reduceRunFrame(
      partial,
      frame('run.result', {
        session_id: 's1',
        run_id: 'r1',
        final_text: 'Hello',
        metadata: { model: 'gpt-5.4' },
      }),
    )

    expect(state.blocksBySession.s1).toHaveLength(1)
    expect(state.blocksBySession.s1?.[0]).toMatchObject({
      kind: 'assistant_message',
      content: 'Hello',
      isStreaming: false,
      metadata: { model: 'gpt-5.4' },
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

  it('marks ask_user_question blocks answered from tool result metadata', () => {
    const started = reduceRunFrame(
      createChatRenderState(),
      frame('tool.started', {
        session_id: 's1',
        run_id: 'r1',
        tool_call_id: 'ask-1',
        tool_name: 'ask_user_question',
        arguments: { questions: [{ id: 'mode', prompt: 'Which mode?' }] },
      }),
    )
    const finished = reduceRunFrame(
      started,
      frame('tool.finished', {
        session_id: 's1',
        run_id: 'r1',
        tool_call_id: 'ask-1',
        tool_name: 'ask_user_question',
        content: 'User has answered your questions.',
        metadata: { answer_pairs: [{ label: 'Mode', value: 'Careful' }] },
      }),
    )

    expect(finished.blocksBySession.s1?.[0]).toMatchObject({
      kind: 'ask_user_question',
      state: 'answered',
      answers: { Mode: 'Careful' },
    })
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
          allow_session: true,
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

    expect(withApproval.pendingPermissionBlock).toMatchObject({ promptId: 'permission-1', allowSession: true })
    expect(withApproval.pendingApprovalBlock).toMatchObject({ promptId: 'approval-1' })
    expect(withApproval.blocksBySession.s1?.map((block) => block.kind)).toEqual([
      'permission_request',
      'approval_request',
    ])
  })

  it('marks only the resolved prompt block when prompt resolution arrives from the socket', () => {
    const withPermission = reduceRunFrame(
      createChatRenderState(),
      frame('permission.requested', {
        session_id: 's1',
        run_id: 'r1',
        prompt_id: 'permission-1',
        request: { tool_name: 'write_file' },
      }),
    )
    const withApproval = reduceRunFrame(
      withPermission,
      frame('approval.requested', {
        session_id: 's1',
        run_id: 'r1',
        prompt_id: 'approval-1',
        kind: 'plan',
      }),
    )

    const resolved = reduceRunFrame(
      withApproval,
      frame('prompt.resolved', { prompt_id: 'permission-1', result: { decision: { type: 'deny' } } }),
    )

    expect(resolved.pendingPermissionBlock).toBeNull()
    expect(resolved.pendingApprovalBlock).toMatchObject({ promptId: 'approval-1' })
    expect(resolved.blocksBySession.s1?.[0]).toMatchObject({
      kind: 'permission_request',
      promptId: 'permission-1',
      state: 'denied',
    })
    expect(resolved.blocksBySession.s1?.[1]).toMatchObject({
      kind: 'approval_request',
      promptId: 'approval-1',
      state: 'pending',
    })
  })

  it('marks stale prompt frames as terminal without applying an optimistic decision', () => {
    const withPermission = reduceRunFrame(
      createChatRenderState(),
      frame('permission.requested', {
        session_id: 's1',
        run_id: 'r1',
        prompt_id: 'permission-stale',
        request: { tool_name: 'write_file' },
      }),
    )

    const stale = reduceRunFrame(
      withPermission,
      frame('prompt.stale', {
        prompt_id: 'permission-stale',
        status: 'stale',
        reason: 'Backend restarted before this prompt was resolved.',
        result: { decision: { type: 'allow_once' } },
      }),
    )

    expect(stale.pendingPermissionBlock).toBeNull()
    expect(stale.blocksBySession.s1?.[0]).toMatchObject({
      kind: 'permission_request',
      promptId: 'permission-stale',
      state: 'stale',
      statusMessage: 'Backend restarted before this prompt was resolved.',
    })
  })

  it('accepts permission frames whose session id only exists inside the request payload', () => {
    const state = reduceRunFrame(
      createChatRenderState(),
      frame('permission.requested', {
        run_id: 'r1',
        prompt_id: 'permission-1',
        request: {
          session_id: 's1',
          tool_name: 'write_file',
          tool_call_id: 'call-write',
          arguments: { path: '/tmp/calc.py' },
        },
      }),
    )

    expect(state.activeRunId).toBe('r1')
    expect(state.pendingPermissionBlock).toMatchObject({
      promptId: 'permission-1',
      sessionId: 's1',
      toolName: 'write_file',
    })
    expect(state.blocksBySession.s1?.[0]).toMatchObject({ kind: 'permission_request' })
  })

  it('normalizes token usage frames across provider payload styles', () => {
    const state = reduceRunFrame(
      createChatRenderState(),
      frame('token.usage', {
        session_id: 's1',
        run_id: 'r1',
        usage: {
          prompt_tokens: 1000,
          completionTokens: 240,
          prompt_tokens_details: { cached_tokens: 120 },
          completion_tokens_details: { reasoning_tokens: 40 },
        },
      }),
    )

    expect(state.tokenUsageByRun.r1).toEqual({
      input_tokens: 1000,
      output_tokens: 240,
      cache_read_tokens: 120,
      cache_write_tokens: undefined,
      reasoning_tokens: 40,
      total_tokens: 1400,
    })
    expect(state.blocksBySession.s1?.[0]).toMatchObject({
      kind: 'streaming_status',
      tokens: { total_tokens: 1400 },
    })
  })

  it('shows lifecycle status for accepted and cancelled runs', () => {
    const accepted = reduceRunFrame(
      createChatRenderState(),
      frame('run.accepted', { session_id: 's1', run_id: 'r1' }),
    )

    expect(accepted.activeRunId).toBe('r1')
    expect(accepted.statusByRun.r1).toMatchObject({ state: 'starting', detail: 'run accepted' })
    expect(accepted.blocksBySession.s1?.[0]).toMatchObject({ kind: 'streaming_status', state: 'starting' })

    const cancelling = reduceRunFrame(
      accepted,
      frame('run.cancel.accepted', { session_id: 's1', run_id: 'r1', cancelled: true }),
    )

    expect(cancelling.statusByRun.r1).toMatchObject({ state: 'cancelling', detail: 'interrupt requested' })
    expect(cancelling.blocksBySession.s1?.[0]).toMatchObject({ kind: 'streaming_status', state: 'cancelling' })
  })

  it('renders service error frames with session details as visible errors', () => {
    const state = reduceRunFrame(
      createChatRenderState(),
      frame('error', {
        code: 'RUN_ALREADY_ACTIVE',
        message: 'run already active',
        details: { session_id: 's1', run_id: 'r1' },
      }),
    )

    expect(state.activeRunId).toBeNull()
    expect(state.blocksBySession.s1?.[0]).toMatchObject({
      kind: 'error',
      message: 'run already active',
      code: 'RUN_ALREADY_ACTIVE',
      details: [
        { label: 'Session Id', value: 's1' },
        { label: 'Run Id', value: 'r1' },
      ],
      suggestions: ['Wait for the active run to finish, or cancel it before starting a new run in this session.'],
    })
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

  it('renders error run results even when a run.failed frame was missed', () => {
    const accepted = reduceRunFrame(
      createChatRenderState(),
      frame('run.accepted', { session_id: 's1', run_id: 'r-provider' }),
    )
    const failed = reduceRunFrame(
      accepted,
      frame('run.result', {
        session_id: 's1',
        run_id: 'r-provider',
        final_text: '',
        exit_reason: 'error',
        metadata: {
          error: {
            type: 'ProviderInvocationError',
            message: 'provider HTTP 404: 404 page not found',
            status_code: 404,
            body_summary: '404 page not found',
            metadata: { url: 'https://provider.test/chat/completions' },
          },
        },
      }),
    )

    expect(failed.activeRunId).toBeNull()
    expect(failed.blocksBySession.s1?.at(-1)).toMatchObject({
      kind: 'error',
      code: 'ProviderInvocationError',
      message: 'provider HTTP 404: 404 page not found',
      details: [
        { label: 'HTTP status', value: '404' },
        { label: 'Endpoint', value: 'https://provider.test/chat/completions' },
        { label: 'Response body', value: '404 page not found' },
      ],
      suggestions: expect.arrayContaining([
        expect.stringContaining('provider base URL'),
        expect.stringContaining('selected model'),
      ]),
    })
  })

  it('attaches run.result metadata to the last tool result when no assistant text is available', () => {
    const started = reduceRunFrame(
      createChatRenderState(),
      frame('tool.started', {
        session_id: 's1',
        run_id: 'r-tool-only',
        tool_call_id: 'call-1',
        tool_name: 'file_edit',
        arguments: { path: 'app.py' },
      }),
    )
    const toolFinished = reduceRunFrame(
      started,
      frame('tool.finished', {
        session_id: 's1',
        run_id: 'r-tool-only',
        tool_call_id: 'call-1',
        tool_name: 'file_edit',
        content: 'edited',
      }),
    )

    const finished = reduceRunFrame(
      toolFinished,
      frame('run.result', {
        session_id: 's1',
        run_id: 'r-tool-only',
        final_text: '',
        metadata: { workspace_checkpoint: { checkpoint_id: 'cp-1' } },
      }),
    )

    expect(finished.blocksBySession.s1?.find((block) => block.kind === 'tool_result')).toMatchObject({
      kind: 'tool_result',
      metadata: { workspace_checkpoint: { checkpoint_id: 'cp-1' } },
    })
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
