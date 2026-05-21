import { describe, expect, it } from 'vitest'
import type { ChatBlock } from './blocks'
import { buildChatRenderModel } from './renderModel'

const base = {
  sessionId: 's1',
  runId: 'r1',
  timestamp: 1,
  source: 'live' as const,
}

describe('buildChatRenderModel', () => {
  it('groups adjacent root tool calls and attaches matching results', () => {
    const blocks: ChatBlock[] = [
      { ...base, id: 'user', kind: 'user_message', content: 'go' },
      { ...base, id: 'tool-a', kind: 'tool_call', toolCallId: 'a', toolName: 'read_file', arguments: {}, status: 'running' },
      { ...base, id: 'tool-b', kind: 'tool_call', toolCallId: 'b', toolName: 'grep', arguments: {}, status: 'running' },
      { ...base, id: 'result-a', kind: 'tool_result', toolCallId: 'a', content: 'ok', isError: false, metadata: {} },
      { ...base, id: 'assistant', kind: 'assistant_message', content: 'done' },
    ]

    const model = buildChatRenderModel(blocks)

    expect(model.items.map((item) => item.kind)).toEqual(['block', 'tool_group', 'block'])
    expect(model.items[1]).toMatchObject({ kind: 'tool_group', toolCalls: [{ toolCallId: 'a' }, { toolCallId: 'b' }] })
    expect(model.toolResultsByCallId.get('a')).toMatchObject({ content: 'ok' })
  })

  it('keeps unmatched tool results and ask_user_question visible as normal blocks', () => {
    const blocks: ChatBlock[] = [
      { ...base, id: 'ask', kind: 'ask_user_question', toolCallId: 'ask-1', questions: [{ question: 'Continue?' }], state: 'pending' },
      { ...base, id: 'orphan', kind: 'tool_result', toolCallId: 'missing', content: 'done', isError: false, metadata: {} },
    ]

    const model = buildChatRenderModel(blocks)

    expect(model.items).toHaveLength(2)
    expect(model.items[0]).toMatchObject({ kind: 'block', block: { kind: 'ask_user_question' } })
    expect(model.items[1]).toMatchObject({ kind: 'block', block: { kind: 'tool_result' } })
  })

  it('does not duplicate tool results already represented by ask_user_question blocks', () => {
    const blocks: ChatBlock[] = [
      { ...base, id: 'ask', kind: 'ask_user_question', toolCallId: 'ask-1', questions: [{ question: 'Continue?' }], state: 'answered', answers: { Continue: 'Yes' } },
      { ...base, id: 'result', kind: 'tool_result', toolCallId: 'ask-1', content: 'answered', isError: false, metadata: {} },
    ]

    const model = buildChatRenderModel(blocks)

    expect(model.items).toHaveLength(1)
    expect(model.items[0]).toMatchObject({ kind: 'block', block: { kind: 'ask_user_question' } })
  })
})
