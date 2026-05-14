import { beforeEach, describe, expect, it } from 'vitest'

import type { TranscriptMessage } from '../gatewayTypes.js'
import {
  chatActions,
  chatItems,
  rebuildChatItemsFromTranscript,
  type ChatItem
} from '../store/chatStore.js'

function only<K extends ChatItem['kind']>(
  items: ChatItem[],
  kind: K
): Array<Extract<ChatItem, { kind: K }>> {
  return items.filter((item): item is Extract<ChatItem, { kind: K }> => item.kind === kind)
}

describe('rebuildChatItemsFromTranscript', () => {
  beforeEach(() => {
    chatActions.resetForTests()
  })

  it('drops system messages and keeps user + assistant text', () => {
    const items = rebuildChatItemsFromTranscript([
      { role: 'system', text: '<environment>...</environment>' },
      { role: 'user', text: 'hello' },
      { role: 'assistant', text: 'hi there' }
    ])
    expect(items.map((i) => i.kind)).toEqual(['user', 'assistant'])
    const [user, assistant] = items as [
      Extract<ChatItem, { kind: 'user' }>,
      Extract<ChatItem, { kind: 'assistant' }>
    ]
    expect(user.text).toBe('hello')
    expect(assistant.text).toBe('hi there')
    expect(assistant.streaming).toBe(false)
  })

  it('reconstructs explore-burst tool calls into a tool-group with a leading Explored entry', () => {
    const messages: TranscriptMessage[] = [
      { role: 'user', text: 'find pyproject' },
      {
        role: 'assistant',
        text: '',
        tool_calls: [
          {
            id: 't1',
            name: 'list_dir',
            arguments: { path: '/workspace/Aether' }
          },
          {
            id: 't2',
            name: 'read_file',
            arguments: { path: '/workspace/Aether/pyproject.toml' }
          }
        ]
      },
      {
        role: 'tool',
        tool_call_id: 't1',
        text: 'aether/\ntui/\npyproject.toml',
        is_error: false
      },
      {
        role: 'tool',
        tool_call_id: 't2',
        text: '# pyproject.toml contents…',
        is_error: false
      }
    ]
    const items = rebuildChatItemsFromTranscript(messages)
    // user message + 2 tool-call entries (coalesced, hidden) + 1 tool-group
    const toolCalls = only(items, 'tool-call')
    const toolGroups = only(items, 'tool-group')
    expect(toolCalls.map((tc) => tc.toolName)).toEqual(['list_dir', 'read_file'])
    expect(toolCalls.every((tc) => tc.coalesce)).toBe(true)
    expect(toolCalls.every((tc) => tc.result !== undefined)).toBe(true)
    expect(toolGroups).toHaveLength(1)
    expect(toolGroups[0]!.group.entries.map((e) => e.toolName)).toEqual([
      'list_dir',
      'read_file'
    ])
    expect(toolGroups[0]!.group.hasError).toBe(false)
  })

  it('flushes the explore group when a new user message arrives', () => {
    const items = rebuildChatItemsFromTranscript([
      { role: 'user', text: 'q1' },
      {
        role: 'assistant',
        text: '',
        tool_calls: [
          { id: 't1', name: 'list_dir', arguments: { path: '/' } }
        ]
      },
      { role: 'tool', tool_call_id: 't1', text: 'ok' },
      { role: 'user', text: 'q2' }
    ])
    // Expected order: user(q1), tool-call(list_dir, coalesce), tool-group, user(q2)
    expect(items.map((i) => i.kind)).toEqual([
      'user',
      'tool-call',
      'tool-group',
      'user'
    ])
  })

  it('flushes the explore group when the assistant emits visible text', () => {
    const items = rebuildChatItemsFromTranscript([
      { role: 'user', text: 'check' },
      {
        role: 'assistant',
        text: '',
        tool_calls: [{ id: 't1', name: 'list_dir', arguments: {} }]
      },
      { role: 'tool', tool_call_id: 't1', text: 'ok' },
      { role: 'assistant', text: 'all done' }
    ])
    expect(items.map((i) => i.kind)).toEqual([
      'user',
      'tool-call',
      'tool-group',
      'assistant'
    ])
  })

  it('attaches edit-summary metadata so EditSummary renders for file_edit results', () => {
    const items = rebuildChatItemsFromTranscript([
      { role: 'user', text: 'fix the bug' },
      {
        role: 'assistant',
        text: '',
        tool_calls: [
          {
            id: 't_edit',
            name: 'file_edit',
            arguments: { path: '/x.py', old_string: 'a', new_string: 'b' }
          }
        ]
      },
      {
        role: 'tool',
        tool_call_id: 't_edit',
        text: 'edited /x.py',
        is_error: false,
        metadata: {
          path: '/x.py',
          lines_added: 3,
          lines_removed: 1,
          hunks: 1
        }
      }
    ])
    const toolCalls = only(items, 'tool-call')
    expect(toolCalls).toHaveLength(1)
    const editCall = toolCalls[0]!
    expect(editCall.coalesce).toBe(true)
    expect(editCall.summary).toEqual({
      path: '/x.py',
      linesAdded: 3,
      linesRemoved: 1,
      hunks: 1
    })
    // Successful edits drop out of the Explored tree — flushPending should
    // have produced no tool-group (only entry would've been the edit).
    expect(only(items, 'tool-group')).toHaveLength(0)
  })

  it('keeps failed write/edit calls visible in the Explored tree with no summary', () => {
    const items = rebuildChatItemsFromTranscript([
      {
        role: 'assistant',
        text: '',
        tool_calls: [
          {
            id: 't_bad',
            name: 'file_edit',
            arguments: { path: '/x', old_string: 'a', new_string: 'b' }
          }
        ]
      },
      {
        role: 'tool',
        tool_call_id: 't_bad',
        text: 'old_string not found',
        is_error: true
      }
    ])
    const editCall = only(items, 'tool-call')[0]!
    expect(editCall.summary).toBeUndefined()
    expect(editCall.result?.isError).toBe(true)
    // Failed entry stays in the explore burst so the user sees (failed).
    const groups = only(items, 'tool-group')
    expect(groups).toHaveLength(1)
    expect(groups[0]!.group.hasError).toBe(true)
    expect(groups[0]!.group.entries[0]!.toolName).toBe('file_edit')
    expect(groups[0]!.group.entries[0]!.isError).toBe(true)
  })

  it('treats shell as a standalone tool-call (no group, no summary)', () => {
    const items = rebuildChatItemsFromTranscript([
      {
        role: 'assistant',
        text: '',
        tool_calls: [
          {
            id: 't_sh',
            name: 'shell',
            arguments: { command: 'ls -la' }
          }
        ]
      },
      {
        role: 'tool',
        tool_call_id: 't_sh',
        text: '[exit 0 · 12ms]\ntotal 0\n',
        is_error: false,
        metadata: { exit_code: 0, duration_ms: 12 }
      }
    ])
    const shellCall = only(items, 'tool-call')[0]!
    expect(shellCall.coalesce).toBe(false)
    expect(shellCall.summary).toBeUndefined()
    expect(shellCall.result?.metadata?.exit_code).toBe(0)
    expect(only(items, 'tool-group')).toHaveLength(0)
  })

  it('emits an orphan tool-result when no matching tool-call was seen', () => {
    const items = rebuildChatItemsFromTranscript([
      // No assistant turn — just a stray tool message (rare race).
      { role: 'tool', tool_call_id: 'lost', text: 'orphan', is_error: false }
    ])
    expect(items.map((i) => i.kind)).toEqual(['tool-result'])
    const orphan = items[0] as Extract<ChatItem, { kind: 'tool-result' }>
    expect(orphan.toolCallId).toBe('lost')
    expect(orphan.text).toBe('orphan')
  })

  it('replaceTranscript writes the rebuilt items into chatItems', () => {
    chatActions.replaceTranscript([
      { role: 'user', text: 'hi' },
      { role: 'assistant', text: 'hello' }
    ])
    const kinds = chatItems.get().map((i) => i.kind)
    expect(kinds).toEqual(['user', 'assistant'])
  })
})
