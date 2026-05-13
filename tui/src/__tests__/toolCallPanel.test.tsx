import { describe, expect, it } from 'vitest'
import { render } from 'ink-testing-library'

import { ToolCallPanel } from '../components/ToolCallPanel.js'
import type { ChatItem } from '../store/chatStore.js'

function makeItem(
  overrides: Partial<Extract<ChatItem, { kind: 'tool-call' }>> = {}
): Extract<ChatItem, { kind: 'tool-call' }> {
  return {
    kind: 'tool-call',
    id: 'tc_render',
    toolCallId: 'tc_1',
    toolName: 'shell',
    args: { command: 'ls -la' },
    argsPreview: '{"command":"ls -la"}',
    iteration: 1,
    coalesce: false,
    durationMs: null,
    ts: 0,
    ...overrides
  }
}

describe('ToolCallPanel', () => {
  it('renders a one-line collapsed row by default', () => {
    const { lastFrame, unmount } = render(
      <ToolCallPanel item={makeItem()} expanded={false} focused={false} />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('▸ shell')
    expect(frame).toContain('$ ls -la')
    expect(frame).not.toContain('arguments:')
    unmount()
  })

  it('shows the duration once the result attaches', () => {
    const { lastFrame, unmount } = render(
      <ToolCallPanel
        item={makeItem({
          durationMs: 420,
          result: { text: 'total 0\n', isError: false }
        })}
        expanded={false}
        focused={false}
      />
    )
    expect(lastFrame()).toContain('420ms')
    unmount()
  })

  it('expanded state shows arguments + result blocks', () => {
    const { lastFrame, unmount } = render(
      <ToolCallPanel
        item={makeItem({ result: { text: 'ok\n', isError: false } })}
        expanded
        focused
      />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('arguments:')
    expect(frame).toContain('"command"')
    expect(frame).toContain('result:')
    expect(frame).toContain('ok')
    unmount()
  })

  it('errored result colours the row red and tags it', () => {
    const { lastFrame, unmount } = render(
      <ToolCallPanel
        item={makeItem({ result: { text: 'boom', isError: true }, durationMs: 12 })}
        expanded={false}
        focused={false}
      />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('error')
    unmount()
  })
})
