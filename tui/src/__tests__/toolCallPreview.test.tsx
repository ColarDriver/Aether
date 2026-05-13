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
    durationMs: 230,
    ts: 0,
    ...overrides
  }
}

describe('ToolCallPanel — inline result preview (collapsed state)', () => {
  it('shows the head of bash output below the collapsed header', () => {
    const item = makeItem({
      result: { text: 'total 12\nfile-a\nfile-b\nfile-c', isError: false }
    })
    const { lastFrame, unmount } = render(
      <ToolCallPanel item={item} expanded={false} focused={false} />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('▸ shell')
    expect(frame).toContain('⎿ total 12')
    expect(frame).toContain('file-a')
    unmount()
  })

  it('truncates and reports remaining lines when output exceeds the preview limit', () => {
    const longOutput = Array.from({ length: 30 }, (_, idx) => `line ${idx}`).join('\n')
    const item = makeItem({ result: { text: longOutput, isError: false } })
    const { lastFrame, unmount } = render(
      <ToolCallPanel item={item} expanded={false} focused={false} />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('line 0')
    expect(frame).toMatch(/\d+ more lines · Tab\+Enter to expand/)
    unmount()
  })

  it('renders red-coloured preview for errored results', () => {
    const item = makeItem({
      result: { text: 'permission denied', isError: true }
    })
    const { lastFrame, unmount } = render(
      <ToolCallPanel item={item} expanded={false} focused={false} />
    )
    expect(lastFrame()).toContain('permission denied')
    unmount()
  })

  it('omits the preview block when the result has not arrived yet', () => {
    const item = makeItem({ durationMs: null })
    const { lastFrame, unmount } = render(
      <ToolCallPanel item={item} expanded={false} focused={false} />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('running…')
    expect(frame).not.toContain('⎿')
    unmount()
  })
})
