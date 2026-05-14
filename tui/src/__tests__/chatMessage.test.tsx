import { describe, expect, it } from 'vitest'
import { render } from 'ink-testing-library'

import { ChatMessage } from '../components/ChatMessage.js'

describe('ChatMessage', () => {
  it('does not render a streaming banner for assistant output', () => {
    const { lastFrame, unmount } = render(
      <ChatMessage
        expanded={false}
        focused={false}
        item={{
          kind: 'assistant',
          id: 'a1',
          runId: 'r1',
          text: 'hello',
          streaming: true,
          ts: Date.now()
        }}
      />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('hello')
    expect(frame).not.toContain('streaming')
    unmount()
  })

  it('hides assistant rows that only contain stripped tool markup', () => {
    const { lastFrame, unmount } = render(
      <ChatMessage
        expanded={false}
        focused={false}
        item={{
          kind: 'assistant',
          id: 'a2',
          runId: 'r2',
          text: '<tool_call>{\"name\":\"read_file\"}</tool_call>',
          streaming: true,
          ts: Date.now()
        }}
      />
    )
    expect(lastFrame()).toBe('')
    unmount()
  })

  it('treats turn footer notes as already-formed lines', () => {
    const { lastFrame, unmount } = render(
      <ChatMessage
        expanded={false}
        focused={false}
        item={{
          kind: 'note',
          id: 'n1',
          text: '✓ done · 1.20s',
          level: 'info',
          ts: Date.now()
        }}
      />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('✓ done · 1.20s')
    expect(frame).not.toContain('ℹ')
    unmount()
  })
})
