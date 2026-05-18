import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { render } from 'ink-testing-library'
import { Text } from 'ink'

import { ChatTranscript } from '../components/ChatTranscript.js'
import { chatItems } from '../store/chatStore.js'
import { focusActions } from '../store/focusStore.js'
import { overlayActions } from '../store/overlayStore.js'

describe('ChatTranscript spacing', () => {
  beforeEach(() => {
    chatItems.set([])
    focusActions.resetForTests()
    overlayActions.resetForTests()
  })

  afterEach(() => {
    chatItems.set([])
    focusActions.resetForTests()
    overlayActions.resetForTests()
  })

  it('keeps a blank line between a turn footer and the next user echo', () => {
    chatItems.set([
      {
        kind: 'note',
        id: 'n1',
        text: '✓ done · 1.20s',
        level: 'info',
        ts: 1
      },
      {
        kind: 'user',
        id: 'u1',
        text: '继续',
        ts: 2
      }
    ])

    const { lastFrame, unmount } = render(<ChatTranscript />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('✓ done · 1.20s')
    expect(frame).toMatch(/✓ done · 1\.20s\s*\n\s*\n.*继续/s)
    unmount()
  })

  it('keeps interrupt and cancelled footer contiguous while separating them from transcript text', () => {
    chatItems.set([
      {
        kind: 'assistant',
        id: 'a1',
        runId: 'r1',
        text: '处理中',
        streaming: false,
        ts: 1
      },
      {
        kind: 'note',
        id: 'n1',
        text: 'interrupt',
        level: 'warn',
        ts: 2
      },
      {
        kind: 'note',
        id: 'n2',
        text: '⏹ cancelled · 1.20s',
        level: 'warn',
        ts: 3
      }
    ])

    const { lastFrame, unmount } = render(<ChatTranscript />)
    const frame = lastFrame() ?? ''
    expect(frame).toMatch(/处理中\s*\n\s*\n.*interrupt\s*\n.*cancelled · 1\.20s/s)
    expect(frame).not.toMatch(/interrupt\s*\n\s*\n.*cancelled · 1\.20s/s)
    unmount()
  })

  it('treats fullscreen leading content as scrollback, not pinned chrome', () => {
    chatItems.set([
      {
        kind: 'user',
        id: 'u1',
        text: 'draw a cyberpunk cat',
        ts: 1
      },
      {
        kind: 'assistant',
        id: 'a1',
        runId: 'r1',
        text: ['line one', 'line two', 'line three', 'line four', 'line five'].join('\n'),
        streaming: true,
        ts: 2
      }
    ])

    const { lastFrame, unmount } = render(
      <ChatTranscript
        viewportRows={4}
        width={80}
        leading={<Text>FULL BANNER</Text>}
        leadingRows={1}
      />
    )
    const frame = lastFrame() ?? ''
    expect(frame).not.toContain('FULL BANNER')
    expect(frame).toContain('line five')
    unmount()
  })

  it('keeps recent stable context visible while the next response streams', () => {
    chatItems.set([
      {
        kind: 'user',
        id: 'u1',
        text: '你好',
        ts: 1
      },
      {
        kind: 'assistant',
        id: 'a1',
        runId: 'r1',
        text: '你好！有什么我可以帮你的吗？',
        streaming: false,
        ts: 2
      },
      {
        kind: 'note',
        id: 'n1',
        text: '✓ done · 1.20s',
        level: 'info',
        ts: 3
      },
      {
        kind: 'user',
        id: 'u2',
        text: '你都会干什么啊',
        ts: 4
      },
      {
        kind: 'assistant',
        id: 'a2',
        runId: 'r2',
        text: '我可以帮你很多事，例如：',
        streaming: true,
        ts: 5
      }
    ])

    const { lastFrame, unmount } = render(<ChatTranscript leading={<Text>FULL BANNER</Text>} />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('你好')
    expect(frame).toContain('你都会干什么啊')
    expect(frame).toContain('我可以帮你很多事')
    unmount()
  })
})
