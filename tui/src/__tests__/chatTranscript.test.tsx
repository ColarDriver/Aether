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

  it('renders pending coalesced edit previews above the permission modal', () => {
    chatItems.set([
      {
        kind: 'tool-call',
        id: 'tc1',
        toolCallId: 'tc1',
        toolName: 'file_edit',
        args: {},
        argsPreview: '',
        iteration: 1,
        coalesce: true,
        durationMs: null,
        ts: 1,
        previewStatus: 'pending',
        diffOpen: true,
        summary: {
          path: 'src/foo.ts',
          linesAdded: 1,
          linesRemoved: 1,
          hunks: 1,
          diff: '--- a/src/foo.ts\n+++ b/src/foo.ts\n@@ -1 +1 @@\n-old\n+new\n'
        }
      }
    ])

    const { lastFrame, unmount } = render(<ChatTranscript />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Update')
    expect(frame).toContain('src/foo.ts')
    expect(frame).toContain('pending approval')
    expect(frame).toContain('old')
    expect(frame).toContain('new')
    unmount()
  })

  it('keeps an approved permission diff visible in the transcript', () => {
    chatItems.set([
      {
        kind: 'tool-call',
        id: 'tc1',
        toolCallId: 'tc1',
        toolName: 'file_edit',
        args: {},
        argsPreview: '',
        iteration: 1,
        coalesce: true,
        durationMs: null,
        ts: 1,
        diffOpen: true,
        summary: {
          path: 'src/foo.ts',
          linesAdded: 1,
          linesRemoved: 1,
          hunks: 1,
          diff: '--- a/src/foo.ts\n+++ b/src/foo.ts\n@@ -1 +1 @@\n-old\n+new\n'
        }
      }
    ])

    const { lastFrame, unmount } = render(<ChatTranscript />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Update')
    expect(frame).toContain('src/foo.ts')
    expect(frame).not.toContain('pending approval')
    expect(frame).toContain('old')
    expect(frame).toContain('new')
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


  it('does not duplicate the latest user echo while the assistant streams', () => {
    chatItems.set([
      {
        kind: 'user',
        id: 'u1',
        text: '有没有其他的',
        ts: 1
      },
      {
        kind: 'assistant',
        id: 'a1',
        runId: 'r1',
        text: '有，甚至可以说很多。',
        streaming: true,
        ts: 2
      }
    ])

    const { lastFrame, unmount } = render(<ChatTranscript />)
    const frame = lastFrame() ?? ''
    expect(frame.match(/有没有其他的/g)?.length).toBe(1)
    expect(frame).toContain('有，甚至可以说很多。')
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

  it('does not pin leading content between static transcript and the composer', () => {
    chatItems.set([
      {
        kind: 'assistant',
        id: 'a1',
        runId: 'r1',
        text: 'older answer',
        streaming: false,
        ts: 1
      },
      {
        kind: 'note',
        id: 'n1',
        text: '✓ done · 1.20s',
        level: 'info',
        ts: 2
      }
    ])

    const { lastFrame, unmount } = render(
      <ChatTranscript
        leading={<Text>FULL BANNER</Text>}
        liveContextRows={1}
      />
    )
    const frame = lastFrame() ?? ''
    expect(frame.match(/FULL BANNER/g)?.length).toBe(1)
    expect(frame.indexOf('FULL BANNER')).toBeLessThan(frame.indexOf('older answer'))
    unmount()
  })

  it('keeps the final assistant line visible after the done footer arrives', () => {
    chatItems.set([
      {
        kind: 'note',
        id: 'old',
        text: Array.from({ length: 20 }, (_, index) => `old-${index}`).join('\n'),
        level: 'info',
        ts: 1
      },
      {
        kind: 'assistant',
        id: 'a1',
        runId: 'r1',
        text: [
          '我可以帮你处理代码、文档和项目问题。',
          '',
          '如果你想，我还可以马上先做一件小事：',
          '比如先帮你定位这个界面渲染问题。'
        ].join('\n'),
        streaming: false,
        ts: 2
      },
      {
        kind: 'note',
        id: 'done',
        text: '✓ done · 11.4s',
        level: 'info',
        ts: 3
      }
    ])

    const { lastFrame, unmount } = render(
      <ChatTranscript
        leading={<Text>FULL BANNER</Text>}
        liveContextRows={1}
      />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('比如先帮你定位这个界面渲染问题。')
    expect(frame).toContain('✓ done · 11.4s')
    unmount()
  })
})
