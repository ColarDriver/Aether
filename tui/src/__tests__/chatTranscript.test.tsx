import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { render } from 'ink-testing-library'

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
})
