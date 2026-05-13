import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render } from 'ink-testing-library'

import type { SessionInfo, TranscriptMessage } from '../gatewayTypes.js'
import { SessionPicker, type SessionPickerPayload } from '../overlays/SessionPicker.js'
import { OVERLAY_PRIORITY, overlayActions, type OverlayState } from '../store/overlayStore.js'

const SETTLE_MS = 80
const flush = (ms: number = SETTLE_MS): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms))

function makeSession(id: string, ageMinutes: number): SessionInfo {
  const now = Math.floor(Date.now() / 1000)
  return {
    session_id: id,
    created_at: now - ageMinutes * 60,
    updated_at: now - ageMinutes * 60,
    provider: 'openai',
    model: 'gpt-4o',
    summary: `summary for ${id}`
  }
}

let nextCreatedAt = 1
function makeOverlay(payload: SessionPickerPayload): OverlayState<SessionPickerPayload> {
  const overlay: OverlayState<SessionPickerPayload> = {
    kind: 'session-picker',
    id: `picker_${nextCreatedAt}`,
    payload,
    createdAt: nextCreatedAt++,
    priority: OVERLAY_PRIORITY['session-picker'],
    onDismiss: vi.fn()
  }
  overlayActions.push(overlay)
  return overlay
}

describe('SessionPicker', () => {
  beforeEach(() => {
    overlayActions.resetForTests()
    nextCreatedAt = 1
  })
  afterEach(() => {
    overlayActions.resetForTests()
  })

  it('renders an empty-state when no sessions exist', () => {
    const overlay = makeOverlay({
      sessions: [],
      resolveResume: vi.fn(),
      onResume: vi.fn()
    })
    const { lastFrame, unmount } = render(<SessionPicker overlay={overlay} />)
    expect(lastFrame()).toContain('No saved sessions found')
    unmount()
  })

  it('shows the first sessions and the entries-count header', () => {
    const sessions: SessionInfo[] = []
    for (let i = 0; i < 5; i++) {
      sessions.push(makeSession(`ses_${i.toString().padStart(8, '0')}`, i * 5))
    }
    const overlay = makeOverlay({
      sessions,
      resolveResume: vi.fn(),
      onResume: vi.fn()
    })
    const { lastFrame, unmount } = render(<SessionPicker overlay={overlay} />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Resume session')
    expect(frame).toContain('5 entries')
    expect(frame).toContain('ses_0000')
    unmount()
  })

  it('Enter resolves the selected session, calls onResume, and dismisses', async () => {
    const sessions = [makeSession('ses_alpha000', 1), makeSession('ses_beta0000', 2)]
    const messages: TranscriptMessage[] = [
      { role: 'user', text: 'hi' },
      { role: 'assistant', text: 'hello' }
    ]
    const resolveResume = vi.fn(async (sessionId: string) => ({
      info: sessions.find((s) => s.session_id === sessionId)!,
      messages
    }))
    const onResume = vi.fn()
    const dismiss = vi.fn()

    const overlay = makeOverlay({ sessions, resolveResume, onResume })
    overlay.onDismiss = dismiss
    overlayActions.push(overlay)

    const { stdin, unmount } = render(<SessionPicker overlay={overlay} />)
    stdin.write('\u001B[B') // ↓
    await flush()
    stdin.write('\r') // Enter
    await flush(150)

    expect(resolveResume).toHaveBeenCalledWith('ses_beta0000')
    expect(onResume).toHaveBeenCalledWith(sessions[1], messages)
    expect(dismiss).toHaveBeenCalledWith('commit', { sessionId: 'ses_beta0000' })
    unmount()
  })

  it('ESC dismisses with cancel', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay({
      sessions: [makeSession('ses_a', 1)],
      resolveResume: vi.fn(),
      onResume: vi.fn()
    })
    overlay.onDismiss = dismiss
    overlayActions.push(overlay)

    const { stdin, unmount } = render(<SessionPicker overlay={overlay} />)
    stdin.write('\u001B')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('cancel', undefined)
    unmount()
  })
})
