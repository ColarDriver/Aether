import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render } from 'ink-testing-library'

import { ApprovalModal } from '../overlays/ApprovalModal.js'
import type { ApprovalRequestParams } from '../gatewayTypes.js'
import { OVERLAY_PRIORITY, overlayActions, type OverlayState } from '../store/overlayStore.js'

// Ink debounces input parsing (~30 ms for ESC disambiguation) and renders on
// the next tick. Each test waits one settle period after every keystroke so
// React state propagates before assertions run.
const SETTLE_MS = 80
const flush = (ms: number = SETTLE_MS): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms))

let nextCreatedAt = 1
function makeOverlay(
  payload: ApprovalRequestParams,
  onDismiss = vi.fn()
): OverlayState<ApprovalRequestParams> {
  const id = `srv_app_${nextCreatedAt}`
  const overlay: OverlayState<ApprovalRequestParams> = {
    kind: 'approval',
    id,
    payload,
    createdAt: nextCreatedAt++,
    priority: OVERLAY_PRIORITY.approval,
    onDismiss
  }
  overlayActions.push(overlay)
  return overlay
}

describe('ApprovalModal — plan kind', () => {
  beforeEach(() => {
    overlayActions.resetForTests()
    nextCreatedAt = 1
  })
  afterEach(() => {
    overlayActions.resetForTests()
  })

  it('Y dismisses with confirmed:true', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(
      {
        kind: 'plan',
        session_id: 's',
        run_id: 'r',
        plan_text: 'do the thing',
        questions: [],
        deadline_ms: 0
      },
      dismiss
    )
    const { stdin, lastFrame, unmount } = render(<ApprovalModal overlay={overlay} />)
    expect(lastFrame()).toContain('Plan approval')
    expect(lastFrame()).toContain('do the thing')

    stdin.write('y')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { confirmed: true, answers: {} })
    unmount()
  })

  it('N dismisses with confirmed:false (cancel)', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(
      {
        kind: 'plan',
        session_id: 's',
        run_id: 'r',
        plan_text: 'do x',
        questions: [],
        deadline_ms: 0
      },
      dismiss
    )
    const { stdin, unmount } = render(<ApprovalModal overlay={overlay} />)

    stdin.write('n')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('cancel', undefined)
    unmount()
  })

  it('ESC dismisses with cancel', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(
      {
        kind: 'plan',
        session_id: 's',
        run_id: 'r',
        plan_text: 'do x',
        questions: [],
        deadline_ms: 0
      },
      dismiss
    )
    const { stdin, unmount } = render(<ApprovalModal overlay={overlay} />)

    stdin.write('\u001B') // ESC
    await flush()

    expect(dismiss).toHaveBeenCalledWith('cancel', undefined)
    unmount()
  })
})

describe('ApprovalModal — questions kind', () => {
  beforeEach(() => {
    overlayActions.resetForTests()
    nextCreatedAt = 1
  })
  afterEach(() => {
    overlayActions.resetForTests()
  })

  it('select question: ↓ then Enter records the second option', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(
      {
        kind: 'questions',
        session_id: 's',
        run_id: 'r',
        questions: [
          { id: 'q1', text: 'pick one', kind: 'select', options: ['a', 'b', 'c'] }
        ],
        deadline_ms: 0
      },
      dismiss
    )
    const { stdin, lastFrame, unmount } = render(<ApprovalModal overlay={overlay} />)
    expect(lastFrame()).toContain('Questions (1/1)')

    stdin.write('\u001B[B') // down arrow
    await flush()
    stdin.write('\r') // Enter
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { confirmed: true, answers: { q1: 'b' } })
    unmount()
  })

  it('select question: number key 3 picks the third option', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(
      {
        kind: 'questions',
        session_id: 's',
        run_id: 'r',
        questions: [
          { id: 'q1', text: 'pick one', kind: 'select', options: ['a', 'b', 'c'] }
        ],
        deadline_ms: 0
      },
      dismiss
    )
    const { stdin, unmount } = render(<ApprovalModal overlay={overlay} />)

    stdin.write('3')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { confirmed: true, answers: { q1: 'c' } })
    unmount()
  })

  it('multiple open questions accumulate answers and submit on the last Enter', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(
      {
        kind: 'questions',
        session_id: 's',
        run_id: 'r',
        questions: [
          { id: 'q1', text: 'first?', kind: 'open', options: [] },
          { id: 'q2', text: 'second?', kind: 'open', options: [] }
        ],
        deadline_ms: 0
      },
      dismiss
    )
    const { stdin, lastFrame, unmount } = render(<ApprovalModal overlay={overlay} />)

    stdin.write('foo')
    await flush()
    stdin.write('\r')
    await flush()
    expect(lastFrame()).toContain('(2/2)')
    expect(dismiss).not.toHaveBeenCalled()

    stdin.write('bar')
    await flush()
    stdin.write('\r')
    await flush()

    expect(dismiss).toHaveBeenCalledOnce()
    expect(dismiss).toHaveBeenCalledWith('commit', {
      confirmed: true,
      answers: { q1: 'foo', q2: 'bar' }
    })
    unmount()
  })

  it('empty questions list auto-confirms', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(
      {
        kind: 'questions',
        session_id: 's',
        run_id: 'r',
        questions: [],
        deadline_ms: 0
      },
      dismiss
    )
    const { unmount } = render(<ApprovalModal overlay={overlay} />)
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { confirmed: true, answers: {} })
    unmount()
  })
})
