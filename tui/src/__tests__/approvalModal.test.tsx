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

  it('renders markdown plan content', () => {
    const overlay = makeOverlay({
      kind: 'plan',
      session_id: 's',
      run_id: 'r',
      plan_text: '# Plan\n- add auth\n\n```ts\nconst ok = true\n```',
      plan_path: '/tmp/plan.md',
      questions: [],
      deadline_ms: 0
    })
    const { lastFrame, unmount } = render(<ApprovalModal overlay={overlay} />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Plan approval')
    expect(frame).toContain('/tmp/plan.md')
    expect(frame).toContain('add auth')
    expect(frame).toContain('const ok = true')
    expect(frame).not.toContain(overlay.id)
    unmount()
  })

  it('N dismisses with confirmed:false', async () => {
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

    expect(dismiss).toHaveBeenCalledWith('commit', { confirmed: false, answers: {} })
    unmount()
  })

  it('ESC dismisses with confirmed:false', async () => {
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

    stdin.write('') // ESC
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { confirmed: false, answers: {} })
    unmount()
  })

  it('Ctrl-C dismisses with cancel', async () => {
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

    stdin.write('')
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

  function makeSelectOverlay(dismiss = vi.fn()): OverlayState<ApprovalRequestParams> {
    return makeOverlay(
      {
        kind: 'questions',
        session_id: 's',
        run_id: 'r',
        questions: [
          {
            id: 'q1',
            text: 'pick one',
            kind: 'select',
            options: [
              { label: 'a', description: 'option a' },
              { label: 'b', description: 'option b' },
              { label: 'c', description: 'option c' }
            ]
          }
        ],
        deadline_ms: 0
      },
      dismiss
    )
  }

  it('renders numbered options with descriptions and no overlay id', () => {
    const overlay = makeSelectOverlay(vi.fn())
    const { lastFrame, unmount } = render(<ApprovalModal overlay={overlay} />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('1. a')
    expect(frame).toContain('option a')
    expect(frame).toContain('2. b')
    expect(frame).toContain('Type something.')
    expect(frame).toContain('Chat about this')
    expect(frame).not.toContain('Skip interview')
    expect(frame).not.toContain(overlay.id)
    unmount()
  })

  it('select question: ↓ then Enter records the second option', async () => {
    const dismiss = vi.fn()
    const overlay = makeSelectOverlay(dismiss)
    const { stdin, unmount } = render(<ApprovalModal overlay={overlay} />)

    stdin.write('[B') // down arrow
    await flush()
    stdin.write('\r') // Enter
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { confirmed: true, answers: { q1: 'b' } })
    unmount()
  })

  it('select question: number key 3 picks the third option', async () => {
    const dismiss = vi.fn()
    const overlay = makeSelectOverlay(dismiss)
    const { stdin, unmount } = render(<ApprovalModal overlay={overlay} />)

    stdin.write('3')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { confirmed: true, answers: { q1: 'c' } })
    unmount()
  })

  it('Type something opens a text input that records on Enter', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(
      {
        kind: 'questions',
        session_id: 's',
        run_id: 'r',
        questions: [{ id: 'q1', text: 'free?', kind: 'open' }],
        deadline_ms: 0
      },
      dismiss
    )
    const { stdin, unmount } = render(<ApprovalModal overlay={overlay} />)

    // First row is "Type something." when the question has no options.
    stdin.write('\r')
    await flush()
    stdin.write('foo')
    await flush()
    stdin.write('\r')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', { confirmed: true, answers: { q1: 'foo' } })
    unmount()
  })

  it('Chat about this dismisses with action=chat', async () => {
    const dismiss = vi.fn()
    const overlay = makeSelectOverlay(dismiss)
    const { stdin, unmount } = render(<ApprovalModal overlay={overlay} />)

    // Rows: 1=a 2=b 3=c 4=Type 5=Chat
    stdin.write('5')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', {
      confirmed: false,
      action: 'chat',
      answers: {}
    })
    unmount()
  })

  it('Skip interview only renders in plan mode and dismisses with action=skip', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(
      {
        kind: 'questions',
        session_id: 's',
        run_id: 'r',
        plan_path: '/tmp/plan.md',
        questions: [
          {
            id: 'q1',
            text: 'pick one',
            kind: 'select',
            options: [
              { label: 'a', description: 'option a' },
              { label: 'b', description: 'option b' }
            ]
          }
        ],
        deadline_ms: 0
      },
      dismiss
    )
    const { stdin, lastFrame, unmount } = render(<ApprovalModal overlay={overlay} />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Planning: /tmp/plan.md')
    expect(frame).toContain('Skip interview and plan immediately')

    // Rows: 1=a 2=b 3=Type 4=Chat 5=Skip
    stdin.write('5')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('commit', {
      confirmed: true,
      action: 'skip',
      answers: {}
    })
    unmount()
  })

  it('renders chip bar across multiple questions and a Submit chip', () => {
    const overlay = makeOverlay({
      kind: 'questions',
      session_id: 's',
      run_id: 'r',
      questions: [
        { id: 'q1', header: 'Interface', text: 'pick interface' },
        { id: 'q2', header: 'Tools', text: 'pick tools' }
      ],
      deadline_ms: 0
    })
    const { lastFrame, unmount } = render(<ApprovalModal overlay={overlay} />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Interface')
    expect(frame).toContain('Tools')
    expect(frame).toContain('Submit')
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

  it('questions flow: Ctrl-C dismisses with cancel', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay(
      {
        kind: 'questions',
        session_id: 's',
        run_id: 'r',
        questions: [{ id: 'q1', text: 'first?', kind: 'open' }],
        deadline_ms: 0
      },
      dismiss
    )
    const { stdin, unmount } = render(<ApprovalModal overlay={overlay} />)

    stdin.write('')
    await flush()

    expect(dismiss).toHaveBeenCalledWith('cancel', undefined)
    unmount()
  })
})
