import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render } from 'ink-testing-library'

import type { SlashCommandInfo } from '../gatewayTypes.js'
import { HelpOverlay, type HelpOverlayPayload } from '../overlays/HelpOverlay.js'
import { OVERLAY_PRIORITY, overlayActions, type OverlayState } from '../store/overlayStore.js'

const SETTLE_MS = 80
const flush = (ms: number = SETTLE_MS): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms))

const CATALOG: SlashCommandInfo[] = [
  { name: '/help', description: 'show help', category: 'local' },
  { name: '/exit', description: 'quit', category: 'local' },
  { name: '/resume', description: 'resume a session', category: 'remote' },
  { name: '/model', description: 'switch model', category: 'session' },
  { name: '/interrupt', description: 'interrupt current run', category: 'control' }
]

let nextCreatedAt = 1
function makeOverlay(payload: HelpOverlayPayload): OverlayState<HelpOverlayPayload> {
  const overlay: OverlayState<HelpOverlayPayload> = {
    kind: 'help',
    id: `help_${nextCreatedAt}`,
    payload,
    createdAt: nextCreatedAt++,
    priority: OVERLAY_PRIORITY.help,
    onDismiss: vi.fn()
  }
  overlayActions.push(overlay)
  return overlay
}

describe('HelpOverlay', () => {
  beforeEach(() => {
    overlayActions.resetForTests()
    nextCreatedAt = 1
  })
  afterEach(() => {
    overlayActions.resetForTests()
  })

  it('renders commands in flat alphabetical order with descriptions', () => {
    // Aligned with Python `_cmd_help` in `aether/cli/commands.py:53-65`,
    // which prints a flat Rich Table sorted by command name. The previous
    // category-grouped layout was a TS-only embellishment.
    const overlay = makeOverlay({ catalog: CATALOG })
    const { lastFrame, unmount } = render(<HelpOverlay overlay={overlay} />)
    const frame = lastFrame() ?? ''
    const exitPos = frame.indexOf('/exit')
    const helpPos = frame.indexOf('/help')
    const interruptPos = frame.indexOf('/interrupt')
    const modelPos = frame.indexOf('/model')
    const resumePos = frame.indexOf('/resume')
    expect(exitPos).toBeGreaterThan(-1)
    expect(helpPos).toBeGreaterThan(-1)
    expect(interruptPos).toBeGreaterThan(-1)
    expect(modelPos).toBeGreaterThan(-1)
    expect(resumePos).toBeGreaterThan(-1)
    // Alphabetical: /exit < /help < /interrupt < /model < /resume
    expect(exitPos).toBeLessThan(helpPos)
    expect(helpPos).toBeLessThan(interruptPos)
    expect(interruptPos).toBeLessThan(modelPos)
    expect(modelPos).toBeLessThan(resumePos)
    // Descriptions appear inline next to each name.
    expect(frame).toContain('show help')
    expect(frame).toContain('switch model')
    unmount()
  })

  it('ESC dismisses', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay({ catalog: CATALOG })
    overlay.onDismiss = dismiss
    overlayActions.push(overlay)

    const { stdin, unmount } = render(<HelpOverlay overlay={overlay} />)
    stdin.write('\u001B')
    await flush()
    expect(dismiss).toHaveBeenCalledWith('cancel', undefined)
    unmount()
  })

  it('? toggles closes the overlay', async () => {
    const dismiss = vi.fn()
    const overlay = makeOverlay({ catalog: CATALOG })
    overlay.onDismiss = dismiss
    overlayActions.push(overlay)

    const { stdin, unmount } = render(<HelpOverlay overlay={overlay} />)
    stdin.write('?')
    await flush()
    expect(dismiss).toHaveBeenCalledWith('cancel', undefined)
    unmount()
  })
})
