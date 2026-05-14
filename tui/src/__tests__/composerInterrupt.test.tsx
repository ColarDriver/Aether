import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render } from 'ink-testing-library'

import { Composer } from '../components/Composer.js'
import { composerActions } from '../store/composerStore.js'
import { focusActions } from '../store/focusStore.js'
import { overlayActions } from '../store/overlayStore.js'
import { sessionActions } from '../store/sessionStore.js'

const SETTLE_MS = 80
const flush = (ms: number = SETTLE_MS): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms))

describe('Composer interrupt parity', () => {
  beforeEach(() => {
    composerActions.resetForTests()
    focusActions.resetForTests()
    overlayActions.resetForTests()
    sessionActions.resetForTests()
  })

  afterEach(() => {
    composerActions.resetForTests()
    focusActions.resetForTests()
    overlayActions.resetForTests()
    sessionActions.resetForTests()
  })

  it('Ctrl-C interrupts a running turn even when the draft is non-empty', async () => {
    const onCancel = vi.fn()
    composerActions.setDraft('typed while running')

    const { stdin, unmount } = render(
      <Composer busy={true} onSubmit={() => undefined} onCancel={onCancel} />
    )

    stdin.write('\u0003')
    await flush()

    expect(onCancel).toHaveBeenCalledOnce()
    expect(composerActions).toBeDefined()
    unmount()
  })
})

describe('Composer slash popup navigation', () => {
  beforeEach(() => {
    composerActions.resetForTests()
    focusActions.resetForTests()
    overlayActions.resetForTests()
    sessionActions.resetForTests()
    sessionActions.setCatalog([
      { name: '/clear', description: 'clear', category: 'local' },
      { name: '/exit', description: 'exit', category: 'local' },
      { name: '/help', description: 'help', category: 'local' }
    ])
    focusActions.set('composer')
  })

  afterEach(() => {
    composerActions.resetForTests()
    focusActions.resetForTests()
    overlayActions.resetForTests()
    sessionActions.resetForTests()
  })

  it('Down arrow cycles the slash popup forward, not the prompt history', async () => {
    composerActions.setDraft('/')

    const { stdin, unmount } = render(
      <Composer onSubmit={() => undefined} onCancel={() => undefined} />
    )
    await flush()

    // Down: cursor 0 -> 1 -> `/exit` becomes active and the draft is stamped.
    stdin.write('[B')
    await flush()
    const { composerState } = await import('../store/composerStore.js')
    expect(composerState.get().draft).toBe('/exit')

    // Second Down: 1 -> 2 -> `/help`.
    stdin.write('[B')
    await flush()
    expect(composerState.get().draft).toBe('/help')
    unmount()
  })

  it('Up arrow cycles the slash popup backward when it is open', async () => {
    composerActions.setDraft('/')

    const { stdin, unmount } = render(
      <Composer onSubmit={() => undefined} onCancel={() => undefined} />
    )
    await flush()

    // Up wraps from 0 to last -> `/help`.
    stdin.write('[A')
    await flush()
    const { composerState } = await import('../store/composerStore.js')
    expect(composerState.get().draft).toBe('/help')
    unmount()
  })
})
