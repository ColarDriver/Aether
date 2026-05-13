import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  OVERLAY_PRIORITY,
  hasOverlay,
  overlayActions,
  overlayStack,
  topOverlay,
  type DismissReason,
  type OverlayKind,
  type OverlayState
} from '../store/overlayStore.js'

let nextCreatedAt = 1

function makeOverlay(
  kind: OverlayKind,
  id: string,
  onDismiss: (reason: DismissReason, result?: unknown) => void = () => undefined
): OverlayState {
  const overlay: OverlayState = {
    kind,
    id,
    payload: null,
    createdAt: nextCreatedAt++,
    priority: OVERLAY_PRIORITY[kind],
    onDismiss
  }
  return overlay
}

describe('overlayStore', () => {
  beforeEach(() => {
    overlayActions.resetForTests()
    nextCreatedAt = 1
  })

  it('starts empty', () => {
    expect(overlayStack.get()).toEqual([])
    expect(topOverlay()).toBeNull()
    expect(hasOverlay()).toBe(false)
  })

  it('orders pushes by priority — top of stack is the highest priority overlay', () => {
    overlayActions.push(makeOverlay('help', 'h1'))
    overlayActions.push(makeOverlay('approval', 'a1'))
    overlayActions.push(makeOverlay('permission', 'p1'))

    expect(overlayStack.get().map((o) => o.kind)).toEqual([
      'help',
      'approval',
      'permission'
    ])
    expect(topOverlay()?.kind).toBe('permission')
  })

  it('keeps later equal-priority overlays on top (stable ordering by createdAt)', () => {
    overlayActions.push(makeOverlay('approval', 'first'))
    overlayActions.push(makeOverlay('approval', 'second'))

    expect(topOverlay()?.id).toBe('second')
  })

  it('reorders correctly when a high-priority overlay arrives after a low one', () => {
    overlayActions.push(makeOverlay('help', 'h1'))
    overlayActions.push(makeOverlay('permission', 'p1'))
    overlayActions.push(makeOverlay('approval', 'a1'))

    expect(overlayStack.get().map((o) => o.kind)).toEqual([
      'help',
      'approval',
      'permission'
    ])
  })

  it('replaces an overlay pushed with the same id', () => {
    const firstDismiss = vi.fn()
    const secondDismiss = vi.fn()
    overlayActions.push(makeOverlay('approval', 'dup', firstDismiss))
    overlayActions.push(makeOverlay('approval', 'dup', secondDismiss))

    const stack = overlayStack.get()
    expect(stack).toHaveLength(1)
    overlayActions.dismissTop()
    expect(firstDismiss).not.toHaveBeenCalled()
    expect(secondDismiss).toHaveBeenCalledOnce()
  })

  it('ESC chain: dismissTop pops one overlay at a time, in the documented priority order', () => {
    const helpDismiss = vi.fn()
    const approvalDismiss = vi.fn()
    const permissionDismiss = vi.fn()

    overlayActions.push(makeOverlay('help', 'h1', helpDismiss))
    overlayActions.push(makeOverlay('approval', 'a1', approvalDismiss))
    overlayActions.push(makeOverlay('permission', 'p1', permissionDismiss))

    expect(topOverlay()?.kind).toBe('permission')

    expect(overlayActions.dismissTop()).toBe(true)
    expect(permissionDismiss).toHaveBeenCalledOnce()
    expect(permissionDismiss).toHaveBeenCalledWith('cancel', undefined)
    expect(approvalDismiss).not.toHaveBeenCalled()
    expect(helpDismiss).not.toHaveBeenCalled()

    expect(topOverlay()?.kind).toBe('approval')
    overlayActions.dismissTop()
    expect(approvalDismiss).toHaveBeenCalledOnce()
    expect(helpDismiss).not.toHaveBeenCalled()

    expect(topOverlay()?.kind).toBe('help')
    overlayActions.dismissTop()
    expect(helpDismiss).toHaveBeenCalledOnce()

    expect(topOverlay()).toBeNull()
    expect(overlayActions.dismissTop()).toBe(false)
  })

  it('ESC chain: session-picker sits between approval and help', () => {
    overlayActions.push(makeOverlay('help', 'h1'))
    overlayActions.push(makeOverlay('session-picker', 'sp1'))
    overlayActions.push(makeOverlay('approval', 'a1'))

    expect(overlayStack.get().map((o) => o.kind)).toEqual([
      'help',
      'session-picker',
      'approval'
    ])
  })

  it('dismiss(id, "commit", payload) forwards the payload to onDismiss', () => {
    const dismiss = vi.fn()
    overlayActions.push(makeOverlay('approval', 'a1', dismiss))

    overlayActions.dismiss('a1', 'commit', { confirmed: true })

    expect(dismiss).toHaveBeenCalledWith('commit', { confirmed: true })
  })

  it('dismiss returns false when the id does not exist', () => {
    expect(overlayActions.dismiss('missing')).toBe(false)
  })

  it('dismissAll clears the stack and invokes every onDismiss with the given reason', () => {
    const a = vi.fn()
    const b = vi.fn()
    overlayActions.push(makeOverlay('help', 'h1', a))
    overlayActions.push(makeOverlay('approval', 'a1', b))

    overlayActions.dismissAll('timeout')

    expect(overlayStack.get()).toEqual([])
    expect(a).toHaveBeenCalledWith('timeout')
    expect(b).toHaveBeenCalledWith('timeout')
  })

  it('an exception thrown by onDismiss does not corrupt the stack', () => {
    const explode = () => {
      throw new Error('boom')
    }
    overlayActions.push(makeOverlay('approval', 'a1', explode))
    overlayActions.push(makeOverlay('permission', 'p1'))

    expect(() => overlayActions.dismiss('a1')).not.toThrow()
    expect(overlayStack.get().map((o) => o.id)).toEqual(['p1'])
  })
})
