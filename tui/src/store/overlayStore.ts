import { atom } from 'nanostores'

export type OverlayKind =
  | 'placeholder'
  | 'approval'
  | 'permission'
  | 'session-picker'
  | 'picker'
  | 'help'

export type DismissReason = 'cancel' | 'commit' | 'timeout'

export interface OverlayState<TPayload = unknown> {
  kind: OverlayKind
  id: string
  payload: TPayload
  createdAt: number
  priority: number
  onDismiss(reason: DismissReason, result?: unknown): void
}

// Priority drives stacking order (higher renders / receives keys first).
// Aligned with PR 8 ESC chain: permission > approval > picker > placeholder > help.
export const OVERLAY_PRIORITY: Record<OverlayKind, number> = {
  permission: 100,
  approval: 90,
  'session-picker': 70,
  picker: 65,
  placeholder: 60,
  help: 50
}

export const overlayStack = atom<OverlayState[]>([])

function setStackSorted(stack: OverlayState[]): void {
  // Stable sort by priority ascending — last element is the visually top overlay.
  const next = [...stack].sort((a, b) => {
    if (a.priority !== b.priority) {
      return a.priority - b.priority
    }
    return a.createdAt - b.createdAt
  })
  overlayStack.set(next)
}

export const overlayActions = {
  resetForTests(): void {
    overlayStack.set([])
  },

  push<TPayload>(state: OverlayState<TPayload>): void {
    const current = overlayStack.get()
    // Replace any existing overlay with the same id (defensive against duplicate
    // server_request frames) so the dismiss callback always points at the latest.
    const filtered = current.filter((entry) => entry.id !== state.id)
    setStackSorted([...filtered, state as OverlayState])
  },

  has(id: string): boolean {
    return overlayStack.get().some((entry) => entry.id === id)
  },

  get(id: string): OverlayState | null {
    return overlayStack.get().find((entry) => entry.id === id) ?? null
  },

  dismiss(id: string, reason: DismissReason = 'cancel', result?: unknown): boolean {
    const current = overlayStack.get()
    const target = current.find((entry) => entry.id === id)
    if (!target) {
      return false
    }
    overlayStack.set(current.filter((entry) => entry.id !== id))
    try {
      target.onDismiss(reason, result)
    } catch {
      // The dismiss callback is supplied by the caller; we deliberately swallow
      // exceptions so a buggy modal cannot leave the store in an inconsistent
      // intermediate state.
    }
    return true
  },

  dismissTop(reason: DismissReason = 'cancel', result?: unknown): boolean {
    const current = overlayStack.get()
    const top = current[current.length - 1]
    if (!top) {
      return false
    }
    return overlayActions.dismiss(top.id, reason, result)
  },

  dismissAll(reason: DismissReason = 'cancel'): void {
    const current = overlayStack.get()
    overlayStack.set([])
    for (const entry of current) {
      try {
        entry.onDismiss(reason)
      } catch {
        // see dismiss()
      }
    }
  }
}

export function topOverlay(): OverlayState | null {
  const stack = overlayStack.get()
  return stack[stack.length - 1] ?? null
}

export function hasOverlay(): boolean {
  return overlayStack.get().length > 0
}
