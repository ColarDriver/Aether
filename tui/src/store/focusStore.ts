import { atom } from 'nanostores'

export type FocusOwner = 'composer' | 'transcript'

/**
 * Coordinates which top-level surface owns the keyboard. We need this because
 * Ink permits multiple `useInput` consumers but key events fire on every
 * active subscriber — without arbitration, Tab in the composer would also
 * trigger transcript focus cycling, and Enter on a focused tool row would
 * also submit a (probably empty) composer message.
 *
 * Overlays are NOT a focus owner here — when an overlay is up, the
 * `overlayStack` length itself disables both composer and transcript via
 * their `isActive` checks.
 */
export const focusOwner = atom<FocusOwner>('composer')

export const focusActions = {
  resetForTests(): void {
    focusOwner.set('composer')
  },
  set(owner: FocusOwner): void {
    if (focusOwner.get() !== owner) {
      focusOwner.set(owner)
    }
  }
}
