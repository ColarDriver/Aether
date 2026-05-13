import { atom } from 'nanostores'

export interface ReasoningState {
  /** Most-recent buffered reasoning text. */
  text: string
  /** Wall-clock ms when the latest delta arrived. */
  updatedAt: number | null
}

const initialState: ReasoningState = { text: '', updatedAt: null }

export const reasoningState = atom<ReasoningState>(initialState)

const MAX_CHARS = 240
const FADE_AFTER_MS = 8000

export const reasoningActions = {
  resetForTests(): void {
    reasoningState.set(initialState)
  },

  /**
   * Append a streaming reasoning chunk and trim to the most recent
   * MAX_CHARS so the bar never grows wider than one terminal row's worth
   * of text. Older characters are evicted from the head.
   */
  appendDelta(text: string): void {
    if (!text) {
      return
    }
    const current = reasoningState.get()
    const merged = (current.text + text).slice(-MAX_CHARS)
    reasoningState.set({ text: merged, updatedAt: Date.now() })
  },

  clear(): void {
    reasoningState.set(initialState)
  },

  /** UI helper — true if the line should be hidden because it is stale. */
  isStale(now: number = Date.now()): boolean {
    const updatedAt = reasoningState.get().updatedAt
    if (updatedAt === null) {
      return true
    }
    return now - updatedAt > FADE_AFTER_MS
  }
}
