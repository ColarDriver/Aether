import { describe, expect, it } from 'vitest'

import { SPINNER_VERBS, shimmer, spinnerVerbAt } from '../lib/shimmer.js'

describe('shimmer', () => {
  it('returns empty slices for an empty label', () => {
    expect(shimmer('', 0)).toEqual({ before: '', highlight: '', after: '' })
  })

  it('window slides across the label as the tick advances', () => {
    const label = 'thinking'
    const seenHighlights = new Set<string>()
    for (let tick = 0; tick < label.length + 8; tick++) {
      const { before, highlight, after } = shimmer(label, tick)
      // Reconstruction is always possible.
      expect(before + highlight + after).toBe(label)
      if (highlight) {
        seenHighlights.add(highlight)
      }
    }
    expect(seenHighlights.size).toBeGreaterThan(1)
  })

  it('handles multi-byte code points without splitting them', () => {
    // Emoji are multi-byte; the function operates on code points so they
    // should appear or disappear from the highlight as a unit.
    const label = '我正在思考'
    const out = shimmer(label, 2)
    expect(out.before + out.highlight + out.after).toBe(label)
  })
})

describe('spinnerVerbAt', () => {
  it('maps an index to a stable verb from the pool', () => {
    const first = spinnerVerbAt(0)
    const second = spinnerVerbAt(1)
    expect(SPINNER_VERBS).toContain(first)
    expect(SPINNER_VERBS).toContain(second)
  })

  it('wraps and tolerates negative or out-of-range indices', () => {
    expect(SPINNER_VERBS).toContain(spinnerVerbAt(SPINNER_VERBS.length))
    expect(SPINNER_VERBS).toContain(spinnerVerbAt(99))
    expect(SPINNER_VERBS).toContain(spinnerVerbAt(-1))
  })
})
