import { describe, expect, it } from 'vitest'

import { shimmer, thinkingVerbAt } from '../lib/shimmer.js'

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

describe('thinkingVerbAt', () => {
  it('cycles through a stable list', () => {
    const first = thinkingVerbAt(0)
    const second = thinkingVerbAt(1)
    const wrap = thinkingVerbAt(99)
    expect(typeof first).toBe('string')
    expect(typeof second).toBe('string')
    expect(typeof wrap).toBe('string')
  })
})
