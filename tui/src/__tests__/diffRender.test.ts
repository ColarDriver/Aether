import { describe, expect, it } from 'vitest'

import {
  DEFAULT_FOLD_THRESHOLD,
  formatStats,
  parseUnifiedDiff,
  visibleLineIndices
} from '../lib/diffRender.js'

const SMALL_DIFF = `--- a/foo.txt
+++ b/foo.txt
@@ -1,3 +1,3 @@
 ctx
-old
+new
 last
`

describe('parseUnifiedDiff', () => {
  it('classifies headers, hunks, additions, deletions, and context', () => {
    const parsed = parseUnifiedDiff(SMALL_DIFF)
    expect(parsed.lines.map((line) => line.kind)).toEqual([
      'file-header',
      'file-header',
      'hunk-header',
      'context',
      'deletion',
      'addition',
      'context'
    ])
    expect(parsed.stats).toEqual({ additions: 1, deletions: 1, hunks: 1 })
  })

  it('handles meta lines and \\ No newline markers', () => {
    const diff = `diff --git a/foo b/foo
index 1234..5678 100644
--- a/foo
+++ b/foo
@@ -1 +1 @@
-old
\\ No newline at end of file
+new
`
    const parsed = parseUnifiedDiff(diff)
    const kinds = parsed.lines.map((line) => line.kind)
    expect(kinds).toContain('meta')
    expect(kinds).toContain('no-newline')
  })

  it('returns an empty result for empty input', () => {
    expect(parseUnifiedDiff('')).toEqual({
      lines: [],
      stats: { additions: 0, deletions: 0, hunks: 0 }
    })
  })

  it('keeps trailing blank lines unobtrusive', () => {
    const parsed = parseUnifiedDiff('@@ -1 +1 @@\n a\n\n')
    expect(parsed.lines.length).toBeGreaterThanOrEqual(2)
  })
})

describe('visibleLineIndices', () => {
  it('returns every index when expanded', () => {
    const parsed = parseUnifiedDiff(SMALL_DIFF)
    const result = visibleLineIndices(parsed.lines, { expanded: true })
    expect(result.indices).toEqual(parsed.lines.map((_, idx) => idx))
    expect(result.hiddenCount).toBe(0)
  })

  it('returns every index when total lines fall below the fold threshold', () => {
    const parsed = parseUnifiedDiff(SMALL_DIFF)
    const result = visibleLineIndices(parsed.lines, { expanded: false, foldThreshold: 50 })
    expect(result.hiddenCount).toBe(0)
  })

  it('hides payload lines beyond the threshold but always keeps headers', () => {
    const lines = [
      '--- a/big.txt',
      '+++ b/big.txt',
      '@@ -1,100 +1,100 @@',
      ...Array.from({ length: 80 }, (_, idx) => `+line ${idx}`)
    ].join('\n')
    const parsed = parseUnifiedDiff(lines)
    const result = visibleLineIndices(parsed.lines, { expanded: false, foldThreshold: 5 })
    // Three structural + five payload lines = 8 visible.
    expect(result.indices.length).toBe(3 + 5)
    expect(result.hiddenCount).toBe(parsed.lines.length - (3 + 5))
    // The first three visible indices must be the headers.
    expect(parsed.lines[result.indices[0]!]?.kind).toBe('file-header')
    expect(parsed.lines[result.indices[1]!]?.kind).toBe('file-header')
    expect(parsed.lines[result.indices[2]!]?.kind).toBe('hunk-header')
  })

  it('uses DEFAULT_FOLD_THRESHOLD when none supplied', () => {
    const lines = Array.from({ length: 200 }, (_, idx) => `+line ${idx}`).join('\n')
    const parsed = parseUnifiedDiff(lines)
    const result = visibleLineIndices(parsed.lines, { expanded: false })
    expect(result.indices.length).toBe(DEFAULT_FOLD_THRESHOLD)
    expect(result.hiddenCount).toBe(parsed.lines.length - DEFAULT_FOLD_THRESHOLD)
  })
})

describe('formatStats', () => {
  it('formats single-hunk stats', () => {
    expect(formatStats({ additions: 5, deletions: 2, hunks: 1 })).toBe('+5 · -2 · 1 hunk')
  })

  it('formats multi-hunk stats with plural', () => {
    expect(formatStats({ additions: 12, deletions: 3, hunks: 4 })).toBe(
      '+12 · -3 · 4 hunks'
    )
  })

  it('omits the hunk segment when hunks is zero', () => {
    expect(formatStats({ additions: 0, deletions: 0, hunks: 0 })).toBe('+0 · -0')
  })
})
