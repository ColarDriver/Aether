/**
 * Animated highlight that "sweeps" across a label by code-point index. Used
 * by the activity bar to make the active verb feel alive without flashing.
 *
 * The implementation is intentionally simple: pick a window of N adjacent
 * code points and treat them as the highlighted slice for the current tick.
 * The renderer can then split the label into three slices (before / during /
 * after) and apply a brighter colour to the middle one.
 */

const SHIMMER_WINDOW = 4
const PADDING = 6

export interface ShimmerSlices {
  before: string
  highlight: string
  after: string
}

export function shimmer(label: string, tick: number): ShimmerSlices {
  if (!label) {
    return { before: '', highlight: '', after: '' }
  }
  const codePoints = Array.from(label)
  const total = codePoints.length + PADDING
  const start = ((tick % total) - PADDING + total) % total - PADDING
  const from = Math.max(0, start)
  const to = Math.min(codePoints.length, Math.max(from, from + SHIMMER_WINDOW))
  return {
    before: codePoints.slice(0, from).join(''),
    highlight: codePoints.slice(from, to).join(''),
    after: codePoints.slice(to).join('')
  }
}

const THINKING_VERBS: readonly string[] = [
  'thinking',
  'pondering',
  'forging',
  'channeling',
  'weaving',
  'composing',
  'refining',
  'planning',
  'sifting',
  'synthesising'
]

export function thinkingVerbAt(index: number): string {
  if (THINKING_VERBS.length === 0) {
    return 'thinking'
  }
  return THINKING_VERBS[index % THINKING_VERBS.length] ?? 'thinking'
}
