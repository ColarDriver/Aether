/**
 * Hand-rolled unified diff parser + colorisable line model.
 *
 * The Python prompter sends `preview.diff` as a unified diff string
 * (the same shape `git diff` produces). A real diff library (`diff` /
 * `unidiff`) would be overkill for what is effectively per-line classification
 * + folding, so we keep this layer dependency-free and test it as pure data.
 *
 * The TSX renderer in diffRender.tsx maps each `DiffLine.kind` to an
 * Ink `<Text>` color. Splitting parser from renderer also lets the unit tests
 * stay environment-agnostic (no React tree needed).
 */

export type DiffLineKind =
  | 'file-header'
  | 'hunk-header'
  | 'addition'
  | 'deletion'
  | 'context'
  | 'no-newline'
  | 'meta'

export interface DiffLine {
  kind: DiffLineKind
  text: string
}

export interface DiffStats {
  additions: number
  deletions: number
  hunks: number
}

export interface ParsedDiff {
  lines: DiffLine[]
  stats: DiffStats
}

/**
 * Classify each line of a unified diff. Anything we cannot recognise is
 * routed to `meta` so the renderer can show it dimmed without losing data.
 */
export function parseUnifiedDiff(input: string): ParsedDiff {
  const lines: DiffLine[] = []
  const stats: DiffStats = { additions: 0, deletions: 0, hunks: 0 }
  if (!input) {
    return { lines, stats }
  }

  // Use split('\n') (not splitlines) so we preserve a trailing empty line
  // exactly as the producer wrote it.
  const raw = input.endsWith('\n') ? input.slice(0, -1) : input
  const rawLines = raw.split('\n')

  for (const text of rawLines) {
    const kind = classifyLine(text)
    if (kind === 'addition') {
      stats.additions++
    } else if (kind === 'deletion') {
      stats.deletions++
    } else if (kind === 'hunk-header') {
      stats.hunks++
    }
    lines.push({ kind, text })
  }
  return { lines, stats }
}

function classifyLine(line: string): DiffLineKind {
  if (line.startsWith('@@')) {
    return 'hunk-header'
  }
  if (line.startsWith('+++') || line.startsWith('---')) {
    return 'file-header'
  }
  if (line.startsWith('diff ') || line.startsWith('index ')) {
    return 'meta'
  }
  if (line.startsWith('+')) {
    return 'addition'
  }
  if (line.startsWith('-')) {
    return 'deletion'
  }
  if (line.startsWith('\\')) {
    return 'no-newline'
  }
  return 'context'
}

export const DEFAULT_FOLD_THRESHOLD = 40

/**
 * Compute which lines to show when collapsed. We always preserve every
 * file/hunk/meta marker (so the user sees structure), then pick the first
 * N additions/deletions/context lines. Returning indices keeps the renderer
 * simple — it just consults `visible.has(idx)`.
 */
export function visibleLineIndices(
  lines: DiffLine[],
  options: { expanded: boolean; foldThreshold?: number } = { expanded: true }
): { indices: number[]; hiddenCount: number } {
  if (options.expanded) {
    return { indices: lines.map((_, idx) => idx), hiddenCount: 0 }
  }
  const limit = options.foldThreshold ?? DEFAULT_FOLD_THRESHOLD
  if (lines.length <= limit) {
    return { indices: lines.map((_, idx) => idx), hiddenCount: 0 }
  }

  const indices: number[] = []
  let payloadShown = 0
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (!line) {
      continue
    }
    const isStructural =
      line.kind === 'file-header' || line.kind === 'hunk-header' || line.kind === 'meta'
    if (isStructural) {
      indices.push(i)
      continue
    }
    if (payloadShown < limit) {
      indices.push(i)
      payloadShown++
    }
  }
  const hiddenCount = lines.length - indices.length
  return { indices, hiddenCount }
}

/**
 * Convenience: returns a one-line summary like `+12 −3 across 2 hunks`.
 * Used at the top of the permission modal so the user sees impact at a glance
 * even before scrolling the diff.
 */
export function formatStats(stats: DiffStats): string {
  const parts: string[] = []
  parts.push(`+${stats.additions}`)
  parts.push(`-${stats.deletions}`)
  if (stats.hunks > 0) {
    parts.push(`${stats.hunks} hunk${stats.hunks === 1 ? '' : 's'}`)
  }
  return parts.join(' · ')
}
