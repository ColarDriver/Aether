import { ChevronDown, ChevronRight, FileCode2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { DiffBlock as DiffChatBlock } from '../../../chat-rendering'
import { DiffViewer, parseUnifiedDiff } from '../DiffViewer'

type Props = {
  diffs: DiffChatBlock[]
}

type FileChange = {
  path: string
  diff: string
  additions: number
  removals: number
}

export function CurrentTurnChangeCard({ diffs }: Props) {
  const changes = useMemo(() => summarizeDiffs(diffs), [diffs])
  const [expanded, setExpanded] = useState(false)
  if (changes.length === 0) return null
  const totals = changes.reduce((acc, change) => ({
    additions: acc.additions + change.additions,
    removals: acc.removals + change.removals,
  }), { additions: 0, removals: 0 })

  return (
    <section className="current-turn-change-card" aria-label="Changed files">
      <button
        type="button"
        className="current-turn-change-header"
        aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}
      >
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <FileCode2 size={15} />
        <strong>{changes.length} changed {changes.length === 1 ? 'file' : 'files'}</strong>
        <span className="current-turn-change-stats">
          <em className="change-add">+{totals.additions}</em>
          <em className="change-remove">-{totals.removals}</em>
        </span>
      </button>
      <div className="current-turn-change-files">
        {changes.map((change) => (
          <div className="current-turn-change-file" key={change.path}>
            <span>{change.path}</span>
            <em className="change-add">+{change.additions}</em>
            <em className="change-remove">-{change.removals}</em>
          </div>
        ))}
      </div>
      {expanded ? (
        <div className="current-turn-change-diffs">
          {changes.map((change) => (
            <article key={change.path} className="current-turn-change-diff">
              <header>{change.path}</header>
              <DiffViewer diff={change.diff} />
            </article>
          ))}
        </div>
      ) : null}
    </section>
  )
}

function summarizeDiffs(diffs: DiffChatBlock[]): FileChange[] {
  const byPath = new Map<string, string[]>()
  for (const diffBlock of diffs) {
    const diff = diffBlock.diff ?? diffFromOldNew(diffBlock.oldText ?? '', diffBlock.newText ?? '')
    if (!diff.trim()) continue
    const path = cleanPath(diffBlock.path) || pathFromUnifiedDiff(diff) || 'Changed file'
    const existing = byPath.get(path) ?? []
    existing.push(diff)
    byPath.set(path, existing)
  }
  return Array.from(byPath.entries()).map(([path, chunks]) => {
    const diff = chunks.join('\n')
    const parsed = parseUnifiedDiff(diff)
    return {
      path,
      diff,
      additions: parsed.filter((line) => line.kind === 'add').length,
      removals: parsed.filter((line) => line.kind === 'remove').length,
    }
  })
}

function diffFromOldNew(oldText: string, newText: string): string {
  const oldLines = oldText ? oldText.split('\n').map((line) => '-' + line) : []
  const newLines = newText ? newText.split('\n').map((line) => '+' + line) : []
  return [...oldLines, ...newLines].join('\n')
}

function cleanPath(path?: string | null): string | null {
  if (!path) return null
  return path.startsWith('tool:') ? null : path
}

function pathFromUnifiedDiff(diff: string): string | null {
  for (const line of diff.split('\n')) {
    if (line.startsWith('+++ ')) {
      return line.slice(4).replace(/^b\//, '').trim() || null
    }
  }
  return null
}
