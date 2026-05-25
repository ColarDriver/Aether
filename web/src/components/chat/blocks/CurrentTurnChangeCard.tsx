import { AlertCircle, AlertTriangle, ChevronDown, ChevronRight, FileCode2, Info } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { DiagnosticEntry, DiagnosticsBlock as DiagnosticsChatBlock, DiffBlock as DiffChatBlock } from '../../../chat-rendering'
import { CopyButton } from '../../shared/CopyButton'
import { DiffViewer, parseUnifiedDiff } from '../DiffViewer'

type Props = {
  diffs: DiffChatBlock[]
  diagnostics?: DiagnosticsChatBlock[]
}

type FileChangeKind = 'created' | 'deleted' | 'modified'
type DiagnosticTone = 'error' | 'warning' | 'info'

type FileChange = {
  path: string
  diff: string
  kind: FileChangeKind
  additions: number
  removals: number
  hunks: number
  diagnostics: DiagnosticEntry[]
}

type DiagnosticCounts = {
  total: number
  errors: number
  warnings: number
  infos: number
}

export function CurrentTurnChangeCard({ diffs, diagnostics = [] }: Props) {
  const changes = useMemo(() => summarizeDiffs(diffs, diagnostics), [diffs, diagnostics])
  const [expanded, setExpanded] = useState(false)
  if (changes.length === 0) return null
  const totals = summarizeChanges(changes)

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
        <span className="current-turn-change-kind-summary" aria-label="File change summary">
          {totals.created > 0 ? <em className="change-created">{totals.created} created</em> : null}
          {totals.modified > 0 ? <em>{totals.modified} modified</em> : null}
          {totals.deleted > 0 ? <em className="change-deleted">{totals.deleted} deleted</em> : null}
        </span>
        <DiagnosticPills counts={totals.diagnostics} compact />
        <span className="current-turn-change-stats">
          <em className="change-add">+{totals.additions}</em>
          <em className="change-remove">-{totals.removals}</em>
        </span>
      </button>
      <div className="current-turn-change-files">
        {changes.map((change) => {
          const diagnosticCounts = countDiagnostics(change.diagnostics)
          return (
            <div className={'current-turn-change-file current-turn-change-file-' + change.kind} key={change.path}>
              <span className="current-turn-change-path">{change.path}</span>
              <em className={'current-turn-change-kind change-' + change.kind}>{change.kind}</em>
              <DiagnosticPills counts={diagnosticCounts} compact />
              <span className="current-turn-change-file-stats">
                {change.hunks > 0 ? <em>{change.hunks} hunk{change.hunks === 1 ? '' : 's'}</em> : null}
                <em className="change-add">+{change.additions}</em>
                <em className="change-remove">-{change.removals}</em>
              </span>
              <CopyButton
                text={change.path}
                label={'Copy ' + change.path}
                displayLabel="Copy"
                displayCopiedLabel="Copied"
                className="current-turn-change-copy"
              />
            </div>
          )
        })}
      </div>
      {expanded ? (
        <div className="current-turn-change-diffs">
          {changes.map((change) => (
            <article key={change.path} className="current-turn-change-diff">
              <header>
                <span>{change.path}</span>
                <span className="current-turn-change-diff-meta">
                  <DiagnosticPills counts={countDiagnostics(change.diagnostics)} />
                  <em className={'change-' + change.kind}>{change.kind}</em>
                </span>
              </header>
              <DiffViewer diff={change.diff} />
              <ChangeDiagnostics diagnostics={change.diagnostics} />
            </article>
          ))}
        </div>
      ) : null}
    </section>
  )
}

function summarizeDiffs(diffs: DiffChatBlock[], diagnostics: DiagnosticsChatBlock[]): FileChange[] {
  const byPath = new Map<string, string[]>()
  for (const diffBlock of diffs) {
    const diff = diffBlock.diff ?? diffFromOldNew(diffBlock.oldText ?? '', diffBlock.newText ?? '')
    if (!diff.trim()) continue
    const parsedHeader = parseUnifiedDiffHeader(diff)
    const path = cleanPath(diffBlock.path) || parsedHeader.path || 'Changed file'
    const existing = byPath.get(path) ?? []
    existing.push(diff)
    byPath.set(path, existing)
  }
  const diagnosticFiles = flattenDiagnostics(diagnostics)
  return Array.from(byPath.entries()).map(([path, chunks]) => {
    const diff = chunks.join('\n')
    const parsed = parseUnifiedDiff(diff)
    const header = parseUnifiedDiffHeader(diff)
    return {
      path,
      diff,
      kind: header.kind,
      additions: parsed.filter((line) => line.kind === 'add').length,
      removals: parsed.filter((line) => line.kind === 'remove').length,
      hunks: diff.split('\n').filter((line) => line.startsWith('@@')).length,
      diagnostics: diagnosticsForPath(path, diagnosticFiles),
    }
  })
}

function summarizeChanges(changes: FileChange[]) {
  const totals = changes.reduce((acc, change) => ({
    additions: acc.additions + change.additions,
    removals: acc.removals + change.removals,
    created: acc.created + (change.kind === 'created' ? 1 : 0),
    modified: acc.modified + (change.kind === 'modified' ? 1 : 0),
    deleted: acc.deleted + (change.kind === 'deleted' ? 1 : 0),
  }), { additions: 0, removals: 0, created: 0, modified: 0, deleted: 0 })
  return {
    ...totals,
    diagnostics: changes.reduce((acc, change) => mergeDiagnosticCounts(acc, countDiagnostics(change.diagnostics)), emptyDiagnosticCounts()),
  }
}

function DiagnosticPills({ counts }: { counts: DiagnosticCounts; compact?: boolean }) {
  if (counts.total === 0) return null
  const items = [
    counts.errors > 0 ? { tone: 'error' as const, label: counts.errors + ' error' + plural(counts.errors), icon: AlertCircle } : null,
    counts.warnings > 0 ? { tone: 'warning' as const, label: counts.warnings + ' warning' + plural(counts.warnings), icon: AlertTriangle } : null,
    counts.infos > 0 ? { tone: 'info' as const, label: counts.infos + ' info', icon: Info } : null,
  ].filter((item): item is { tone: DiagnosticTone; label: string; icon: typeof AlertCircle } => Boolean(item))
  return (
    <span className="current-turn-change-diagnostics" aria-label={counts.total.toLocaleString() + ' diagnostics'}>
      {items.map((item) => {
        const Icon = item.icon
        return (
          <em className={'change-diagnostic change-diagnostic-' + item.tone} key={item.tone}>
            <Icon size={11} aria-hidden="true" />
            {item.label}
          </em>
        )
      })}
    </span>
  )
}

function ChangeDiagnostics({ diagnostics }: { diagnostics: DiagnosticEntry[] }) {
  if (diagnostics.length === 0) return null
  return (
    <div className="current-turn-change-diagnostic-list" aria-label="Diagnostics for changed file">
      {diagnostics.map((diagnostic, index) => {
        const tone = diagnosticTone(diagnostic.severity)
        const Icon = tone === 'error' ? AlertCircle : tone === 'warning' ? AlertTriangle : Info
        return (
          <div className={'current-turn-change-diagnostic current-turn-change-diagnostic-' + tone} key={index}>
            <span>
              <Icon size={12} aria-hidden="true" />
              <strong>{diagnostic.severity}</strong>
            </span>
            <code>{diagnostic.line}:{diagnostic.column}</code>
            <span>
              <strong>{diagnostic.source}{diagnostic.code ? ' [' + diagnostic.code + ']' : ''}</strong>
              <em>{diagnostic.message}</em>
            </span>
          </div>
        )
      })}
    </div>
  )
}

function flattenDiagnostics(blocks: DiagnosticsChatBlock[]): Array<{ path: string; diagnostics: DiagnosticEntry[] }> {
  const files: Array<{ path: string; diagnostics: DiagnosticEntry[] }> = []
  for (const block of blocks) {
    for (const file of block.files) {
      if (file.diagnostics.length === 0) continue
      files.push({ path: file.path, diagnostics: file.diagnostics })
    }
  }
  return files
}

function diagnosticsForPath(path: string, files: Array<{ path: string; diagnostics: DiagnosticEntry[] }>): DiagnosticEntry[] {
  const normalizedPath = normalizeComparablePath(path)
  const matches: DiagnosticEntry[] = []
  for (const file of files) {
    const candidate = normalizeComparablePath(file.path)
    if (candidate === normalizedPath || candidate.endsWith('/' + normalizedPath) || normalizedPath.endsWith('/' + candidate)) {
      matches.push(...file.diagnostics)
    }
  }
  return matches
}

function countDiagnostics(diagnostics: DiagnosticEntry[]): DiagnosticCounts {
  const counts = emptyDiagnosticCounts()
  for (const diagnostic of diagnostics) {
    counts.total += 1
    const tone = diagnosticTone(diagnostic.severity)
    if (tone === 'error') counts.errors += 1
    else if (tone === 'warning') counts.warnings += 1
    else counts.infos += 1
  }
  return counts
}

function mergeDiagnosticCounts(left: DiagnosticCounts, right: DiagnosticCounts): DiagnosticCounts {
  return {
    total: left.total + right.total,
    errors: left.errors + right.errors,
    warnings: left.warnings + right.warnings,
    infos: left.infos + right.infos,
  }
}

function emptyDiagnosticCounts(): DiagnosticCounts {
  return { total: 0, errors: 0, warnings: 0, infos: 0 }
}

function diagnosticTone(severity: string): DiagnosticTone {
  const normalized = severity.toLowerCase()
  if (normalized === 'error') return 'error'
  if (normalized === 'warning') return 'warning'
  return 'info'
}

function plural(value: number): string {
  return value === 1 ? '' : 's'
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

function parseUnifiedDiffHeader(diff: string): { path: string | null; kind: FileChangeKind } {
  let oldPath: string | null = null
  let newPath: string | null = null
  for (const line of diff.split('\n')) {
    if (line.startsWith('--- ')) oldPath = normalizeDiffPath(line.slice(4))
    if (line.startsWith('+++ ')) newPath = normalizeDiffPath(line.slice(4))
    if (oldPath !== null && newPath !== null) break
  }
  const kind: FileChangeKind = oldPath === null && newPath ? 'created' : newPath === null && oldPath ? 'deleted' : 'modified'
  return { path: newPath || oldPath, kind }
}

function normalizeDiffPath(value: string): string | null {
  const trimmed = value.trim().split(/\t/, 1)[0] || ''
  if (!trimmed || trimmed === '/dev/null') return null
  return trimmed.replace(/^a\//, '').replace(/^b\//, '') || null
}

function normalizeComparablePath(value: string): string {
  return value.trim().replace(/\\/g, '/').replace(/^a\//, '').replace(/^b\//, '').replace(/^\.\//, '').replace(/\/+$/g, '')
}
