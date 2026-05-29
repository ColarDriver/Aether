import { AlertCircle, AlertTriangle, Check, ChevronDown, ChevronRight, Eye, FileCode2, Info, RotateCcw, ShieldCheck, XCircle } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { ApiError, api } from '../../../api/client'
import type { SessionCheckpointActionBody, SessionTurnCheckpoint, WorkspaceChange, WorkspaceChangeVerificationResult } from '../../../api/types'
import type { ChatAttachment, DiagnosticEntry, DiagnosticsBlock as DiagnosticsChatBlock, DiffBlock as DiffChatBlock } from '../../../chat-rendering'
import { ConfirmDialog } from '../../shared/ConfirmDialog'
import { CopyButton } from '../../shared/CopyButton'
import { DiffViewer, parseUnifiedDiff } from '../DiffViewer'

type Props = {
  diffs: DiffChatBlock[]
  diagnostics?: DiagnosticsChatBlock[]
  verifications?: CurrentTurnVerification[]
  checkpoint?: CurrentTurnCheckpoint | null
  sessionId?: string | null
  serverCheckpoint?: SessionTurnCheckpoint | null
  undoAction?: CurrentTurnUndoAction | null
  undoDisabled?: boolean
  onAcceptFile?: (change: CurrentTurnFileChangeAction) => Promise<void> | void
  onRevertFile?: (change: CurrentTurnFileChangeAction) => Promise<void> | void
  onOpenFile?: (path: string) => void
  onUndoTurn?: (action: CurrentTurnUndoAction) => void
}

export type FileChangeKind = 'created' | 'deleted' | 'modified'
type DiagnosticTone = 'error' | 'warning' | 'info'
type ChangeResolution = 'accepting' | 'accepted' | 'reverting' | 'reverted' | 'conflict' | 'error'
type ServerDiffState = {
  loading?: boolean
  diff?: string | null
  error?: string | null
}

export type CurrentTurnFileChangeAction = {
  path: string
  kind: FileChangeKind
  diff: string
  oldText?: string | null
  newText?: string | null
  checkpointId?: string | null
  checkpointFiles?: string[]
  currentHash?: string | null
}

export type CurrentTurnCheckpoint = {
  checkpointId: string
  label?: string | null
  files?: string[]
}

export type CurrentTurnUndoAction = {
  body: SessionCheckpointActionBody
  promptContent: string
  attachments?: ChatAttachment[]
  checkpointId?: string | null
  paths: string[]
}

export type CurrentTurnVerification = {
  id: string
  toolName: string
  label: string
  command?: string | null
  status: 'passed' | 'failed' | 'warning'
  exitCode?: number | null
  durationMs?: number | null
  summary?: string | null
}

type FileChange = {
  path: string
  diff: string
  kind: FileChangeKind
  additions: number
  removals: number
  hunks: number
  diagnostics: DiagnosticEntry[]
  oldText?: string | null
  newText?: string | null
}

type DiagnosticCounts = {
  total: number
  errors: number
  warnings: number
  infos: number
}

export function CurrentTurnChangeCard({ diffs, diagnostics = [], verifications = [], checkpoint = null, sessionId = null, serverCheckpoint = null, undoAction = null, undoDisabled = false, onAcceptFile, onRevertFile, onOpenFile, onUndoTurn }: Props) {
  const localChanges = useMemo(() => summarizeDiffs(diffs, diagnostics), [diffs, diagnostics])
  const [serverDiffsByPath, setServerDiffsByPath] = useState<Record<string, ServerDiffState>>({})
  const changes = useMemo(() => {
    if (localChanges.length > 0) return localChanges
    return summarizeServerCheckpoint(serverCheckpoint, serverDiffsByPath)
  }, [localChanges, serverCheckpoint, serverDiffsByPath])
  const effectiveCheckpoint = useMemo<CurrentTurnCheckpoint | null>(() => {
    if (checkpoint) return checkpoint
    const checkpointId = serverCheckpoint?.code.checkpoint_id
    if (!checkpointId) return null
    return {
      checkpointId,
      label: null,
      files: serverCheckpoint?.code.files_changed ?? [],
    }
  }, [checkpoint, serverCheckpoint])
  const [expanded, setExpanded] = useState(false)
  const [resolutionByPath, setResolutionByPath] = useState<Record<string, ChangeResolution>>({})
  const [errorByPath, setErrorByPath] = useState<Record<string, string>>({})
  const [workspaceChangeByPath, setWorkspaceChangeByPath] = useState<Record<string, WorkspaceChange>>({})
  const [verificationByPath, setVerificationByPath] = useState<Record<string, CurrentTurnVerification>>({})
  const [verifyingByPath, setVerifyingByPath] = useState<Record<string, boolean>>({})
  const [pendingRevertChange, setPendingRevertChange] = useState<FileChange | null>(null)
  const usingServerCheckpoint = localChanges.length === 0 && hasServerCheckpointEvidence(serverCheckpoint)
  const shouldTrackWorkspaceChanges = Boolean(onAcceptFile || onRevertFile)
  const changePathsKey = useMemo(() => changes.map((change) => normalizeComparablePath(change.path)).sort().join('\n'), [changes])
  const totals = summarizeChanges(changes)
  const displayedTotals = usingServerCheckpoint && serverCheckpoint
    ? {
        ...totals,
        additions: totals.additions || serverCheckpoint.code.insertions,
        removals: totals.removals || serverCheckpoint.code.deletions,
      }
    : totals
  const allVerifications = [...verifications, ...Object.values(verificationByPath)]
  const verificationTotals = summarizeVerifications(allVerifications)
  useEffect(() => {
    if (!expanded || !usingServerCheckpoint || !serverCheckpoint || !sessionId) return
    for (const path of serverCheckpoint.code.files_changed) {
      const existing = serverDiffsByPath[path]
      if (existing?.loading || existing?.diff !== undefined || existing?.error) continue
      setServerDiffsByPath((current) => ({ ...current, [path]: { loading: true } }))
      void api.sessionTurnCheckpointDiff(sessionId, {
        path,
        target_user_message_id: serverCheckpoint.target.target_user_message_id,
        user_message_index: serverCheckpoint.target.user_message_index,
      })
        .then((result) => {
          setServerDiffsByPath((current) => ({
            ...current,
            [path]: result.state === 'ok'
              ? { diff: result.diff ?? '' }
              : { error: result.error || 'Diff unavailable for this file.' },
          }))
        })
        .catch((error) => {
          setServerDiffsByPath((current) => ({
            ...current,
            [path]: { error: error instanceof Error ? error.message : String(error) },
          }))
      })
    }
  }, [expanded, serverCheckpoint, serverDiffsByPath, sessionId, usingServerCheckpoint])
  const refreshWorkspaceChanges = useCallback(async () => {
    if (!shouldTrackWorkspaceChanges || !changePathsKey) {
      setWorkspaceChangeByPath({})
      return
    }
    try {
      const result = await api.workspaceChanges()
      const next: Record<string, WorkspaceChange> = {}
      for (const change of result.changes ?? []) {
        next[normalizeComparablePath(change.path)] = change
      }
      setWorkspaceChangeByPath(next)
    } catch {
      setWorkspaceChangeByPath({})
    }
  }, [changePathsKey, shouldTrackWorkspaceChanges])

  useEffect(() => {
    void refreshWorkspaceChanges()
  }, [refreshWorkspaceChanges])

  if (changes.length === 0) return null
  const actionForChange = (change: FileChange): CurrentTurnFileChangeAction => ({
    ...change,
    ...(effectiveCheckpoint?.checkpointId ? { checkpointId: effectiveCheckpoint.checkpointId, checkpointFiles: effectiveCheckpoint.files ?? [] } : {}),
    currentHash: workspaceChangeForPath(workspaceChangeByPath, change.path)?.current_hash ?? null,
  })
  const acceptChange = (change: FileChange) => {
    if (resolutionByPath[change.path] === 'accepting') return
    setResolutionByPath((current) => ({ ...current, [change.path]: onAcceptFile ? 'accepting' : 'accepted' }))
    setErrorByPath((current) => {
      const next = { ...current }
      delete next[change.path]
      return next
    })
    if (!onAcceptFile) return
    Promise.resolve(onAcceptFile(actionForChange(change)))
      .then(() => {
        setResolutionByPath((current) => ({ ...current, [change.path]: 'accepted' }))
        void refreshWorkspaceChanges()
      })
      .catch((error: unknown) => {
        setResolutionByPath((current) => ({ ...current, [change.path]: resolutionFromError(error) }))
        setErrorByPath((current) => ({
          ...current,
          [change.path]: changeActionErrorMessage(error),
        }))
      })
  }
  const revertChange = (change: FileChange) => {
    if (!onRevertFile || (!effectiveCheckpoint?.checkpointId && change.oldText == null) || resolutionByPath[change.path] === 'reverting') return
    setResolutionByPath((current) => ({ ...current, [change.path]: 'reverting' }))
    setErrorByPath((current) => {
      const next = { ...current }
      delete next[change.path]
      return next
    })
    Promise.resolve(onRevertFile(actionForChange(change)))
      .then(() => {
        setResolutionByPath((current) => ({ ...current, [change.path]: 'reverted' }))
        void refreshWorkspaceChanges()
      })
      .catch((error: unknown) => {
        setResolutionByPath((current) => ({ ...current, [change.path]: resolutionFromError(error) }))
        setErrorByPath((current) => ({
          ...current,
          [change.path]: changeActionErrorMessage(error),
        }))
      })
  }
  const requestRevertChange = (change: FileChange) => {
    const canRevert = Boolean(onRevertFile && (change.oldText != null || effectiveCheckpoint?.checkpointId))
    if (!canRevert || resolutionByPath[change.path] === 'reverting' || resolutionByPath[change.path] === 'reverted') return
    setPendingRevertChange(change)
  }

  const confirmRevertChange = () => {
    if (!pendingRevertChange) return
    const change = pendingRevertChange
    setPendingRevertChange(null)
    revertChange(change)
  }

  const verifyChange = (change: FileChange) => {
    if (verifyingByPath[change.path]) return
    setVerifyingByPath((current) => ({ ...current, [change.path]: true }))
    setErrorByPath((current) => {
      const next = { ...current }
      delete next[change.path]
      return next
    })
    void api.verifyWorkspaceChanges({ paths: [change.path] })
      .then((result) => {
        setVerificationByPath((current) => ({
          ...current,
          [change.path]: verificationFromWorkspaceResult(change.path, result),
        }))
      })
      .catch((error: unknown) => {
        setErrorByPath((current) => ({
          ...current,
          [change.path]: 'Verification failed: ' + (error instanceof Error ? error.message : String(error)),
        }))
      })
      .finally(() => {
        setVerifyingByPath((current) => {
          const next = { ...current }
          delete next[change.path]
          return next
        })
      })
  }

  return (
    <>
    <section className="current-turn-change-card" aria-label="Changed files">
      <div className="current-turn-change-header">
        <button
          type="button"
          className="current-turn-change-toggle"
          aria-expanded={expanded}
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <FileCode2 size={15} />
          <strong>{changes.length} changed {changes.length === 1 ? 'file' : 'files'}</strong>
          <span className="current-turn-change-kind-summary" aria-label="File change summary">
            {displayedTotals.created > 0 ? <em className="change-created">{displayedTotals.created} created</em> : null}
            {displayedTotals.modified > 0 ? <em>{displayedTotals.modified} modified</em> : null}
            {displayedTotals.deleted > 0 ? <em className="change-deleted">{displayedTotals.deleted} deleted</em> : null}
          </span>
          <DiagnosticPills counts={displayedTotals.diagnostics} compact />
          {allVerifications.length > 0 ? <VerificationPill totals={verificationTotals} /> : null}
          {effectiveCheckpoint?.checkpointId ? <span className="current-turn-change-checkpoint">checkpoint {effectiveCheckpoint.checkpointId.slice(0, 8)}</span> : null}
          {hasChangeStats(displayedTotals) ? (
            <span className="current-turn-change-stats">
              <em className="change-add">+{displayedTotals.additions}</em>
              <em className="change-remove">-{displayedTotals.removals}</em>
            </span>
          ) : null}
        </button>
        {undoAction && onUndoTurn ? (
          <button
            type="button"
            className="current-turn-change-undo"
            disabled={undoDisabled}
            onClick={() => onUndoTurn(undoAction)}
          >
            <RotateCcw size={12} aria-hidden="true" />
            Undo turn
          </button>
        ) : null}
      </div>
      <div className="current-turn-change-files">
        {changes.map((change) => {
          const diagnosticCounts = countDiagnostics(change.diagnostics)
          const workspaceChange = workspaceChangeForPath(workspaceChangeByPath, change.path)
          const resolution = resolutionByPath[change.path] ?? (workspaceChange?.accepted ? 'accepted' : undefined)
          const canRevert = Boolean(onRevertFile && (change.oldText != null || effectiveCheckpoint?.checkpointId))
          const showFileStats = change.hunks > 0 || hasChangeStats(change) || (!usingServerCheckpoint && change.diff.trim().length > 0)
          return (
            <div className={'current-turn-change-file current-turn-change-file-' + change.kind} key={change.path}>
              <span className="current-turn-change-path">{change.path}</span>
              <em className={'current-turn-change-kind change-' + change.kind}>{change.kind}</em>
              {resolution ? <em className={'current-turn-change-resolution current-turn-change-resolution-' + resolution}>{resolutionLabel(resolution)}</em> : null}
              <DiagnosticPills counts={diagnosticCounts} compact />
              {showFileStats ? (
                <span className="current-turn-change-file-stats">
                  {change.hunks > 0 ? <em>{change.hunks} hunk{change.hunks === 1 ? '' : 's'}</em> : null}
                  <em className="change-add">+{change.additions}</em>
                  <em className="change-remove">-{change.removals}</em>
                </span>
              ) : null}
              <div className="current-turn-change-actions">
                {onOpenFile ? (
                  <button type="button" aria-label={'Open ' + change.path + ' in workspace preview'} onClick={() => onOpenFile(change.path)}>
                    <Eye size={12} aria-hidden="true" />
                    Open
                  </button>
                ) : null}
                <button type="button" onClick={() => verifyChange(change)} disabled={Boolean(verifyingByPath[change.path])}>
                  <ShieldCheck size={12} aria-hidden="true" />
                  {verifyingByPath[change.path] ? 'Verifying' : 'Verify'}
                </button>
                <button type="button" onClick={() => acceptChange(change)} disabled={resolution === 'accepting' || resolution === 'accepted' || resolution === 'reverted'}>
                  <Check size={12} aria-hidden="true" />
                  {resolution === 'accepting' ? 'Accepting' : 'Accept'}
                </button>
                {canRevert ? (
                  <button type="button" onClick={() => requestRevertChange(change)} disabled={resolution === 'reverting' || resolution === 'reverted'}>
                    <RotateCcw size={12} aria-hidden="true" />
                    {resolution === 'reverting' ? 'Reverting' : 'Revert'}
                  </button>
                ) : null}
              </div>
              <CopyButton
                text={change.path}
                label={'Copy ' + change.path}
                displayLabel="Copy"
                displayCopiedLabel="Copied"
                className="current-turn-change-copy"
              />
              {errorByPath[change.path] ? <span className="current-turn-change-error">{errorByPath[change.path]}</span> : null}
            </div>
          )
        })}
      </div>
      <VerificationBundle verifications={allVerifications} />
      {expanded ? (
        <div className="current-turn-change-diffs">
          {changes.map((change) => (
            <article key={change.path} className="current-turn-change-diff">
              <header>
                <span>{change.path}</span>
                <span className="current-turn-change-diff-meta">
                  <DiagnosticPills counts={countDiagnostics(change.diagnostics)} />
                  {resolutionByPath[change.path] ? <em className={'current-turn-change-resolution current-turn-change-resolution-' + resolutionByPath[change.path]}>{resolutionLabel(resolutionByPath[change.path]!)}</em> : null}
                  <em className={'change-' + change.kind}>{change.kind}</em>
                </span>
              </header>
              {usingServerCheckpoint ? (
                <ServerCheckpointDiff path={change.path} state={serverDiffsByPath[change.path]} diff={change.diff} />
              ) : (
                <DiffViewer diff={change.diff} />
              )}
              <ChangeDiagnostics diagnostics={change.diagnostics} />
            </article>
          ))}
        </div>
      ) : null}
    </section>
    {pendingRevertChange ? (
      <ConfirmDialog
        title="Revert file change"
        description={revertDescription(pendingRevertChange, effectiveCheckpoint)}
        confirmLabel="Revert file"
        cancelLabel="Keep change"
        onCancel={() => setPendingRevertChange(null)}
        onConfirm={confirmRevertChange}
      />
    ) : null}
    </>
  )
}

function revertDescription(change: FileChange, checkpoint: CurrentTurnCheckpoint | null): string {
  const base = 'Restore `' + change.path + '` and discard the current workspace content for that file.'
  if (checkpoint?.checkpointId) return base + ' The restore will use checkpoint `' + checkpoint.checkpointId.slice(0, 8) + '` when the backend can verify it is safe.'
  return base + ' The backend will refuse the restore if the current file no longer matches the captured change state.'
}

function VerificationPill({ totals }: { totals: VerificationTotals }) {
  const failed = totals.failed > 0
  const warning = !failed && totals.warning > 0
  const Icon = failed ? XCircle : ShieldCheck
  const label = failed
    ? totals.failed + ' failed'
    : warning
      ? totals.warning + ' warning' + plural(totals.warning)
      : totals.passed + ' passed'
  return (
    <span className={'current-turn-change-verification-pill current-turn-change-verification-pill-' + (failed ? 'failed' : warning ? 'warning' : 'passed')}>
      <Icon size={11} aria-hidden="true" />
      {label}
    </span>
  )
}

function VerificationBundle({ verifications }: { verifications: CurrentTurnVerification[] }) {
  if (verifications.length === 0) return null
  return (
    <section className="current-turn-verification-bundle" aria-label="Post-edit verification">
      <header>
        <span><ShieldCheck size={13} aria-hidden="true" /> Verification</span>
        <em>{verifications.length} check{verifications.length === 1 ? '' : 's'}</em>
      </header>
      <div>
        {verifications.map((verification) => {
          const failed = verification.status === 'failed'
          const Icon = failed ? XCircle : ShieldCheck
          return (
            <article className={'current-turn-verification current-turn-verification-' + verification.status} key={verification.id}>
              <span>
                <Icon size={13} aria-hidden="true" />
                <strong>{verification.label}</strong>
              </span>
              <em>{verification.status}</em>
              {verification.command ? <code>{verification.command}</code> : null}
              <small>{verificationMeta(verification).join(' · ')}</small>
              {verification.summary ? <p>{verification.summary}</p> : null}
            </article>
          )
        })}
      </div>
    </section>
  )
}

function verificationFromWorkspaceResult(path: string, result: WorkspaceChangeVerificationResult): CurrentTurnVerification {
  const status = verificationStatusFromWorkspaceResult(result)
  const command = result.command?.join(' ') ?? null
  const summary = firstVerificationLine(result.message, result.stderr, result.stdout)
    ?? (status === 'passed' ? 'Verification passed for ' + path + '.' : 'Verification failed for ' + path + '.')
  return {
    id: 'workspace-verification-' + path,
    toolName: 'workspace.verify',
    label: 'Verify ' + shortFileName(path),
    command,
    status,
    exitCode: result.exit_code ?? null,
    durationMs: null,
    summary,
  }
}

function verificationStatusFromWorkspaceResult(result: WorkspaceChangeVerificationResult): CurrentTurnVerification['status'] {
  const normalized = result.status.toLowerCase()
  if (normalized === 'warning') return 'warning'
  if (normalized === 'passed' || normalized === 'ok' || result.exit_code === 0) return 'passed'
  return 'failed'
}

function firstVerificationLine(...values: Array<string | null | undefined>): string | null {
  for (const value of values) {
    const line = value?.split(/\r?\n/).map((item) => item.trim()).find(Boolean)
    if (line) return line.length > 180 ? line.slice(0, 177) + '...' : line
  }
  return null
}

function shortFileName(path: string): string {
  const parts = path.split(/[\/]+/).filter(Boolean)
  return parts.at(-1) || path
}

type VerificationTotals = {
  passed: number
  failed: number
  warning: number
}

function summarizeVerifications(verifications: CurrentTurnVerification[]): VerificationTotals {
  return verifications.reduce((acc, verification) => ({
    passed: acc.passed + (verification.status === 'passed' ? 1 : 0),
    failed: acc.failed + (verification.status === 'failed' ? 1 : 0),
    warning: acc.warning + (verification.status === 'warning' ? 1 : 0),
  }), { passed: 0, failed: 0, warning: 0 })
}

function verificationMeta(verification: CurrentTurnVerification): string[] {
  return [
    verification.toolName,
    verification.exitCode != null ? 'exit ' + verification.exitCode : null,
    verification.durationMs != null ? formatDurationMs(verification.durationMs) : null,
  ].filter((item): item is string => Boolean(item))
}

function formatDurationMs(ms: number): string {
  const seconds = Math.max(0, ms / 1000)
  if (seconds < 10) return seconds.toFixed(1) + 's'
  if (seconds < 60) return Math.round(seconds) + 's'
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  return minutes + 'm ' + rest + 's'
}

function summarizeDiffs(diffs: DiffChatBlock[], diagnostics: DiagnosticsChatBlock[]): FileChange[] {
  const byPath = new Map<string, { chunks: string[]; oldText: string | null; newText: string | null }>()
  for (const diffBlock of diffs) {
    const diff = diffBlock.diff ?? diffFromOldNew(diffBlock.oldText ?? '', diffBlock.newText ?? '')
    if (!diff.trim()) continue
    const parsedHeader = parseUnifiedDiffHeader(diff)
    const path = cleanPath(diffBlock.path) || parsedHeader.path || 'Changed file'
    const existing = byPath.get(path) ?? { chunks: [], oldText: null, newText: null }
    existing.chunks.push(diff)
    if (existing.oldText === null && diffBlock.oldText != null) existing.oldText = diffBlock.oldText
    if (existing.newText === null && diffBlock.newText != null) existing.newText = diffBlock.newText
    byPath.set(path, existing)
  }
  const diagnosticFiles = flattenDiagnostics(diagnostics)
  return Array.from(byPath.entries()).map(([path, entry]) => {
    const diff = entry.chunks.join('\n')
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
      oldText: entry.oldText,
      newText: entry.newText,
    }
  })
}

function hasServerCheckpointEvidence(checkpoint: SessionTurnCheckpoint | null): checkpoint is SessionTurnCheckpoint {
  return Boolean(
    checkpoint?.code.available
    && checkpoint.code.files_changed.length > 0
    && (checkpoint.code.insertions > 0 || checkpoint.code.deletions > 0)
  )
}

function summarizeServerCheckpoint(checkpoint: SessionTurnCheckpoint | null, diffStateByPath: Record<string, ServerDiffState>): FileChange[] {
  if (!hasServerCheckpointEvidence(checkpoint)) return []
  return checkpoint.code.files_changed.map((path) => {
    const state = diffStateByPath[path]
    const diff = state?.diff ?? ''
    const parsed = diff ? parseUnifiedDiff(diff) : []
    const header = diff ? parseUnifiedDiffHeader(diff) : { path: null, kind: 'modified' as FileChangeKind }
    return {
      path: cleanPath(header.path) || path,
      diff,
      kind: header.kind,
      additions: parsed.filter((line) => line.kind === 'add').length,
      removals: parsed.filter((line) => line.kind === 'remove').length,
      hunks: diff.split('\n').filter((line) => line.startsWith('@@')).length,
      diagnostics: [],
      oldText: null,
      newText: null,
    }
  })
}

function ServerCheckpointDiff({ path, state, diff }: { path: string; state?: ServerDiffState; diff: string }) {
  if (state?.loading) {
    return <div className="current-turn-change-diff-placeholder">Loading diff for {path}...</div>
  }
  if (state?.error) {
    return <div className="current-turn-change-error">{state.error}</div>
  }
  if (!diff.trim()) {
    return <div className="current-turn-change-diff-placeholder">Diff unavailable for {path}.</div>
  }
  return <DiffViewer diff={diff} />
}

function resolutionLabel(value: ChangeResolution): string {
  if (value === 'accepting') return 'accepting'
  if (value === 'accepted') return 'accepted'
  if (value === 'reverting') return 'reverting'
  if (value === 'reverted') return 'reverted'
  if (value === 'conflict') return 'conflict'
  return 'revert failed'
}

function resolutionFromError(error: unknown): ChangeResolution {
  return error instanceof ApiError && error.status === 409 ? 'conflict' : 'error'
}

function changeActionErrorMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 409) {
    return error.message + '. The file changed after this card was rendered; refresh the workspace diff before reverting.'
  }
  return error instanceof Error ? error.message : String(error)
}

function workspaceChangeForPath(changes: Record<string, WorkspaceChange>, path: string): WorkspaceChange | undefined {
  return changes[normalizeComparablePath(path)]
}

function hasChangeStats(value: { additions: number; removals: number }): boolean {
  return value.additions > 0 || value.removals > 0
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
