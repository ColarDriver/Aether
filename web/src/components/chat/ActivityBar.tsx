import { Brain, Loader2, MessageSquareText, Wrench } from 'lucide-react'
import type { RunStatusSnapshot, TokenUsage } from '../../chat-rendering'
import { tokenUsageSummary } from '../../chat-rendering'

type Props = {
  activeRunId: string | null
  status?: RunStatusSnapshot
  tokens?: TokenUsage
  sessionId?: string | null
  model?: string | null
  verbose?: boolean
}

export function ActivityBar({ activeRunId, status, tokens, sessionId, model, verbose = false }: Props) {
  if (!activeRunId) return null
  const state = status?.state || 'thinking'
  const label = activityLabel(state)
  const detail = friendlyDetail(status?.detail)
  const tokenSummary = tokenUsageSummary(tokens ?? status?.tokens)
  const Icon = activityIcon(state)
  const verboseMeta = verbose
    ? [
      activeRunId ? 'run ' + activeRunId.slice(0, 8) : null,
      status?.elapsedMs ? formatDuration(status.elapsedMs) : null,
      state || null,
    ]
    : []
  const meta = [
    tokenSummary,
    ...verboseMeta,
    sessionId ? sessionId.slice(0, 8) : null,
    model || null,
  ].filter((item): item is string => Boolean(item))

  return (
    <div className={'web-activity-bar web-activity-' + activityTone(state)} role="status" aria-label="Current activity">
      <span className="web-activity-icon" aria-hidden="true"><Icon size={14} /></span>
      <strong className="aether-shimmer-text">{label}</strong>
      {detail ? <span className="web-activity-detail">{detail}</span> : null}
      {meta.length > 0 ? <span className="web-activity-meta">{meta.join(' · ')}</span> : null}
    </div>
  )
}

function activityLabel(state: string): string {
  if (state === 'thinking') return 'Thinking'
  if (state === 'responding') return 'Responding'
  if (state === 'tool_use') return 'Running tool'
  if (state === 'starting') return 'Starting'
  if (state === 'cancelling') return 'Cancelling'
  return state || 'Working'
}

function friendlyDetail(detail?: string | null): string | null {
  if (!detail) return null
  if (/^[A-Z][A-Z0-9_]*$/.test(detail)) return null
  return detail.replace(/_/g, ' ')
}

function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '0ms'
  if (ms < 1000) return Math.round(ms) + 'ms'
  return (ms / 1000).toFixed(2).replace(/\.00$/, '') + 's'
}

function activityIcon(state: string) {
  if (state === 'thinking') return Brain
  if (state === 'responding') return MessageSquareText
  if (state === 'tool_use') return Wrench
  return Loader2
}

function activityTone(state: string): string {
  if (state === 'thinking') return 'thinking'
  if (state === 'responding') return 'responding'
  if (state === 'tool_use') return 'tool'
  if (state === 'cancelling') return 'working'
  return 'working'
}

