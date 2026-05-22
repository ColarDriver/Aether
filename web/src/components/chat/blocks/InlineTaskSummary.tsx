import { AlertTriangle, Bot, CheckCircle2, Clock3, FileText, Hash, XCircle } from 'lucide-react'

export type InlineTaskSummaryProps = {
  title: string
  status: string
  taskId?: string | null
  subagentType?: string | null
  model?: string | null
  durationSeconds?: number | null
  inputTokens?: number | null
  outputTokens?: number | null
  summary?: string | null
  error?: string | null
  outputTail?: string | null
  outputFile?: string | null
  onOpenTask?: (taskId: string) => void
  className?: string
  ariaLabel?: string
}

export function InlineTaskSummary({
  title,
  status,
  taskId,
  subagentType,
  model,
  durationSeconds,
  inputTokens,
  outputTokens,
  summary,
  error,
  outputTail,
  outputFile,
  onOpenTask,
  className = '',
  ariaLabel = 'Subagent task summary',
}: InlineTaskSummaryProps) {
  const Icon = iconForStatus(status)
  const tone = toneForStatus(status)
  const normalizedStatus = status || 'unknown'
  const stats = taskStats({ durationSeconds, inputTokens, outputTokens, model, outputFile })
  return (
    <section className={'inline-task-summary inline-task-summary-' + tone + (className ? ' ' + className : '')} aria-label={ariaLabel}>
      <header className="inline-task-summary-header">
        <span className="inline-task-summary-icon" aria-hidden="true"><Icon size={16} /></span>
        <span className="inline-task-summary-title">
          <strong>{title || 'Subagent task'}</strong>
          <small>{[subagentType, model].filter(Boolean).join(' / ') || 'subagent'}</small>
        </span>
        <em>{normalizedStatus}</em>
      </header>
      {taskId ? (
        <div className="inline-task-summary-task-id">
          <Hash size={12} aria-hidden="true" />
          <code>{taskId}</code>
        </div>
      ) : null}
      {stats.length > 0 ? (
        <div className="inline-task-summary-stats">
          {stats.map((stat) => (
            <span key={stat.label} title={stat.title || stat.label}>
              {stat.icon === 'file' ? <FileText size={12} aria-hidden="true" /> : null}
              <strong>{stat.value}</strong>
              <small>{stat.label}</small>
            </span>
          ))}
        </div>
      ) : null}
      {summary ? <p className="inline-task-summary-text">{summary}</p> : null}
      {error ? <p className="inline-task-summary-error">{error}</p> : null}
      {outputTail ? <pre className="inline-task-summary-output"><code>{outputTail}</code></pre> : null}
      {onOpenTask && taskId ? (
        <button type="button" className="inline-task-summary-action" onClick={() => onOpenTask(taskId)}>
          Open task details
        </button>
      ) : null}
    </section>
  )
}

type Stat = {
  label: string
  value: string
  title?: string
  icon?: 'file'
}

function taskStats({
  durationSeconds,
  inputTokens,
  outputTokens,
  model,
  outputFile,
}: Pick<InlineTaskSummaryProps, 'durationSeconds' | 'inputTokens' | 'outputTokens' | 'model' | 'outputFile'>): Stat[] {
  const stats: Stat[] = []
  if (typeof durationSeconds === 'number' && Number.isFinite(durationSeconds)) {
    stats.push({ label: 'duration', value: formatDuration(durationSeconds) })
  }
  if (model) stats.push({ label: 'model', value: model })
  const totalTokens = Math.max(0, inputTokens ?? 0) + Math.max(0, outputTokens ?? 0)
  if (totalTokens > 0) {
    stats.push({
      label: 'tokens',
      value: totalTokens.toLocaleString(),
      title: (inputTokens ?? 0).toLocaleString() + ' input / ' + (outputTokens ?? 0).toLocaleString() + ' output',
    })
  }
  if (outputFile) stats.push({ label: 'result', value: outputFile, icon: 'file' })
  return stats
}

function formatDuration(seconds: number): string {
  if (seconds < 10) return seconds.toFixed(1) + 's'
  if (seconds < 60) return Math.round(seconds) + 's'
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  return minutes + 'm ' + rest + 's'
}

function iconForStatus(status: string) {
  const normalized = status.toLowerCase()
  if (normalized === 'completed' || normalized === 'finished' || normalized === 'success') return CheckCircle2
  if (normalized === 'failed' || normalized === 'killed' || normalized === 'error') return XCircle
  if (normalized === 'interrupted' || normalized === 'stopped' || normalized === 'cancelled') return AlertTriangle
  if (normalized === 'running' || normalized === 'pending') return Clock3
  return Bot
}

function toneForStatus(status: string): string {
  const normalized = status.toLowerCase()
  if (normalized === 'completed' || normalized === 'finished' || normalized === 'success') return 'complete'
  if (normalized === 'failed' || normalized === 'killed' || normalized === 'error') return 'error'
  if (normalized === 'interrupted' || normalized === 'stopped' || normalized === 'cancelled') return 'warn'
  if (normalized === 'running' || normalized === 'pending') return 'active'
  return 'neutral'
}
