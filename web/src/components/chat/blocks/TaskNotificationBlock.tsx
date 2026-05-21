import { AlertTriangle, CheckCircle2, Clock3, XCircle } from 'lucide-react'
import type { TaskNotificationBlock as TaskNotification } from '../../../chat-rendering'

type Props = {
  block: TaskNotification
  onOpenTask?: (taskId: string) => void
}

export function TaskNotificationBlock({ block, onOpenTask }: Props) {
  const Icon = iconForStatus(block.status)
  return (
    <article className={'chat-block task-notification task-notification-' + toneForStatus(block.status)}>
      <header>
        <span className="task-notification-icon" aria-hidden="true"><Icon size={15} /></span>
        <div>
          <strong>Subagent {block.status}</strong>
          {block.subagentType ? <small>{block.subagentType}</small> : null}
        </div>
        <span>{block.taskId}</span>
      </header>
      <div className="task-notification-body">
        {block.summary ? <p>{block.summary}</p> : null}
        {block.error ? <p className="task-notification-error">{block.error}</p> : null}
        <div>
          {typeof block.durationSeconds === 'number' ? <span>{block.durationSeconds.toFixed(1)}s</span> : null}
          {block.outputFile ? <span>{block.outputFile}</span> : null}
        </div>
        {onOpenTask ? (
          <button type="button" onClick={() => onOpenTask(block.taskId)}>
            Open task details
          </button>
        ) : null}
      </div>
    </article>
  )
}

function iconForStatus(status: string) {
  if (status === 'completed') return CheckCircle2
  if (status === 'failed' || status === 'killed') return XCircle
  if (status === 'interrupted' || status === 'stopped') return AlertTriangle
  return Clock3
}

function toneForStatus(status: string): string {
  if (status === 'completed') return 'complete'
  if (status === 'failed' || status === 'killed') return 'error'
  if (status === 'interrupted' || status === 'stopped') return 'warn'
  return 'active'
}
