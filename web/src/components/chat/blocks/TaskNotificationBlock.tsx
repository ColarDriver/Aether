import { AlertTriangle, CheckCircle2, Clock3, XCircle } from 'lucide-react'
import type { TaskNotificationBlock as TaskNotification } from '../../../chat-rendering'

type Props = {
  block: TaskNotification
}

export function TaskNotificationBlock({ block }: Props) {
  const Icon = iconForStatus(block.status)
  return (
    <article className={'chat-block task-notification task-notification-' + toneForStatus(block.status)}>
      <header>
        <Icon size={15} aria-hidden="true" />
        <strong>Subagent {block.status}</strong>
        <span>{block.taskId}</span>
      </header>
      <div className="task-notification-body">
        {block.summary ? <p>{block.summary}</p> : null}
        {block.error ? <p className="task-notification-error">{block.error}</p> : null}
        <div>
          {block.subagentType ? <span>{block.subagentType}</span> : null}
          {typeof block.durationSeconds === 'number' ? <span>{block.durationSeconds.toFixed(1)}s</span> : null}
          {block.outputFile ? <span>{block.outputFile}</span> : null}
        </div>
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
