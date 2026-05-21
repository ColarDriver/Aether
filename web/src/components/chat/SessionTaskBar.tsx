import { AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, Circle, Clock3, ListTodo, XCircle } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { TaskSummary } from '../../api/types'

type Props = {
  tasks: TaskSummary[]
}

export function SessionTaskBar({ tasks }: Props) {
  const [expanded, setExpanded] = useState(false)
  const activeCount = tasks.filter((task) => !isTaskTerminal(task)).length
  const completedCount = tasks.filter((task) => task.status === 'completed').length
  const hasActiveTasks = activeCount > 0
  const sortedTasks = useMemo(() => [...tasks].sort((a, b) => b.started_at - a.started_at), [tasks])

  useEffect(() => {
    if (hasActiveTasks) setExpanded(true)
  }, [hasActiveTasks])

  if (tasks.length === 0) return null

  const progress = tasks.length > 0 ? Math.round((completedCount / tasks.length) * 100) : 0

  return (
    <section className="session-task-bar" aria-label="Session tasks">
      <button type="button" className="session-task-bar-header" onClick={() => setExpanded((value) => !value)}>
        <span className="session-task-icon">
          <ListTodo size={16} aria-hidden="true" />
        </span>
        <strong>Subagent tasks</strong>
        <span className="session-task-progress" aria-hidden="true">
          <span style={{ width: progress + '%' }} />
        </span>
        <span className="session-task-count">
          {completedCount}/{tasks.length}
          {activeCount > 0 ? ' · ' + activeCount + ' active' : ''}
        </span>
        {expanded ? <ChevronUp size={15} aria-hidden="true" /> : <ChevronDown size={15} aria-hidden="true" />}
      </button>
      {expanded ? (
        <ol className="session-task-list">
          {sortedTasks.map((task) => (
            <TaskItem key={task.task_id} task={task} />
          ))}
        </ol>
      ) : null}
    </section>
  )
}

export function isTaskTerminal(task: Pick<TaskSummary, 'status'>): boolean {
  return task.status === 'completed' || task.status === 'failed' || task.status === 'interrupted' || task.status === 'killed'
}

function TaskItem({ task }: { task: TaskSummary }) {
  const Icon = iconForTask(task)
  const tone = toneForTask(task)
  const detail = task.error || task.summary || activityDetail(task)
  return (
    <li className={'session-task-item session-task-' + tone}>
      <Icon size={15} aria-hidden="true" />
      <div>
        <div className="session-task-main">
          <span>{task.prompt}</span>
          <em>{task.subagent_type}</em>
        </div>
        {detail ? <p>{detail}</p> : null}
      </div>
      <span className="session-task-meta">
        {task.status}
        {task.model ? ' · ' + task.model : ''}
      </span>
    </li>
  )
}

function iconForTask(task: TaskSummary) {
  if (task.status === 'completed') return CheckCircle2
  if (task.status === 'failed' || task.status === 'killed') return XCircle
  if (task.status === 'interrupted') return AlertTriangle
  if (task.status === 'running') return Clock3
  return Circle
}

function toneForTask(task: TaskSummary): string {
  if (task.status === 'completed') return 'complete'
  if (task.status === 'failed' || task.status === 'killed') return 'error'
  if (task.status === 'interrupted') return 'warn'
  if (task.status === 'running') return 'active'
  return 'pending'
}

function activityDetail(task: TaskSummary): string {
  const parts: string[] = []
  if (task.background) parts.push('background')
  if (task.iterations > 0) parts.push(task.iterations + ' iterations')
  if (task.tool_use_count > 0) parts.push(task.tool_use_count + ' tool calls')
  const tokenTotal = task.input_tokens + task.output_tokens
  if (tokenTotal > 0) parts.push(tokenTotal + ' tokens')
  return parts.join(' · ')
}
