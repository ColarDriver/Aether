import { AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, Circle, Clock3, ListTodo, XCircle } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { TaskSummary } from '../../api/types'

type Props = {
  tasks: TaskSummary[]
  onOpenTask?: (task: TaskSummary) => void
}

export function SessionTaskBar({ tasks, onOpenTask }: Props) {
  const [expanded, setExpanded] = useState(false)
  const activeCount = tasks.filter((task) => !isTaskTerminal(task)).length
  const completedCount = tasks.filter((task) => task.status === 'completed').length
  const hasActiveTasks = activeCount > 0
  const sortedTasks = useMemo(() => sortTasksForSessionBar(tasks), [tasks])

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
        <span className="session-task-title">
          <strong>Subagent tasks</strong>
          <small>{hasActiveTasks ? 'Running background work' : 'Task history'}</small>
        </span>
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
            <TaskItem key={task.task_id} onOpenTask={onOpenTask} task={task} />
          ))}
        </ol>
      ) : null}
    </section>
  )
}

export function isTaskTerminal(task: Pick<TaskSummary, 'status'>): boolean {
  return task.status === 'completed' || task.status === 'failed' || task.status === 'interrupted' || task.status === 'killed'
}

export function sortTasksForSessionBar(tasks: TaskSummary[]): TaskSummary[] {
  const childrenByParent = new Map<string, TaskSummary[]>()
  const roots: TaskSummary[] = []
  const ids = new Set(tasks.map((task) => task.task_id))
  for (const task of tasks) {
    if (task.parent_task_id && ids.has(task.parent_task_id)) {
      const children = childrenByParent.get(task.parent_task_id) ?? []
      children.push(task)
      childrenByParent.set(task.parent_task_id, children)
    } else {
      roots.push(task)
    }
  }

  const byStartedDesc = (a: TaskSummary, b: TaskSummary) => b.started_at - a.started_at
  roots.sort(byStartedDesc)
  for (const children of childrenByParent.values()) children.sort(byStartedDesc)

  const sorted: TaskSummary[] = []
  const visit = (task: TaskSummary) => {
    sorted.push(task)
    for (const child of childrenByParent.get(task.task_id) ?? []) visit(child)
  }
  for (const root of roots) visit(root)
  return sorted
}

function taskIndent(task: TaskSummary): number {
  return Math.max(0, Math.min(3, task.child_depth - 1)) * 16
}

function TaskItem({ task, onOpenTask }: { task: TaskSummary; onOpenTask?: (task: TaskSummary) => void }) {
  const Icon = iconForTask(task)
  const tone = toneForTask(task)
  const detail = task.error || task.summary || activityDetail(task)
  return (
    <li className={'session-task-item session-task-' + tone} style={{ marginLeft: taskIndent(task) }}>
      <button
        type="button"
        className="session-task-row"
        onClick={() => onOpenTask?.(task)}
        aria-label={'Open task ' + task.task_id}
      >
        <Icon size={15} aria-hidden="true" />
        <div>
          <div className="session-task-main">
            <span>{task.prompt}</span>
            <em>{task.subagent_type}</em>
          </div>
          {detail ? <p>{detail}</p> : null}
        </div>
        <span className="session-task-meta">
          <span>{task.status}</span>
          {task.model ? <em>{task.model}</em> : null}
        </span>
      </button>
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
