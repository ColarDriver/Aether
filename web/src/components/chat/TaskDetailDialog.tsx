import { GitBranch, RefreshCw, X } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { TaskSummary } from '../../api/types'
import { CodeBlock } from './blocks'

type Props = {
  taskId: string
  initialTask?: TaskSummary
  sessionTasks?: TaskSummary[]
  onOpenTask?: (taskId: string) => void
  onClose: () => void
}

export const TASK_DETAIL_REFRESH_MS = 2000

export function TaskDetailDialog({ taskId, initialTask, sessionTasks = [], onOpenTask, onClose }: Props) {
  const [task, setTask] = useState<TaskSummary | null>(initialTask ?? null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const mountedRef = useRef(false)

  useEffect(() => {
    setTask(initialTask ?? null)
    setLoading(true)
    setError(null)
  }, [initialTask, taskId])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  const loadTask = useCallback(async (options: { initial?: boolean } = {}) => {
    if (options.initial) setLoading(true)
    setRefreshing(true)
    setError(null)
    try {
      const detail = await api.taskDetail(taskId)
      if (!mountedRef.current) return
      setTask(detail)
    } catch (reason) {
      if (!mountedRef.current) return
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      if (!mountedRef.current) return
      setLoading(false)
      setRefreshing(false)
    }
  }, [taskId])

  useEffect(() => {
    void loadTask({ initial: true })
  }, [loadTask])

  useEffect(() => {
    if (!task || isTaskTerminal(task)) return
    const interval = window.setInterval(() => {
      void loadTask()
    }, TASK_DETAIL_REFRESH_MS)
    return () => window.clearInterval(interval)
  }, [loadTask, task])

  return (
    <div className="modal-backdrop" role="presentation">
      <section className="prompt-modal task-detail-modal" role="dialog" aria-modal="true" aria-label="Task details">
        <header>
          <div>
            <strong>{task?.prompt || taskId}</strong>
            <span>{taskId}</span>
          </div>
          <div className="task-detail-actions">
            <button type="button" aria-label="Refresh task details" disabled={refreshing} onClick={() => void loadTask()}>
              <RefreshCw size={16} aria-hidden="true" />
            </button>
            <button type="button" aria-label="Close task details" onClick={onClose}>
              <X size={16} aria-hidden="true" />
            </button>
          </div>
        </header>
        <div className="prompt-body task-detail-body">
          {error ? <div className="chat-block chat-block-error">{error}</div> : null}
          {refreshing && task ? <div className="task-detail-refreshing">Refreshing task details...</div> : null}
          {task ? <TaskDetailContent task={task} sessionTasks={sessionTasks} onOpenTask={onOpenTask} /> : loading ? <div className="empty-chat">Loading task details...</div> : null}
        </div>
      </section>
    </div>
  )
}

function TaskDetailContent({ task, sessionTasks, onOpenTask }: { task: TaskSummary; sessionTasks: TaskSummary[]; onOpenTask?: (taskId: string) => void }) {
  const metadataText = task.metadata && Object.keys(task.metadata).length > 0
    ? JSON.stringify(task.metadata, null, 2)
    : ''
  return (
    <div className="task-detail-content">
      <section className="task-detail-grid">
        <Info label="Status" value={task.status} />
        <Info label="Subagent" value={task.subagent_type} />
        <Info label="Session" value={task.parent_session_id} />
        <Info label="Model" value={task.model} />
        <Info label="Background" value={task.background ? 'yes' : 'no'} />
        <Info label="Iterations" value={String(task.iterations)} />
        <Info label="Tool calls" value={String(task.tool_use_count)} />
        <Info label="Tokens" value={String(task.input_tokens + task.output_tokens)} />
        <Info label="Started" value={formatTimestamp(task.started_at)} />
        <Info label="Finished" value={formatTimestamp(task.finished_at)} />
        <Info label="Worktree" value={task.worktree_path} wide />
        <Info label="Result" value={task.result_path} wide />
      </section>
      <RelatedTasks currentTask={task} sessionTasks={sessionTasks} onOpenTask={onOpenTask} />
      {task.summary ? <TaskTextSection title="Summary" text={task.summary} /> : null}
      {task.error ? <TaskTextSection title="Error" text={task.error} tone="error" /> : null}
      {task.output_tail ? <CodeBlock code={task.output_tail} language="text" title="Output tail" /> : null}
      {metadataText ? <CodeBlock code={metadataText} language="json" title="Metadata" /> : null}
    </div>
  )
}

type RelatedTaskRow = {
  task: TaskSummary
  relation: 'parent' | 'current' | 'child'
  depth: number
}

function RelatedTasks({ currentTask, sessionTasks, onOpenTask }: { currentTask: TaskSummary; sessionTasks: TaskSummary[]; onOpenTask?: (taskId: string) => void }) {
  const rows = relatedTaskRows(currentTask, sessionTasks)
  if (rows.length <= 1) return null
  const childCount = rows.filter((row) => row.relation === 'child').length
  return (
    <section className="task-detail-related" aria-label="Related tasks">
      <header>
        <span><GitBranch size={14} aria-hidden="true" />Related tasks</span>
        <small>{childCount.toLocaleString()} child task{childCount === 1 ? '' : 's'}</small>
      </header>
      <ol>
        {rows.map((row) => (
          <li className={'task-detail-related-item task-detail-related-' + row.relation} key={row.task.task_id} style={{ paddingLeft: 10 + row.depth * 14 }}>
            <button
              type="button"
              disabled={row.relation === 'current' || !onOpenTask}
              aria-label={row.relation === 'current' ? 'Current task ' + row.task.task_id : 'Open related task ' + row.task.task_id}
              onClick={() => onOpenTask?.(row.task.task_id)}
            >
              <span>
                <strong>{row.task.prompt || row.task.task_id}</strong>
                <small>{row.task.task_id}</small>
              </span>
              <em>{row.relation}</em>
              <code>{row.task.status}</code>
            </button>
            <p>{relatedTaskDetail(row.task)}</p>
          </li>
        ))}
      </ol>
    </section>
  )
}

function relatedTaskRows(currentTask: TaskSummary, sessionTasks: TaskSummary[]): RelatedTaskRow[] {
  const byId = new Map<string, TaskSummary>()
  for (const task of sessionTasks) byId.set(task.task_id, task)
  byId.set(currentTask.task_id, currentTask)

  const rows: RelatedTaskRow[] = []
  const parent = currentTask.parent_task_id ? byId.get(currentTask.parent_task_id) : null
  if (parent) rows.push({ task: parent, relation: 'parent', depth: 0 })

  const currentDepth = parent ? 1 : 0
  rows.push({ task: currentTask, relation: 'current', depth: currentDepth })

  const descendants = [...byId.values()]
    .filter((task) => task.task_id !== currentTask.task_id && isDescendantOf(task, currentTask.task_id, byId))
    .sort((a, b) => a.child_depth - b.child_depth || a.started_at - b.started_at)

  for (const child of descendants) {
    rows.push({
      task: child,
      relation: 'child',
      depth: currentDepth + Math.max(1, child.child_depth - currentTask.child_depth),
    })
  }
  return rows
}

function isDescendantOf(task: TaskSummary, parentTaskId: string, byId: Map<string, TaskSummary>): boolean {
  let nextParentId = task.parent_task_id
  const seen = new Set<string>()
  while (nextParentId) {
    if (nextParentId === parentTaskId) return true
    if (seen.has(nextParentId)) return false
    seen.add(nextParentId)
    nextParentId = byId.get(nextParentId)?.parent_task_id ?? null
  }
  return false
}

function relatedTaskDetail(task: TaskSummary): string {
  const parts: string[] = []
  if (task.subagent_type) parts.push(task.subagent_type)
  if (task.model) parts.push(task.model)
  if (task.background) parts.push('background')
  if (task.iterations > 0) parts.push(task.iterations + ' iterations')
  if (task.tool_use_count > 0) parts.push(task.tool_use_count + ' tool calls')
  const tokens = task.input_tokens + task.output_tokens
  if (tokens > 0) parts.push(tokens.toLocaleString() + ' tokens')
  if (task.error) parts.push(task.error)
  else if (task.summary) parts.push(task.summary)
  return parts.join(' / ') || 'No progress metadata yet'
}

function Info({ label, value, wide = false }: { label: string; value?: string | null; wide?: boolean }) {
  if (!value) return null
  return (
    <div className={wide ? 'task-detail-info task-detail-info-wide' : 'task-detail-info'}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function TaskTextSection({ title, text, tone }: { title: string; text: string; tone?: 'error' }) {
  return (
    <section className={tone === 'error' ? 'task-detail-note task-detail-note-error' : 'task-detail-note'}>
      <span>{title}</span>
      <p>{text}</p>
    </section>
  )
}

function formatTimestamp(value?: number | null): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return null
  return new Date(value * 1000).toLocaleString()
}

function isTaskTerminal(task: Pick<TaskSummary, 'status'>): boolean {
  return task.status === 'completed' || task.status === 'failed' || task.status === 'interrupted' || task.status === 'killed'
}
