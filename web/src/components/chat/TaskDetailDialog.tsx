import { RefreshCw, X } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { TaskSummary } from '../../api/types'
import { CodeBlock } from './blocks'

type Props = {
  taskId: string
  initialTask?: TaskSummary
  onClose: () => void
}

export const TASK_DETAIL_REFRESH_MS = 2000

export function TaskDetailDialog({ taskId, initialTask, onClose }: Props) {
  const [task, setTask] = useState<TaskSummary | null>(initialTask ?? null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const mountedRef = useRef(false)

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
          {task ? <TaskDetailContent task={task} /> : loading ? <div className="empty-chat">Loading task details...</div> : null}
        </div>
      </section>
    </div>
  )
}

function TaskDetailContent({ task }: { task: TaskSummary }) {
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
      {task.summary ? <TaskTextSection title="Summary" text={task.summary} /> : null}
      {task.error ? <TaskTextSection title="Error" text={task.error} tone="error" /> : null}
      {task.output_tail ? <CodeBlock code={task.output_tail} language="text" title="Output tail" /> : null}
      {metadataText ? <CodeBlock code={metadataText} language="json" title="Metadata" /> : null}
    </div>
  )
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
