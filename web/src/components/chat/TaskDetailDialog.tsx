import { ChevronDown, ChevronRight, ExternalLink, FileArchive, FileCode2, FileText, GitBranch, Mail, MessageSquareText, RefreshCw, Terminal, X } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { TaskChildMessageStream, TaskDeliveredMessage, TaskMessage, TaskPendingMessage, TaskResultArtifact, TaskSummary } from '../../api/types'
import { CopyButton } from '../shared/CopyButton'
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
  const [messages, setMessages] = useState<TaskMessage[]>([])
  const [pendingMessages, setPendingMessages] = useState<TaskPendingMessage[]>([])
  const [deliveredMessages, setDeliveredMessages] = useState<TaskDeliveredMessage[]>([])
  const [childStreams, setChildStreams] = useState<TaskChildMessageStream[]>([])
  const [childStreamsError, setChildStreamsError] = useState<string | null>(null)
  const [messagesError, setMessagesError] = useState<string | null>(null)
  const [taskResult, setTaskResult] = useState<TaskResultArtifact | null>(null)
  const [resultError, setResultError] = useState<string | null>(null)
  const mountedRef = useRef(false)

  useEffect(() => {
    setTask(initialTask ?? null)
    setLoading(true)
    setError(null)
    setMessages([])
    setPendingMessages([])
    setDeliveredMessages([])
    setChildStreams([])
    setChildStreamsError(null)
    setMessagesError(null)
    setTaskResult(null)
    setResultError(null)
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
    setMessagesError(null)
    setChildStreamsError(null)
    setResultError(null)
    try {
      const detail = await api.taskDetail(taskId)
      if (!mountedRef.current) return
      setTask(detail)
      try {
        const result = await api.taskMessages(taskId, { limit: 100 })
        if (!mountedRef.current) return
        setMessages(result.messages ?? [])
        setPendingMessages(result.pending_messages ?? [])
        setDeliveredMessages(result.delivered_messages ?? [])
      } catch (reason) {
        if (!mountedRef.current) return
        setMessagesError(reason instanceof Error ? reason.message : String(reason))
      }
      try {
        const result = await api.taskChildMessages(taskId, { limit: 50, perTaskLimit: 25 })
        if (!mountedRef.current) return
        setChildStreams(result.streams ?? [])
      } catch (reason) {
        if (!mountedRef.current) return
        setChildStreamsError(reason instanceof Error ? reason.message : String(reason))
      }
      if (detail.result_path) {
        try {
          const result = await api.taskResult(taskId)
          if (!mountedRef.current) return
          setTaskResult(result)
        } catch (reason) {
          if (!mountedRef.current) return
          setTaskResult(null)
          setResultError(reason instanceof Error ? reason.message : String(reason))
        }
      } else {
        setTaskResult(null)
      }
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
          {task ? <TaskDetailContent task={task} messages={messages} pendingMessages={pendingMessages} deliveredMessages={deliveredMessages} messagesError={messagesError} childStreams={childStreams} childStreamsError={childStreamsError} taskResult={taskResult} resultError={resultError} sessionTasks={sessionTasks} onOpenTask={onOpenTask} /> : loading ? <div className="empty-chat">Loading task details...</div> : null}
        </div>
      </section>
    </div>
  )
}

function TaskDetailContent({ task, messages, pendingMessages, deliveredMessages, messagesError, childStreams, childStreamsError, taskResult, resultError, sessionTasks, onOpenTask }: { task: TaskSummary; messages: TaskMessage[]; pendingMessages: TaskPendingMessage[]; deliveredMessages: TaskDeliveredMessage[]; messagesError: string | null; childStreams: TaskChildMessageStream[]; childStreamsError: string | null; taskResult: TaskResultArtifact | null; resultError: string | null; sessionTasks: TaskSummary[]; onOpenTask?: (taskId: string) => void }) {
  const metadataText = task.metadata && Object.keys(task.metadata).length > 0
    ? JSON.stringify(task.metadata, null, 2)
    : ''
  const provider = taskProvider(task)
  return (
    <div className="task-detail-content">
      <section className="task-detail-grid">
        <Info label="Status" value={task.status} />
        <Info label="Subagent" value={task.subagent_type} />
        <Info label="Session" value={task.parent_session_id} />
        <Info label="Provider" value={provider} />
        <Info label="Model" value={task.model} />
        <Info label="Duration" value={formatTaskDuration(task)} />
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
      <TaskMessageTimeline messages={messages} pendingMessages={pendingMessages} deliveredMessages={deliveredMessages} error={messagesError} />
      <ChildTaskMessageStreams streams={childStreams} error={childStreamsError} onOpenTask={onOpenTask} />
      {task.summary ? <TaskTextSection title="Summary" text={task.summary} /> : null}
      {task.error ? <TaskTextSection title="Error" text={task.error} tone="error" /> : null}
      {resultError ? <TaskTextSection title="Result artifact" text={resultError} tone="error" /> : null}
      {taskResult ? <TaskResultArtifactView artifact={taskResult} /> : null}
      {task.output_tail ? <CodeBlock code={task.output_tail} language="text" title="Output tail" /> : null}
      {metadataText ? <CodeBlock code={metadataText} language="json" title="Metadata" /> : null}
    </div>
  )
}

function TaskResultArtifactView({ artifact }: { artifact: TaskResultArtifact }) {
  const code = JSON.stringify(artifact.result, null, 2)
  const artifacts = taskResultArtifacts(artifact)
  return (
    <section className="task-result-artifact" aria-label="Task result artifact">
      <header>
        <span><FileArchive size={14} aria-hidden="true" />Result artifact</span>
        {artifact.result_path ? (
          <span className="task-result-artifact-path">
            <code>{artifact.result_path}</code>
            <CopyButton text={artifact.result_path} label="Copy task result path" displayLabel="Copy path" displayCopiedLabel="Copied" />
          </span>
        ) : null}
      </header>
      <div className="task-result-actions">
        <CopyButton text={code} label="Copy task result JSON" displayLabel="Copy JSON" displayCopiedLabel="Copied" />
      </div>
      {artifacts.length > 0 ? <TaskResultArtifactList artifacts={artifacts} /> : null}
      <CodeBlock code={code} language="json" title="result.json" />
    </section>
  )
}

function TaskResultArtifactList({ artifacts }: { artifacts: TaskResultFile[] }) {
  return (
    <div className="task-result-file-list" aria-label="Task result files">
      {artifacts.map((artifact, index) => {
        const Icon = taskResultFileIcon(artifact)
        return (
          <article className="task-result-file" key={(artifact.href || artifact.path || artifact.name) + '-' + index}>
            <Icon size={14} aria-hidden="true" />
            <span>
              <strong>{artifact.name}</strong>
              <small>{taskResultFileMeta(artifact)}</small>
              {artifact.note ? <em>{artifact.note}</em> : null}
            </span>
            <span className="task-result-file-actions">
              {artifact.path || artifact.href ? (
                <CopyButton
                  text={artifact.path || artifact.href || ''}
                  label={'Copy ' + artifact.name + ' path'}
                  displayLabel="Copy"
                  displayCopiedLabel="Copied"
                />
              ) : null}
              {artifact.href ? (
                <a href={artifact.href} target="_blank" rel="noreferrer">
                  <ExternalLink size={13} aria-hidden="true" />
                  Open
                </a>
              ) : null}
            </span>
          </article>
        )
      })}
    </div>
  )
}

function ChildTaskMessageStreams({ streams, error, onOpenTask }: { streams: TaskChildMessageStream[]; error: string | null; onOpenTask?: (taskId: string) => void }) {
  const [filter, setFilter] = useState<ChildStreamFilter>('with-activity')
  const [collapsedTaskIds, setCollapsedTaskIds] = useState<Set<string>>(() => new Set())
  const activityStreams = streams.filter(hasTaskStreamActivity)
  const visibleStreams = filterChildStreams(streams, filter)
  if (streams.length === 0 && !error) return null
  const activeCount = streams.filter((stream) => !isTaskTerminal(stream.task)).length
  const terminalCount = streams.filter((stream) => isTaskTerminal(stream.task)).length
  const toggleCollapsed = (taskId: string) => {
    setCollapsedTaskIds((current) => {
      const next = new Set(current)
      if (next.has(taskId)) next.delete(taskId)
      else next.add(taskId)
      return next
    })
  }
  return (
    <section className="task-child-streams" aria-label="Child task message streams">
      <header>
        <span><GitBranch size={14} aria-hidden="true" />Child message streams</span>
        <small>{visibleStreams.length.toLocaleString()} of {streams.length.toLocaleString()} task{streams.length === 1 ? '' : 's'}</small>
      </header>
      <div className="task-child-stream-controls" role="group" aria-label="Filter child task streams">
        <button type="button" className={filter === 'with-activity' ? 'active' : ''} onClick={() => setFilter('with-activity')}>
          With activity <span>{activityStreams.length}</span>
        </button>
        <button type="button" className={filter === 'active' ? 'active' : ''} onClick={() => setFilter('active')}>
          Active <span>{activeCount}</span>
        </button>
        <button type="button" className={filter === 'terminal' ? 'active' : ''} onClick={() => setFilter('terminal')}>
          Finished <span>{terminalCount}</span>
        </button>
        <button type="button" className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>
          All <span>{streams.length}</span>
        </button>
      </div>
      {error ? <div className="task-message-error">{error}</div> : null}
      {visibleStreams.length === 0 ? <div className="task-child-stream-empty">No child streams match this filter.</div> : null}
      {visibleStreams.map((stream) => {
        const collapsed = collapsedTaskIds.has(stream.task.task_id)
        return (
          <article className="task-child-stream" key={stream.task.task_id}>
            <header>
              <button
                type="button"
                className="task-child-stream-toggle"
                aria-expanded={!collapsed}
                onClick={() => toggleCollapsed(stream.task.task_id)}
              >
                {collapsed ? <ChevronRight size={14} aria-hidden="true" /> : <ChevronDown size={14} aria-hidden="true" />}
                <span>
                  <strong>{stream.task.prompt || stream.task.task_id}</strong>
                  <small>{stream.task.task_id}</small>
                </span>
              </button>
              <em>{stream.task.status}</em>
            </header>
            {!collapsed ? (
              <>
                <TaskMessageTimeline messages={stream.messages} pendingMessages={stream.pending_messages} deliveredMessages={stream.delivered_messages} error={null} />
                {onOpenTask ? (
                  <button type="button" onClick={() => onOpenTask(stream.task.task_id)}>
                    Open child task
                  </button>
                ) : null}
              </>
            ) : null}
          </article>
        )
      })}
    </section>
  )
}

function TaskMessageTimeline({ messages, pendingMessages, deliveredMessages, error }: { messages: TaskMessage[]; pendingMessages: TaskPendingMessage[]; deliveredMessages: TaskDeliveredMessage[]; error: string | null }) {
  if (messages.length === 0 && pendingMessages.length === 0 && deliveredMessages.length === 0 && !error) return null
  return (
    <section className="task-message-timeline" aria-label="Task message stream">
      <header>
        <span><MessageSquareText size={14} aria-hidden="true" />Message stream</span>
        <small>{(messages.length + pendingMessages.length + deliveredMessages.length).toLocaleString()} event{messages.length + pendingMessages.length + deliveredMessages.length === 1 ? '' : 's'}</small>
      </header>
      {error ? <div className="task-message-error">{error}</div> : null}
      {pendingMessages.length > 0 ? (
        <div className="task-pending-messages" aria-label="Queued parent messages">
          <span><Mail size={13} aria-hidden="true" />Queued parent messages</span>
          <ol>
            {pendingMessages.map((message) => (
              <TaskPendingMessageRow message={message} key={message.index + '-' + message.message} />
            ))}
          </ol>
        </div>
      ) : null}
      {deliveredMessages.length > 0 ? (
        <div className="task-delivered-messages" aria-label="Delivered parent messages">
          <span><Mail size={13} aria-hidden="true" />Delivered parent messages</span>
          <ol>
            {deliveredMessages.map((message) => (
              <TaskDeliveredMessageRow message={message} key={message.index + '-' + message.message} />
            ))}
          </ol>
        </div>
      ) : null}
      {messages.length > 0 ? (
        <ol>
          {messages.map((message) => (
            <TaskMessageRow message={message} key={message.index + '-' + message.role + '-' + (message.tool_call_id || message.name || '')} />
          ))}
        </ol>
      ) : null}
    </section>
  )
}

function TaskPendingMessageRow({ message }: { message: TaskPendingMessage }) {
  return (
    <li className="task-message-row task-message-pending">
      <div className="task-message-head">
        <span>
          <Mail size={13} aria-hidden="true" />
          <strong>parent message</strong>
        </span>
        {typeof message.ts === 'number' ? <em>{formatTimestamp(message.ts)}</em> : null}
      </div>
      <pre>{message.message}</pre>
    </li>
  )
}

function TaskDeliveredMessageRow({ message }: { message: TaskDeliveredMessage }) {
  const detail = typeof message.delivered_at === 'number' ? 'delivered ' + formatTimestamp(message.delivered_at) : null
  return (
    <li className="task-message-row task-message-delivered">
      <div className="task-message-head">
        <span>
          <Mail size={13} aria-hidden="true" />
          <strong>delivered parent message</strong>
        </span>
        {detail ? <em>{detail}</em> : typeof message.ts === 'number' ? <em>{formatTimestamp(message.ts)}</em> : null}
      </div>
      <pre>{message.message}</pre>
    </li>
  )
}

function TaskMessageRow({ message }: { message: TaskMessage }) {
  const tool = message.role === 'tool'
  const title = tool ? message.name || 'tool' : message.role === 'assistant' ? 'assistant' : message.role || 'message'
  const detail = taskMessageDetail(message)
  const body = message.error || message.content || ''
  return (
    <li className={'task-message-row task-message-' + taskMessageTone(message)}>
      <div className="task-message-head">
        <span>
          {tool ? <Terminal size={13} aria-hidden="true" /> : <MessageSquareText size={13} aria-hidden="true" />}
          <strong>{title}</strong>
        </span>
        {detail ? <em>{detail}</em> : null}
      </div>
      {body ? <pre>{body}</pre> : <p>No message body captured.</p>}
    </li>
  )
}

function taskMessageDetail(message: TaskMessage): string {
  const parts: string[] = []
  if (typeof message.iteration === 'number') parts.push('iteration ' + message.iteration)
  if (typeof message.elapsed_ms === 'number') parts.push(formatElapsedMs(message.elapsed_ms))
  if (message.tool_call_id) parts.push(message.tool_call_id)
  return parts.join(' / ')
}

function taskMessageTone(message: TaskMessage): string {
  if (message.is_error || message.error) return 'error'
  if (message.role === 'tool') return 'tool'
  if (message.role === 'assistant') return 'assistant'
  return 'neutral'
}

type ChildStreamFilter = 'with-activity' | 'active' | 'terminal' | 'all'

type TaskResultFile = {
  name: string
  path: string | null
  href: string | null
  kind: string | null
  mimeType: string | null
  size: number | null
  note: string | null
  binary: boolean
  language: string | null
}

function hasTaskStreamActivity(stream: TaskChildMessageStream): boolean {
  return stream.messages.length > 0 || stream.pending_messages.length > 0 || stream.delivered_messages.length > 0
}

function filterChildStreams(streams: TaskChildMessageStream[], filter: ChildStreamFilter): TaskChildMessageStream[] {
  if (filter === 'all') return streams
  if (filter === 'active') return streams.filter((stream) => !isTaskTerminal(stream.task))
  if (filter === 'terminal') return streams.filter((stream) => isTaskTerminal(stream.task))
  return streams.filter(hasTaskStreamActivity)
}

function taskResultArtifacts(artifact: TaskResultArtifact): TaskResultFile[] {
  const files: TaskResultFile[] = []
  const seen = new Set<string>()
  const add = (file: TaskResultFile | null) => {
    if (!file) return
    const key = file.href || file.path || file.name
    if (seen.has(key)) return
    seen.add(key)
    files.push(file)
  }
  collectTaskResultArtifacts(artifact.result, add)
  if (artifact.result_path) {
    add(taskResultFileFromRecord({ path: artifact.result_path, name: 'result.json', kind: 'result', mime_type: 'application/json' }))
  }
  return files
}

function collectTaskResultArtifacts(value: unknown, add: (file: TaskResultFile | null) => void): void {
  if (!isRecord(value)) return
  for (const key of ['artifacts', 'attachments', 'files', 'outputs', 'results']) {
    const items = value[key]
    if (!Array.isArray(items)) continue
    for (const item of items) add(taskResultFileFromRecord(recordOrNull(item)))
  }
  add(taskResultFileFromRecord(pickTaskResultFileRecord(value)))
}

function taskResultFileFromRecord(record: Record<string, unknown> | null): TaskResultFile | null {
  if (!record) return null
  const path = firstString(record.path, record.file_path, record.filePath, record.result_path, record.resultPath, record.output_file, record.outputFile)
  const href = safeHref(firstString(record.url, record.href, record.uri, record.download_url, record.downloadUrl, record.preview_url, record.previewUrl))
  if (!path && !href) return null
  const fallback = path || href || 'artifact'
  const name = firstString(record.title, record.name, record.filename, record.file_name, record.label) || artifactNameFromPath(fallback)
  const mimeType = firstString(record.mime_type, record.mimeType, record.media_type, record.mediaType)
  const kind = firstString(record.kind, record.type)
  const binary = isBinaryResultFile(record, mimeType, kind, fallback)
  return {
    name,
    path: path || null,
    href,
    kind,
    mimeType,
    size: numberValue(record.size) ?? numberValue(record.size_bytes) ?? numberValue(record.sizeBytes) ?? numberValue(record.bytes),
    note: firstString(record.caption, record.description, record.summary),
    binary,
    language: binary ? null : firstString(record.language, record.lang) || languageFromPath(fallback),
  }
}

function pickTaskResultFileRecord(record: Record<string, unknown>): Record<string, unknown> | null {
  const path = firstString(record.result_path, record.resultPath, record.output_file, record.outputFile, record.artifact_path, record.artifactPath)
  if (!path) return null
  return { path, name: firstString(record.result_name, record.resultName, record.output_name, record.outputName), kind: 'result' }
}

function taskResultFileMeta(file: TaskResultFile): string {
  const parts = [file.kind, file.binary ? 'binary' : null, file.mimeType, file.language, file.size != null ? file.size.toLocaleString() + ' bytes' : null, file.path].filter(Boolean)
  return parts.join(' / ') || 'artifact'
}

function taskResultFileIcon(file: TaskResultFile) {
  if (file.binary || file.mimeType?.includes('zip') || file.kind?.toLowerCase().includes('archive')) return FileArchive
  if (file.language || file.mimeType?.includes('json') || file.mimeType?.startsWith('text/')) return FileCode2
  return FileText
}

function isBinaryResultFile(record: Record<string, unknown>, mimeType: string | null, kind: string | null, fallbackPath: string): boolean {
  const explicit = booleanValue(record.binary) ?? booleanValue(record.is_binary) ?? booleanValue(record.isBinary)
  if (explicit != null) return explicit
  const normalizedMime = (mimeType || '').toLowerCase()
  if (normalizedMime.startsWith('text/') || normalizedMime.includes('json') || normalizedMime.includes('xml') || normalizedMime.includes('yaml') || normalizedMime.includes('toml')) return false
  if (normalizedMime === 'application/octet-stream' || normalizedMime.includes('zip') || normalizedMime.includes('pdf') || normalizedMime.startsWith('image/')) return true
  const normalizedKind = (kind || '').toLowerCase()
  if (normalizedKind.includes('binary') || normalizedKind.includes('archive')) return true
  const language = languageFromPath(fallbackPath)
  return language == null && /\.(?:bin|pdf|zip|tar|gz|tgz|sqlite|db|parquet|png|jpe?g|gif|webp)$/i.test(fallbackPath)
}

function taskProvider(task: TaskSummary): string | null {
  const metadata = task.metadata ?? {}
  return firstString(metadata.provider, metadata.provider_name, metadata.providerName, metadata.llm_provider, metadata.llmProvider)
}

function formatTaskDuration(task: TaskSummary): string | null {
  const metadata = task.metadata ?? {}
  const explicitSeconds = numberValue(metadata.duration_seconds) ?? numberValue(metadata.durationSeconds)
  if (explicitSeconds != null) return formatDurationSeconds(explicitSeconds)
  const explicitMs = numberValue(metadata.duration_ms) ?? numberValue(metadata.durationMs)
  if (explicitMs != null) return formatDurationSeconds(explicitMs / 1000)
  const end = task.finished_at ?? task.last_heartbeat
  if (!task.started_at || !end || end < task.started_at) return null
  return formatDurationSeconds(end - task.started_at)
}

function formatDurationSeconds(seconds: number): string | null {
  if (!Number.isFinite(seconds) || seconds < 0) return null
  if (seconds < 1) return Math.round(seconds * 1000) + 'ms'
  if (seconds < 10) return seconds.toFixed(1).replace(/\.0$/, '') + 's'
  if (seconds < 60) return Math.round(seconds) + 's'
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  return minutes + 'm ' + rest + 's'
}

function languageFromPath(path?: string | null): string | null {
  if (!path) return null
  const ext = path.split(/[?#]/, 1)[0]?.split('.').pop()?.toLowerCase()
  if (!ext || ext === path) return null
  if (ext === 'py') return 'python'
  if (ext === 'js') return 'javascript'
  if (ext === 'ts') return 'typescript'
  if (ext === 'tsx') return 'tsx'
  if (ext === 'jsx') return 'jsx'
  if (ext === 'md') return 'markdown'
  if (ext === 'sh') return 'bash'
  return ext
}

function firstString(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return null
}

function numberValue(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

function booleanValue(value: unknown): boolean | null {
  if (typeof value === 'boolean') return value
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase()
    if (['true', 'yes', '1'].includes(normalized)) return true
    if (['false', 'no', '0'].includes(normalized)) return false
  }
  return null
}

function safeHref(href: string | null): string | null {
  if (!href) return null
  const trimmed = href.trim()
  return /^(https?:|#|\/(?!\/)|\.\/|\.\.\/)/i.test(trimmed) ? trimmed : null
}

function artifactNameFromPath(path: string): string {
  const clean = path.split(/[?#]/, 1)[0] || path
  return clean.split('/').filter(Boolean).pop() || 'artifact'
}

function recordOrNull(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function formatElapsedMs(value: number): string {
  if (!Number.isFinite(value) || value < 0) return ''
  if (value < 1000) return Math.round(value) + 'ms'
  return (value / 1000).toFixed(1).replace(/\.0$/, '') + 's'
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
