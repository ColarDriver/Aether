import { ArrowDown, Bot } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { SessionInfo, TaskSummary } from '../../api/types'
import type { ChatBlock } from '../../chat-rendering'
import { useAppStore } from '../../stores/appStore'
import { useChatStore } from '../../stores/chatStore'
import { useProviderStore } from '../../stores/providerStore'
import { useSessionStore } from '../../stores/sessionStore'
import { useTaskStore } from '../../stores/taskStore'
import { EmptyState } from '../shared/EmptyState'
import { ApprovalDialog } from './ApprovalDialog'
import { ChatTimeline } from './ChatTimeline'
import { Composer } from './Composer'
import { PermissionDialog } from './PermissionDialog'
import { isTaskTerminal, SessionTaskBar } from './SessionTaskBar'
import { executeWebSlashCommand } from './slashExecute'
import { TaskDetailDialog } from './TaskDetailDialog'

type Props = {
  session: SessionInfo | null
}

const AUTO_SCROLL_BOTTOM_THRESHOLD_PX = 48
const EMPTY_CHAT_BLOCKS: ChatBlock[] = []
const EMPTY_TASKS: TaskSummary[] = []

export function ChatView({ session }: Props) {
  const sessionId = session?.session_id ?? null
  const loadTranscript = useChatStore((state) => state.loadTranscript)
  const blocks = useChatStore((state) => (sessionId ? state.blocksBySession[sessionId] ?? EMPTY_CHAT_BLOCKS : EMPTY_CHAT_BLOCKS))
  const tokenUsageByRun = useChatStore((state) => state.tokenUsageByRun)
  const startRun = useChatStore((state) => state.startRun)
  const cancelRun = useChatStore((state) => state.cancelRun)
  const appendLocalNotice = useChatStore((state) => state.appendLocalNotice)
  const appendLocalError = useChatStore((state) => state.appendLocalError)
  const activeRunId = useChatStore((state) => state.activeRunId)
  const pendingPermission = useChatStore((state) => state.pendingPermission)
  const pendingApproval = useChatStore((state) => state.pendingApproval)
  const respondPermission = useChatStore((state) => state.respondPermission)
  const respondApproval = useChatStore((state) => state.respondApproval)
  const tasks = useTaskStore((state) => (sessionId ? state.tasksBySession[sessionId] ?? EMPTY_TASKS : EMPTY_TASKS))
  const loadSessionTasks = useTaskStore((state) => state.loadSessionTasks)
  const setActiveView = useAppStore((state) => state.setActiveView)
  const createSession = useSessionStore((state) => state.createSession)
  const resumeSession = useSessionStore((state) => state.resumeSession)
  const updateSession = useSessionStore((state) => state.updateSession)
  const setSessionMode = useSessionStore((state) => state.setSessionMode)
  const currentProvider = useProviderStore((state) => state.current)
  const providers = useProviderStore((state) => state.providers)
  const scrollRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const shouldAutoScrollRef = useRef(true)
  const programmaticScrollRef = useRef(false)
  const lastSessionIdRef = useRef<string | null>(null)
  const [showJumpToLatest, setShowJumpToLatest] = useState(false)
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null)

  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'auto') => {
    shouldAutoScrollRef.current = true
    programmaticScrollRef.current = true
    if (typeof bottomRef.current?.scrollIntoView === 'function') {
      bottomRef.current.scrollIntoView({ behavior, block: 'end' })
    }
    setShowJumpToLatest(false)
    requestAnimationFrame(() => {
      programmaticScrollRef.current = false
    })
  }, [])

  const handleScroll = useCallback(() => {
    if (programmaticScrollRef.current) return
    const element = scrollRef.current
    if (!element) return
    const atBottom = isNearChatBottom(element)
    shouldAutoScrollRef.current = atBottom
    setShowJumpToLatest(!atBottom)
  }, [])

  useEffect(() => {
    if (!sessionId) return
    void loadTranscript(sessionId)
  }, [loadTranscript, sessionId])

  useEffect(() => {
    if (!sessionId) return
    void loadSessionTasks(sessionId)
  }, [loadSessionTasks, sessionId])

  useEffect(() => {
    if (!sessionId) return
    const hasActiveTasks = tasks.some((task) => !isTaskTerminal(task))
    if (!activeRunId && !hasActiveTasks) return
    const interval = window.setInterval(() => {
      void loadSessionTasks(sessionId)
    }, 2000)
    return () => window.clearInterval(interval)
  }, [activeRunId, loadSessionTasks, sessionId, tasks])

  useEffect(() => {
    if (lastSessionIdRef.current === sessionId) return
    lastSessionIdRef.current = sessionId
    setSelectedTaskId(null)
    shouldAutoScrollRef.current = true
    setShowJumpToLatest(false)
    requestAnimationFrame(() => scrollToBottom('auto'))
  }, [scrollToBottom, sessionId])

  useEffect(() => {
    if (!session) return
    if (!shouldAutoScrollRef.current) {
      setShowJumpToLatest(true)
      return
    }
    requestAnimationFrame(() => scrollToBottom('smooth'))
  }, [blocks, scrollToBottom, session])

  const usage = activeRunId ? tokenUsageByRun[activeRunId] : undefined
  const fallbackProvider = currentProvider?.provider_name || providers[0]?.name || 'openai'
  const fallbackModel = currentProvider?.model || 'gpt-5.4'

  const createSessionAndRun = (message: string, attachments?: Parameters<typeof startRun>[2]) => {
    void createSession({ provider: fallbackProvider, model: fallbackModel })
      .then((created) => startRun(created.session_id, message, attachments))
      .catch((error) => {
        // There is no session transcript yet, so surface the failure in devtools
        // and keep the composer usable for retry.
        console.error('Failed to create session before run', error)
      })
  }

  if (!session) {
    return (
      <div className="chat-surface chat-surface-empty">
        <div className="chat-scroll">
          <EmptyState
            icon={<Bot />}
            title="Start a session"
            description="Type a message below to create a browser session and run Aether."
          />
        </div>
        <Composer
          disabled={false}
          running={false}
          provider={fallbackProvider}
          model={fallbackModel}
          onSend={createSessionAndRun}
          onCancel={() => undefined}
        />
      </div>
    )
  }

  const handleSlashCommand = (command: string) => {
    void executeWebSlashCommand(command, {
      session,
      createSession,
      resumeSession,
      updateSession,
      openView: setActiveView,
      onSessionMode: setSessionMode,
    })
      .then((result) => {
        if (result.type === 'notice') {
          appendLocalNotice(session.session_id, result.message)
          return
        }
        if (result.type === 'error') {
          appendLocalError(session.session_id, result.message)
          return
        }
        startRun(session.session_id, result.message)
      })
      .catch((error) => {
        appendLocalError(session.session_id, error instanceof Error ? error.message : String(error))
      })
  }

  return (
    <div className="chat-surface">
      <div className="chat-header">
        <Bot size={18} />
        <div>
          <div className="chat-title">{session.summary || session.session_id}</div>
          <div className="muted">{session.provider} / {session.model}</div>
        </div>
        {usage ? (
          <div className="token-pill">{usage.input_tokens ?? 0} in / {usage.output_tokens ?? 0} out</div>
        ) : null}
      </div>
      <SessionTaskBar tasks={tasks} onOpenTask={(task) => setSelectedTaskId(task.task_id)} />
      <div className="chat-scroll" onScroll={handleScroll} ref={scrollRef}>
        <ChatTimeline
          blocks={blocks}
          onOpenTask={setSelectedTaskId}
          onRespondPermission={respondPermission}
          onRespondApproval={respondApproval}
        />
        <div ref={bottomRef} />
        {showJumpToLatest ? (
          <button type="button" className="chat-jump-latest" onClick={() => scrollToBottom('smooth')}>
            <ArrowDown size={14} />
            Jump to latest
          </button>
        ) : null}
      </div>
      <Composer
        disabled={!session}
        running={Boolean(activeRunId)}
        provider={session.provider}
        model={session.model}
        mode={session.mode}
        inputTokens={usage?.input_tokens}
        outputTokens={usage?.output_tokens}
        onSend={(message, attachments) => startRun(session.session_id, message, attachments)}
        onSlashCommand={handleSlashCommand}
        onCancel={() => cancelRun(session.session_id)}
      />
      {pendingPermission ? (
        <PermissionDialog
          prompt={pendingPermission}
          onAllow={() => respondPermission({ type: 'allow_once' })}
          onAllowSession={() => respondPermission({ type: 'allow_session' })}
          onDeny={() => respondPermission({ type: 'deny' })}
        />
      ) : null}
      {pendingApproval ? (
        <ApprovalDialog
          prompt={pendingApproval}
          onApprove={(answers) => respondApproval({ confirmed: true, ...(answers ? { answers } : {}) })}
          onReject={() => respondApproval({ confirmed: false })}
        />
      ) : null}
      {selectedTaskId ? (
        <TaskDetailDialog
          taskId={selectedTaskId}
          initialTask={tasks.find((task) => task.task_id === selectedTaskId)}
          onClose={() => setSelectedTaskId(null)}
        />
      ) : null}
    </div>
  )
}

export function isNearChatBottom(element: Pick<HTMLElement, 'scrollHeight' | 'scrollTop' | 'clientHeight'>): boolean {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= AUTO_SCROLL_BOTTOM_THRESHOLD_PX
}
